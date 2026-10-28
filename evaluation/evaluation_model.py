#!/usr/bin/env python3
"""Analytical evaluation model for "Big Data Architecture for Large
Organizations: Incorporating Generative AI and Agentic AI" (release v1.1).

Reproduces Tables II, III and V of the paper, the cost ablation, and every
sensitivity figure quoted in Section IV. Standard library only; deterministic.

    python3 evaluation_model.py

All figures are model-based estimates from list prices and assumed component
latencies -- not measured bills or benchmarks.
"""
import random

# ---------------------------------------------------------------------------
# Assumptions (Table I of the paper). Prices are USD list prices, mid-2026.
# ---------------------------------------------------------------------------
WORKLOAD = {
    "days": 30,
    "rag_queries_per_day": 1_000_000,
    "tokens_in": 4_000,           # per model call (RAG query or agent step)
    "tokens_out": 600,
    "agent_tasks_per_day": 100_000,
    "calls_per_task": 6,          # sequential model calls per agentic task
    "data_pb": {"hot": 0.5, "warm": 1.5, "archive": 3.0},
    "gb_per_pb": 1_000_000,
}

# Model prices are $ per million tokens (input, output). cache_read is the
# cached-input price as a fraction of the input price. Storage is $/GB-month
# at the first volume tier (volume discounts ignored). The three "other" lines
# are top-down sizing estimates, identical in both configurations.
PROVIDERS = {
    "AWS": {
        "region": "us-east-1 (Bedrock global inference profile)",
        "flagship": ("Claude Sonnet 4.5", 3.00, 15.00),
        "small": ("Claude Haiku 4.5", 1.00, 5.00),
        "cache_read": 0.10,
        "storage": {"hot": ("S3 Standard", 0.023),
                    "warm": ("S3 Standard-IA", 0.0125),
                    "archive": ("S3 Glacier Deep Archive", 0.00099)},
        "other": {"ingest_transform": 75_000, "retrieval": 32_000, "governance": 18_000},
    },
    "GCP": {
        "region": "us-central1",
        "flagship": ("Gemini 2.5 Pro (<=200k ctx)", 1.25, 10.00),
        "small": ("Gemini 2.5 Flash", 0.30, 2.50),
        "cache_read": 0.10,
        "storage": {"hot": ("Standard", 0.020),
                    "warm": ("Nearline", 0.010),
                    "archive": ("Archive", 0.0012)},
        "other": {"ingest_transform": 70_000, "retrieval": 30_000, "governance": 17_000},
    },
    "Azure": {
        "region": "West US 2 (ADLS Gen2); OpenAI Global Standard",
        "flagship": ("GPT-4o (2024-11-20)", 2.50, 10.00),
        "small": ("GPT-4o mini", 0.15, 0.60),
        "cache_read": 0.50,
        "storage": {"hot": ("ADLS Gen2 Hot LRS", 0.0184),
                    "warm": ("ADLS Gen2 Cool LRS", 0.010),
                    "archive": ("ADLS Gen2 Archive LRS", 0.00099)},
        "other": {"ingest_transform": 72_000, "retrieval": 31_000, "governance": 18_000},
    },
}

ROUTING = {"rag_small": 0.80, "agent_small": 0.50}   # share sent to the small model
CACHED_INPUT_FRACTION = 0.50                         # static system/context tokens

# Latency budget (seconds). Stage values are assumed P50/P95 budgets.
LATENCY = {
    "stages": {"Query embedding": (0.015, 0.030),
               "Vector ANN search (500M)": (0.025, 0.060),
               "Re-ranking (cross-encoder)": (0.040, 0.090),
               "AI Gateway + network": (0.030, 0.070),
               "Prefill / TTFT (~4k input)": (0.400, 0.900)},
    "decode_tok_per_s": {"flagship": 60.0, "small": 150.0},
    "tool_call": (0.5, 4.0),        # one tool invocation per agent step
    "replica_rps": 8.0,             # assumed operating point, not a benchmark
    "replicas": (10, 50, 200),
}

# Provider capability scores (Table V).
DIMENSIONS = [  # (dimension, weight, AWS, GCP, Azure)
    ("Data ingestion", 0.08, 5, 4, 4),
    ("Data Ocean, Mesh & Contracts", 0.12, 4, 5, 4),
    ("Transformation & analytics", 0.10, 4, 5, 4),
    ("AI Infrastructure", 0.15, 5, 5, 4),
    ("GenAI platform", 0.13, 5, 4, 4),
    ("Agentic AI", 0.12, 5, 4, 4),
    ("Analytics & BI", 0.08, 4, 5, 5),
    ("AI Trust, Governance & Security", 0.13, 4, 4, 5),
    ("Cost transparency / FinOps", 0.05, 4, 4, 4),
    ("Reliability guidance", 0.04, 5, 4, 5),
]
SEED, DRAWS = 2026, 100_000


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def monthly_mtok():
    """Monthly RAG token volume (millions) and the agentic/RAG call ratio."""
    w = WORKLOAD
    queries = w["rag_queries_per_day"] * w["days"]
    agent_calls = w["agent_tasks_per_day"] * w["calls_per_task"] * w["days"]
    return (queries * w["tokens_in"] / 1e6, queries * w["tokens_out"] / 1e6,
            agent_calls / queries)


def model_cost(model, cache_read, cached, extra_in=0.0):
    """Monthly cost of one RAG-volume of calls on `model`."""
    _, p_in, p_out = model
    m_in, m_out, _ = monthly_mtok()
    f = CACHED_INPUT_FRACTION if cached else 0.0
    return m_in * p_in * (1 - f + f * cache_read) + (m_out * p_out) + extra_in * p_in


def inference(p, rag_small, agent_small, cached, agent_extra_in=0.0, agent_retry=0.0):
    """(RAG, agentic) monthly inference cost. agent_extra_in is extra uncached
    input (M tokens per RAG-volume) from context growth; agent_retry scales
    agentic calls."""
    _, _, ratio = monthly_mtok()
    flag = model_cost(p["flagship"], p["cache_read"], cached)
    small = model_cost(p["small"], p["cache_read"], cached)
    rag = (1 - rag_small) * flag + rag_small * small
    extra = agent_extra_in * ((1 - agent_small) * p["flagship"][1] + agent_small * p["small"][1])
    agent = ratio * ((1 - agent_small) * flag + agent_small * small + extra) * (1 + agent_retry)
    return rag, agent


def storage(p, tiered=True):
    w = WORKLOAD
    if not tiered:  # all data in the hot class
        return sum(w["data_pb"].values()) * w["gb_per_pb"] * p["storage"]["hot"][1]
    return sum(w["data_pb"][t] * w["gb_per_pb"] * p["storage"][t][1] for t in w["data_pb"])


def total(p, rag_small=0.0, agent_small=0.0, cached=False, tiered=True, **kw):
    rag, agent = inference(p, rag_small, agent_small, cached, **kw)
    return storage(p, tiered) + sum(p["other"].values()) + rag + agent


def cost_report():
    print("=" * 78 + "\nTABLE II -- modelled monthly cost (USD)\n" + "=" * 78)
    hdr = f"{'':38}" + "".join(f"{k:>13}" for k in PROVIDERS)
    print(hdr)
    rows = {}
    for k, p in PROVIDERS.items():
        b_rag, b_ag = inference(p, 0, 0, False)
        o_rag, o_ag = inference(p, ROUTING["rag_small"], ROUTING["agent_small"], True)
        base = total(p)
        opt = total(p, ROUTING["rag_small"], ROUTING["agent_small"], True)
        rows[k] = {
            "Data Ocean storage (tiered)": storage(p),
            "Ingestion & transformation": p["other"]["ingest_transform"],
            "AI retrieval infra": p["other"]["retrieval"],
            "Governance & observability": p["other"]["governance"],
            "GenAI/RAG inference - baseline": b_rag,
            "Agentic inference - baseline": b_ag,
            "TOTAL - baseline": base,
            "GenAI/RAG inference - optimised": o_rag,
            "Agentic inference - optimised": o_ag,
            "TOTAL - optimised": opt,
            "Total cost reduction (%)": 100 * (1 - opt / base),
            "Ablation: caching only": total(p, cached=True),
            "Ablation: routing only": total(p, ROUTING["rag_small"], ROUTING["agent_small"]),
            "Storage untiered (all hot)": storage(p, tiered=False),
        }
    for line in next(iter(rows.values())):
        print(f"{line:38}" + "".join(f"{rows[k][line]:>13,.1f}" if "%" in line
                                     else f"{rows[k][line]:>13,.0f}" for k in PROVIDERS))

    print("\nAblation, reduction of total vs baseline (%):")
    for k, p in PROVIDERS.items():
        r = rows[k]
        base, opt = r["TOTAL - baseline"], r["TOTAL - optimised"]
        cache_only, route_only = r["Ablation: caching only"], r["Ablation: routing only"]
        untiered_total = base - r["Data Ocean storage (tiered)"] + r["Storage untiered (all hot)"]
        tier_saving = r["Storage untiered (all hot)"] - r["Data Ocean storage (tiered)"]
        print(f"  {k:6} caching {100*(1-cache_only/base):5.1f}  routing {100*(1-route_only/base):5.1f}"
              f"  both {100*(1-opt/base):5.1f}  routing vs cached-flagship baseline "
              f"{100*(1-opt/cache_only):5.1f}  | tiering saves {tier_saving:,.0f}"
              f" ({100*tier_saving/r['Storage untiered (all hot)']:.1f}% of storage,"
              f" {100*tier_saving/untiered_total:.1f}% of an untiered baseline total)")
    inf = {k: 100 * (1 - (r["GenAI/RAG inference - optimised"] + r["Agentic inference - optimised"])
                     / (r["GenAI/RAG inference - baseline"] + r["Agentic inference - baseline"]))
           for k, r in rows.items()}
    print("  Inference-only reduction: " + ", ".join(f"{k} {v:.1f}" for k, v in inf.items()))

    print("\nSensitivity:")
    for k, p in PROVIDERS.items():
        rs, ag = ROUTING["rag_small"], ROUTING["agent_small"]
        opt_inf = sum(inference(p, rs, ag, True))
        shift = sum(inference(p, rs - 0.1, ag - 0.1, True)) - opt_inf
        # Context growth: +1,000 uncached tokens per step over 6 steps -> mean +2,500 per call.
        n = WORKLOAD["calls_per_task"]
        mean_extra = 1_000 * (n - 1) / 2
        m_in, _, _ = monthly_mtok()
        extra = m_in * mean_extra / WORKLOAD["tokens_in"]
        b_ag = inference(p, 0, 0, False)[1]
        b_ag_grown = inference(p, 0, 0, False, agent_extra_in=extra)[1]
        base_g = total(p, agent_extra_in=extra)
        opt_g = total(p, rs, ag, True, agent_extra_in=extra)
        base_gr = total(p, agent_extra_in=extra, agent_retry=0.10)
        opt_gr = total(p, rs, ag, True, agent_extra_in=extra, agent_retry=0.10)
        print(f"  {k:6} +10pp to flagship: +{shift:,.0f} = {100*shift/opt_inf:.1f}% of optimised inference"
              f" | context growth: agentic baseline +{100*(b_ag_grown/b_ag-1):.1f}%,"
              f" reduction {100*(1-opt_g/base_g):.1f}% (+10% retries: {100*(1-opt_gr/base_gr):.1f}%)")
    opt = {k: total(p, ROUTING["rag_small"], ROUTING["agent_small"], True) for k, p in PROVIDERS.items()}
    print(f"  Uniform discount d leaves every % reduction unchanged. Optimised totals: "
          + ", ".join(f"{k} {v:,.0f}" for k, v in opt.items())
          + f"; Azure/GCP gap {100*(opt['Azure']/opt['GCP']-1):.1f}%")


# ---------------------------------------------------------------------------
# Latency budget and throughput arithmetic
# ---------------------------------------------------------------------------
def latency_report():
    L = LATENCY
    print("\n" + "=" * 78 + "\nTABLE III -- illustrative latency budget (s), P50 / P95\n" + "=" * 78)
    for name, (a, b) in L["stages"].items():
        print(f"  {name:32} {a:6.3f} {b:6.3f}")
    ttft = [sum(v[i] for v in L["stages"].values()) for i in (0, 1)]
    gen = {m: WORKLOAD["tokens_out"] / r for m, r in L["decode_tok_per_s"].items()}
    print(f"  {'Time-to-first-token':32} {ttft[0]:6.2f} {ttft[1]:6.2f}   (sum of stage budgets)")
    for m in ("flagship", "small"):
        print(f"  {'Generation (' + m + ', 600 tok)':32} {gen[m]:6.1f} {gen[m]:6.1f}   (deterministic decode)")
        print(f"  {'RAG E2E (' + m + ')':32} {ttft[0]+gen[m]:6.2f} {ttft[1]+gen[m]:6.2f}")
    mix = (1 - ROUTING["agent_small"]) * gen["flagship"] + ROUTING["agent_small"] * gen["small"]
    step = [ttft[i] + mix + L["tool_call"][i] for i in (0, 1)]
    n = WORKLOAD["calls_per_task"]
    print(f"  {'Agent step: TTFT + mean gen':32} {ttft[0]+mix:6.2f} {ttft[1]+mix:6.2f}   (gen = {mix:.1f} s at the routing mix)")
    print(f"  {'Agent step: tool invocation':32} {L['tool_call'][0]:6.2f} {L['tool_call'][1]:6.2f}")
    print(f"  {'Agentic workflow E2E (%d steps)' % n:32} {n*step[0]:6.1f} {n*step[1]:6.1f}")

    rps = L["replica_rps"]
    print("\nThroughput arithmetic (assumed operating point, not benchmarked):")
    for r in L["replicas"]:
        print(f"  {r:4d} replicas x {rps:.0f} req/s -> {r*rps*86_400/1e6:6.1f} M requests/day")
    for m in ("small", "flagship"):
        e2e = ttft[0] + gen[m]
        print(f"  Little's law, {m}: {rps:.0f} req/s x {e2e:.2f} s = {rps*e2e:.0f} concurrent streams")
    print(f"  Per replica: {rps*WORKLOAD['tokens_out']:,.0f} decode tok/s, "
          f"{rps*WORKLOAD['tokens_in']:,.0f} prefill tok/s "
          f"({rps*WORKLOAD['tokens_in']*(1-CACHED_INPUT_FRACTION):,.0f} after prefix-cache hits)")


# ---------------------------------------------------------------------------
# Provider comparison and weight sensitivity
# ---------------------------------------------------------------------------
def scores_report():
    names = list(PROVIDERS)
    w0 = [d[1] for d in DIMENSIONS]
    S = {k: [d[2 + i] for d in DIMENSIONS] for i, k in enumerate(names)}

    def totals(w):
        return {k: sum(a * b for a, b in zip(w, v)) for k, v in S.items()}

    print("\n" + "=" * 78 + "\nTABLE V -- weighted provider scores\n" + "=" * 78)
    print("  nominal weights:", {k: round(v, 2) for k, v in totals(w0).items()})
    print("  equal weights:  ", {k: round(v, 2) for k, v in totals([0.1] * 10).items()})
    rng = random.Random(SEED)
    for spread in (0.25, 0.50):
        wins = dict.fromkeys(names, 0)
        for _ in range(DRAWS):
            w = [x * rng.uniform(1 - spread, 1 + spread) for x in w0]
            t = totals([x / sum(w) for x in w])
            wins[max(t, key=t.get)] += 1
        print(f"  +/-{int(spread*100)}% weight perturbation, share ranked first:",
              {k: f"{100*v/DRAWS:.1f}%" for k, v in wins.items()})
    wins = dict.fromkeys(names, 0)
    for _ in range(DRAWS):
        g = [rng.expovariate(1.0) for _ in w0]   # flat Dirichlet
        t = totals([x / sum(g) for x in g])
        wins[max(t, key=t.get)] += 1
    print("  uninformative (flat Dirichlet) weights, share ranked first:",
          {k: f"{100*v/DRAWS:.1f}%" for k, v in wins.items()})


if __name__ == "__main__":
    print("Models and rates:")
    for k, p in PROVIDERS.items():
        print(f"  {k}: {p['region']}; flagship {p['flagship']}; small {p['small']}; "
              f"cache-read {p['cache_read']}x; storage "
              + ", ".join(f"{n} {r}" for n, r in p["storage"].values()))
    cost_report()
    latency_report()
    scores_report()
