# Evaluation model (release v1.1)

`evaluation_model.py` reproduces every number in Section IV of
*Big Data Architecture for Large Organizations: Incorporating Generative AI and Agentic AI*:
- Table II (modelled monthly cost), with the caching, routing and storage-tiering ablations.
- Table III (illustrative latency budget), with the agentic step decomposition.
- The throughput arithmetic.
- Table V (weighted provider scores), with weight sensitivity.
- Every sensitivity figure quoted in the text.

```bash
python3 evaluation_model.py
```

It needs only the Python 3 standard library. The output is deterministic: the Monte Carlo draws use seed 2026.

## Assumptions

All assumptions live at the top of the script, so you can re-parameterise the model with your own workload:
- `WORKLOAD`: 30-day month, 1 M RAG queries/day, 100 k agentic tasks/day × 6 calls. Each call is 4,000 input / 600 output tokens. 5 PB is tiered 0.5 / 1.5 / 3.0 PB.
- `PROVIDERS`: model, region and storage class, with USD list prices.
- `ROUTING`: 80 % of RAG and 50 % of agent calls go to the small model.
- `CACHED_INPUT_FRACTION`: 0.5.
- `LATENCY`: stage budgets, decode speeds, tool-call budget, and the replica operating point.

| Provider | Flagship ($/M in/out) | Small ($/M in/out) | Cached input | Storage hot / warm / archive ($/GB-month) |
|---|---|---|---|---|
| AWS (us-east-1, Bedrock global profile) | Claude Sonnet 4.5, 3.00 / 15.00 | Claude Haiku 4.5, 1.00 / 5.00 | 0.1× | S3 Standard 0.023 / Standard-IA 0.0125 / Glacier Deep Archive 0.00099 |
| GCP (us-central1) | Gemini 2.5 Pro, 1.25 / 10.00 | Gemini 2.5 Flash, 0.30 / 2.50 | 0.1× | Standard 0.020 / Nearline 0.010 / Archive 0.0012 |
| Azure (West US 2; OpenAI Global Standard) | GPT-4o 2024-11-20, 2.50 / 10.00 | GPT-4o mini, 0.15 / 0.60 | 0.5× | ADLS Gen2 Hot 0.0184 / Cool 0.010 / Archive 0.00099 (LRS) |

These are the mid-2026 list prices the paper cites (pricing pages accessed June 2026). They were re-checked on 28 October 2026 against the AWS Price List API, Google's Vertex AI pricing page and the Azure Retail Prices API, and were unchanged.

### Simplifications
- First-tier storage rates are applied to all volume. S3 Standard and Azure Hot get cheaper above 50 TB, so this overstates the hot class by about 4 % in the tiered layout and by about 9 % in the all-hot ablation. With volume tiers, tiering cuts the storage line by 68–71 % instead of 70–71 %; the headline reductions are unchanged.
- Cache-write premiums, cache storage fees, archive retrieval and early-deletion fees, batch tiers and negotiated discounts are all ignored.
- Ingestion/transformation, retrieval infrastructure and governance are top-down sizing estimates. They are identical in both configurations.

### What the outputs are, and are not
- **Cost:** a modelled bill, not a measured one.
- **Routing:** the routing shares are admissible only if the routed configuration meets your task-quality floor. The model does not measure quality.
- **Latency:** the totals are illustrative budgets. Adding stage P95s gives the end-to-end P95 only if the stages are comonotonic; otherwise the sum is usually conservative.
- **Throughput:** 8 requests/s per replica is an assumed operating point, not a benchmark.
