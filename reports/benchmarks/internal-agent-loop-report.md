# k6 Benchmark Report

- Scenario: `internal-agent-loop`
- Environment: `local-dev`
- Target: 1 req/s for 30 seconds
- Duration: n/a

## Results

- Total requests: `29`
- Request rate: `1.0 req/s`
- Error rate: `0.00%`
- Avg latency: `1298.2 ms`
- Median latency: `1218.4 ms`
- P95 latency: `2007.4 ms`
- P99 latency: `2581.8 ms`
- Max latency: `2773.7 ms`
- Peak VUs: `2`

## Bottleneck Hints

- Application processing time dominates request latency.

## Notes

Real _run_agent_loop path over HTTP, tools unbound, real LLM calls.
