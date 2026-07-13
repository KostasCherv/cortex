# k6 Benchmark Report

- Scenario: `health`
- Environment: `production-cloudrun`
- Target: 200 req/s for 1 minute
- Duration: n/a

## Results

- Total requests: `12000`
- Request rate: `199.5 req/s`
- Error rate: `0.00%`
- Avg latency: `147.4 ms`
- Median latency: `146.2 ms`
- P95 latency: `155.4 ms`
- P99 latency: `165.7 ms`
- Max latency: `3154.3 ms`
- Peak VUs: `32`

## Bottleneck Hints

- Application processing time dominates request latency.

## Notes

10x load, zero errors, flat p95 vs 20 req/s. Max 3.15s reflects one instance cold start.
