# k6 Benchmark Report

- Scenario: `health`
- Environment: `local-dev`
- Target: 200 req/s for 1 minute
- Duration: n/a

## Results

- Total requests: `12000`
- Request rate: `200.0 req/s`
- Error rate: `0.00%`
- Avg latency: `2.0 ms`
- Median latency: `1.9 ms`
- P95 latency: `3.2 ms`
- P99 latency: `4.1 ms`
- Max latency: `12.7 ms`
- Peak VUs: `1`

## Bottleneck Hints

- Application processing time dominates request latency.

## Notes

10x baseline load; rate limiter raised for benchmark.
