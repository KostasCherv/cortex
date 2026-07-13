# k6 Benchmark Report

- Scenario: `health`
- Environment: `local-dev`
- Target: 20 req/s for 1 minute
- Duration: n/a

## Results

- Total requests: `1201`
- Request rate: `20.0 req/s`
- Error rate: `0.00%`
- Avg latency: `3.8 ms`
- Median latency: `3.9 ms`
- P95 latency: `5.2 ms`
- P99 latency: `6.7 ms`
- Max latency: `14.1 ms`
- Peak VUs: `1`

## Bottleneck Hints

- Application processing time dominates request latency.

## Notes

Local liveness baseline; RATE_LIMIT_DEFAULT raised for benchmark.
