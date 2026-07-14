# k6 Benchmark Report

- Scenario: `health`
- Environment: `production-cloudrun`
- Target: 20 req/s for 1 minute
- Duration: n/a

## Results

- Total requests: `1197`
- Request rate: `19.9 req/s`
- Error rate: `0.00%`
- Avg latency: `147.2 ms`
- Median latency: `145.9 ms`
- P95 latency: `153.8 ms`
- P99 latency: `164.1 ms`
- Max latency: `372.7 ms`
- Peak VUs: `5`

## Bottleneck Hints

- Application processing time dominates request latency.

## Notes

Cloud Run us-central1, client in Europe (RTT ~145ms dominates latency). Rate limiter temporarily raised, then restored.
