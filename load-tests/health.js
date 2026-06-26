import http from "k6/http";

import { baseUrl, buildOptions, checkStatus, numberEnv, optionalEnv } from "./lib.js";

export const options = buildOptions({
  scenario: "health",
  rate: numberEnv("BENCHMARK_RATE", 20),
  duration: optionalEnv("BENCHMARK_DURATION", "1m"),
  preAllocatedVUs: numberEnv("BENCHMARK_PREALLOCATED_VUS", 4),
  maxVUs: numberEnv("BENCHMARK_MAX_VUS", 16),
  p95Ms: numberEnv("BENCHMARK_P95_MS", 300),
  maxErrorRate: numberEnv("BENCHMARK_MAX_ERROR_RATE", 0.01),
});

export default function () {
  const response = http.get(`${baseUrl()}/health`);
  checkStatus(response);
}
