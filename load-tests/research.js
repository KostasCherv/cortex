import http from "k6/http";

import {
  baseUrl,
  buildOptions,
  checkStatus,
  jsonHeaders,
  numberEnv,
  optionalEnv,
  requiredEnv,
} from "./lib.js";

const sessionId = requiredEnv("BENCHMARK_SESSION_ID");
const token = requiredEnv("BENCHMARK_BEARER_TOKEN");

export const options = buildOptions({
  scenario: "research",
  rate: numberEnv("BENCHMARK_RATE", 1),
  duration: optionalEnv("BENCHMARK_DURATION", "1m"),
  preAllocatedVUs: numberEnv("BENCHMARK_PREALLOCATED_VUS", 2),
  maxVUs: numberEnv("BENCHMARK_MAX_VUS", 6),
  p95Ms: numberEnv("BENCHMARK_P95_MS", 4000),
  maxErrorRate: numberEnv("BENCHMARK_MAX_ERROR_RATE", 0.01),
});

export default function () {
  const payload = JSON.stringify({
    query: optionalEnv(
      "BENCHMARK_RESEARCH_QUERY",
      "Summarize the key risks and opportunities for this topic with grounded sources."
    ),
  });
  const response = http.post(`${baseUrl()}/sessions/${sessionId}/research`, payload, {
    headers: jsonHeaders(token),
  });
  checkStatus(response);
}
