import http from "k6/http";

import {
  baseUrl,
  boolEnv,
  buildOptions,
  checkStatus,
  jsonHeaders,
  numberEnv,
  optionalEnv,
  requiredEnv,
} from "./lib.js";

const internalSecret = requiredEnv("BENCHMARK_INTERNAL_SECRET");

export const options = buildOptions({
  scenario: "internal_agent_loop",
  rate: numberEnv("BENCHMARK_RATE", 1),
  duration: optionalEnv("BENCHMARK_DURATION", "30s"),
  preAllocatedVUs: numberEnv("BENCHMARK_PREALLOCATED_VUS", 2),
  maxVUs: numberEnv("BENCHMARK_MAX_VUS", 6),
  p95Ms: numberEnv("BENCHMARK_P95_MS", 5000),
  maxErrorRate: numberEnv("BENCHMARK_MAX_ERROR_RATE", 0.01),
});

export default function () {
  const payload = JSON.stringify({
    message: optionalEnv(
      "BENCHMARK_CHAT_MESSAGE",
      "Summarize the available context in two sentences."
    ),
    bind_tools: boolEnv("BENCHMARK_BIND_TOOLS", false),
    allow_web_search: boolEnv("BENCHMARK_ALLOW_WEB_SEARCH", false),
    rag_context: optionalEnv(
      "BENCHMARK_RAG_CONTEXT",
      "Sample document context for benchmarking."
    ),
    user_memory_context: optionalEnv("BENCHMARK_USER_MEMORY_CONTEXT", ""),
    system_instructions: optionalEnv("BENCHMARK_SYSTEM_INSTRUCTIONS", "Keep answers brief."),
  });

  const headers = jsonHeaders(internalSecret);
  headers.Authorization = `Bearer ${internalSecret}`;

  const response = http.post(`${baseUrl()}/internal/benchmark/agent-loop`, payload, {
    headers,
  });
  checkStatus(response);
}
