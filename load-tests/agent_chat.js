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

const agentId = requiredEnv("BENCHMARK_AGENT_ID");
const token = requiredEnv("BENCHMARK_BEARER_TOKEN");

export const options = buildOptions({
  scenario: "agent_chat",
  rate: numberEnv("BENCHMARK_RATE", 2),
  duration: optionalEnv("BENCHMARK_DURATION", "1m"),
  preAllocatedVUs: numberEnv("BENCHMARK_PREALLOCATED_VUS", 4),
  maxVUs: numberEnv("BENCHMARK_MAX_VUS", 12),
  p95Ms: numberEnv("BENCHMARK_P95_MS", 10000),
  maxErrorRate: numberEnv("BENCHMARK_MAX_ERROR_RATE", 0.01),
});

export default function () {
  const payload = {
    message: optionalEnv(
      "BENCHMARK_CHAT_MESSAGE",
      "Give me a two-sentence summary of the most relevant grounded context you have."
    ),
  };
  const sessionId = optionalEnv("BENCHMARK_CHAT_SESSION_ID", "");
  if (sessionId) {
    payload.session_id = sessionId;
  }
  const response = http.post(`${baseUrl()}/api/rag/agents/${agentId}/chat`, JSON.stringify(payload), {
    headers: jsonHeaders(token),
  });
  checkStatus(response);
}
