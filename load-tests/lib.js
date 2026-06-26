import { check } from "k6";

export function requiredEnv(name) {
  const value = __ENV[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

export function optionalEnv(name, fallback) {
  const value = __ENV[name];
  return value === undefined || value === "" ? fallback : value;
}

export function numberEnv(name, fallback) {
  const raw = optionalEnv(name, String(fallback));
  const value = Number(raw);
  if (Number.isNaN(value)) {
    throw new Error(`Expected numeric env var for ${name}, got ${raw}`);
  }
  return value;
}

export function boolEnv(name, fallback) {
  const raw = optionalEnv(name, fallback ? "true" : "false").toLowerCase();
  return raw === "1" || raw === "true" || raw === "yes";
}

export function baseUrl() {
  return optionalEnv("BENCHMARK_BASE_URL", "http://127.0.0.1:8000").replace(/\/$/, "");
}

export function jsonHeaders(token) {
  const headers = { "Content-Type": "application/json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

export function buildOptions({
  scenario,
  rate,
  duration,
  preAllocatedVUs,
  maxVUs,
  p95Ms,
  maxErrorRate,
}) {
  return {
    summaryTrendStats: ["avg", "med", "p(95)", "p(99)", "max"],
    thresholds: {
      http_req_failed: [`rate<${maxErrorRate}`],
      http_req_duration: [`p(95)<${p95Ms}`],
    },
    scenarios: {
      [scenario]: {
        executor: "constant-arrival-rate",
        rate,
        timeUnit: "1s",
        duration,
        preAllocatedVUs,
        maxVUs,
      },
    },
  };
}

export function checkStatus(response, allowedStatuses = [200]) {
  const ok = check(response, {
    "status is expected": (r) => allowedStatuses.includes(r.status),
  });
  if (!ok) {
    console.error(`unexpected status=${response.status} body=${String(response.body || "").slice(0, 500)}`);
  }
  return ok;
}
