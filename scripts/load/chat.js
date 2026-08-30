// k6 load test for the chat endpoint.
//
//   k6 run scripts/load/chat.js
//   k6 run -e SOAK=true scripts/load/chat.js     # 4 hours, lower concurrency
//
// Thresholds fail the run rather than printing a summary nobody reads. A load
// test that always exits 0 is a load test that stops being run.
//
// Time to first token is measured separately from total duration because they
// answer different questions: TTFT is what the user experiences as
// responsiveness, total duration is what capacity planning needs. A system can
// be fine on one and unusable on the other.

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate } from "k6/metrics";

const ttft = new Trend("ttft_ms", true);
const citationRate = new Rate("answers_with_citations");
const refusalRate = new Rate("refusals");

const BASE = __ENV.AGRAG_API_URL || "http://localhost:8000";
const API_KEY = __ENV.AGRAG_API_KEY || "";
const SOAK = __ENV.SOAK === "true";

// A spread of intents, not one question repeated. Hammering a single query
// measures the semantic cache, which is not what anyone means by a load test.
const QUESTIONS = [
  "What is the carry-over limit for annual leave?",
  "How many days of paid sick leave do I get?",
  "Who approves a $5,000 expense?",
  "What is the deploy freeze policy?",
  "How long is an on-call shift and what does it pay?",
  "What notice do directors give?",
  "I spent $5,000 two months ago. Who approves it and can I still claim it?",
  "What is the hotel cap in London?",
  "Summarise the handbook's policy on remote work.",
  "What is the company's dental insurance provider?",
];

export const options = SOAK
  ? {
      // Four hours at modest concurrency. Leaks in an embedding cache or an
      // unbounded conversation buffer are invisible in 60 seconds.
      stages: [
        { duration: "2m", target: 10 },
        { duration: "4h", target: 10 },
        { duration: "2m", target: 0 },
      ],
      thresholds: {
        "http_req_failed": ["rate<0.001"],
        "ttft_ms": ["p(95)<3000"],
      },
    }
  : {
      stages: [
        { duration: "30s", target: 10 },
        { duration: "1m", target: 50 },
        { duration: "2m", target: 50 },
        { duration: "30s", target: 0 },
      ],
      thresholds: {
        "http_req_failed": ["rate<0.001"],
        "ttft_ms": ["p(95)<3000"],
        "http_req_duration": ["p(95)<15000"],
        // Not a performance threshold — a correctness one. A system returning
        // fast, uncited answers has failed at the thing it exists to do.
        "answers_with_citations": ["rate>0.85"],
      },
    };

export default function () {
  const question = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];

  const started = Date.now();
  const response = http.post(
    `${BASE}/api/chat`,
    JSON.stringify({ message: question }),
    {
      headers: {
        "Content-Type": "application/json",
        ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
      },
      timeout: "120s",
    },
  );

  const ok = check(response, {
    "status is 200": (r) => r.status === 200,
    "has content": (r) => {
      try {
        return (r.json("content") || "").length > 0;
      } catch {
        return false;
      }
    },
  });

  if (ok && response.status === 200) {
    ttft.add(Date.now() - started);
    let body = {};
    try {
      body = response.json();
    } catch {
      body = {};
    }
    citationRate.add((body.citations || []).length > 0);
    refusalRate.add(body.stop_reason === "guardrail_blocked" || body.stop_reason === "refused");
  }

  // Think time. Back-to-back requests from every VU is a benchmark of the
  // rate limiter, not of the system under a realistic load.
  sleep(Math.random() * 3 + 1);
}

export function handleSummary(data) {
  const m = data.metrics;
  const line = (name, value) => `  ${name.padEnd(28)} ${value}`;

  return {
    stdout: [
      "",
      SOAK ? "SOAK RESULT" : "LOAD RESULT",
      line("requests", m.http_reqs?.values?.count ?? 0),
      line("error rate", `${((m.http_req_failed?.values?.rate ?? 0) * 100).toFixed(3)}%`),
      line("ttft p95", `${(m.ttft_ms?.values?.["p(95)"] ?? 0).toFixed(0)}ms`),
      line("ttft p99", `${(m.ttft_ms?.values?.["p(99)"] ?? 0).toFixed(0)}ms`),
      line("duration p95", `${(m.http_req_duration?.values?.["p(95)"] ?? 0).toFixed(0)}ms`),
      line("with citations", `${((m.answers_with_citations?.values?.rate ?? 0) * 100).toFixed(1)}%`),
      line("refusals", `${((m.refusals?.values?.rate ?? 0) * 100).toFixed(1)}%`),
      "",
    ].join("\n"),
    "scripts/load/summary.json": JSON.stringify(data, null, 2),
  };
}
