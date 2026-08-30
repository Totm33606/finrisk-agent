import { useState } from "react";
import { Send } from "lucide-react";
import ScoreCard from "./components/ScoreCard.jsx";
import ShapChart from "./components/ShapChart.jsx";
import AgentTrace from "./components/AgentTrace.jsx";
import { DECISION_META } from "./lib/decisionMeta.js";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080";

// The MCP server only serves clients from the held-out test split (see
// ClientStore in scoring_service.py) — not the full synthetic dataset — so
// these three are verified present in models/holdout_test.parquet from an
// actual `make train` run, not just guessed to be "somewhere in range".
// Re-verify against a fresh holdout_test.parquet if you regenerate the
// dataset with a different --n-clients/seed.
const EXAMPLE_CLIENTS = ["SME-000001", "SME-000007", "SME-000009"];
const EXAMPLE_QUESTIONS = [
  "Should we approve this client's credit request?",
  "What if their revenue dropped 20%?",
  "Why is this client flagged as high risk?",
];

export default function App() {
  const [clientId, setClientId] = useState("SME-000123");
  const [question, setQuestion] = useState("Should we approve this client's credit request?");
  const [result, setResult] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);

  async function runAnalysis(e) {
    e.preventDefault();
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: clientId, question }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Request failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message ?? "Something went wrong reaching the agent API.");
    } finally {
      setIsRunning(false);
    }
  }

  // The score/explanation objects rendered by ScoreCard/ShapChart are
  // reconstructed from the agent's tool trace so the UI reflects exactly
  // what the agent itself saw — not a second, independent API call.
  const score = extractLatestToolPayload(result, "get_credit_score");
  const explanation = extractLatestToolPayload(result, "get_shap_explanation");

  return (
    <div className="min-h-screen bg-ink-950">
      <header className="border-b border-ink-700 px-8 py-5">
        <div className="flex items-baseline gap-3">
          <h1 className="font-display text-2xl text-paper-100">FinRisk-Agent</h1>
          <span className="font-mono text-xs text-paper-500 uppercase tracking-widest">
            Credit Decisioning Console
          </span>
        </div>
      </header>

      <form
        onSubmit={runAnalysis}
        className="flex flex-wrap gap-3 px-8 py-5 border-b border-ink-700"
      >
        <label htmlFor="client-id" className="sr-only">
          Client ID
        </label>
        <input
          id="client-id"
          value={clientId}
          onChange={(e) => setClientId(e.target.value)}
          placeholder="Client ID, e.g. SME-000123"
          className="font-mono text-sm bg-ink-900 border border-ink-600 rounded-md px-3 py-2 text-paper-100 w-48 placeholder:text-paper-500"
        />
        <label htmlFor="analyst-question" className="sr-only">
          Question for the analyst agent
        </label>
        <input
          id="analyst-question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask the analyst agent…"
          className="font-body text-sm bg-ink-900 border border-ink-600 rounded-md px-3 py-2 text-paper-100 flex-1 min-w-[240px] placeholder:text-paper-500"
        />
        <button
          type="submit"
          disabled={isRunning}
          className="flex items-center gap-2 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-ink-950 font-body font-medium text-sm rounded-md px-4 py-2 transition-colors"
        >
          <Send className="w-4 h-4" />
          {isRunning ? "Analyzing…" : "Run analysis"}
        </button>
      </form>

      <div className="flex flex-wrap items-center gap-2 px-8 py-3 border-b border-ink-700 bg-ink-900/40">
        <span className="font-body text-[11px] uppercase tracking-wider text-paper-500">Try</span>
        {EXAMPLE_CLIENTS.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setClientId(id)}
            className="font-mono text-[11px] rounded-full border border-ink-600 px-2.5 py-1 text-paper-300 hover:border-amber-500 hover:text-amber-400 transition-colors"
          >
            {id}
          </button>
        ))}
        <span className="w-px h-4 bg-ink-700 mx-1" aria-hidden="true" />
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => setQuestion(q)}
            className="font-body text-[11px] rounded-full border border-ink-600 px-2.5 py-1 text-paper-300 hover:border-amber-500 hover:text-amber-400 transition-colors max-w-[220px] truncate"
            title={q}
          >
            {q}
          </button>
        ))}
      </div>

      {error && (
        <div className="mx-8 mt-4 rounded-md border border-signal-critical/40 bg-signal-critical/10 px-4 py-3 text-sm text-signal-critical">
          {error}
        </div>
      )}

      <main className="grid grid-cols-1 lg:grid-cols-[320px_1fr_380px] gap-5 p-8">
        <ScoreCard score={score} loading={isRunning && !score} />
        <ShapChart explanation={explanation} loading={isRunning && !explanation} />
        <AgentTrace steps={result?.steps ?? []} isRunning={isRunning} />
      </main>

      {result && (
        <section className="mx-8 mb-8 rounded-lg border border-ink-600 bg-ink-900 overflow-hidden">
          <div className="flex flex-wrap items-center gap-3 px-6 py-4 border-b border-ink-700">
            <h3 className="font-display text-lg text-paper-100">Analyst summary</h3>
            <DecisionBadge decision={result.decision} />
            <span className="ml-auto font-mono text-[11px] text-paper-500">
              {(result.total_latency_ms / 1000).toFixed(1)}s
              {result.langfuse_trace_id && ` · trace ${result.langfuse_trace_id.slice(0, 8)}`}
            </span>
          </div>

          <div className="grid gap-6 px-6 py-5 md:grid-cols-[1fr_260px]">
            <p className="font-body text-sm text-paper-300 leading-relaxed whitespace-pre-wrap">
              {result.summary}
            </p>

            {result.key_drivers?.length > 0 && (
              <div className="md:border-l md:border-ink-700 md:pl-6">
                <h4 className="font-body text-xs uppercase tracking-wider text-paper-500 mb-3">
                  Key drivers
                </h4>
                <ul className="space-y-2">
                  {result.key_drivers.map((driver, i) => (
                    <li key={i} className="flex gap-2 font-body text-sm text-paper-300">
                      <span className="text-amber-500 shrink-0">▸</span>
                      <span>{driver}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}

function DecisionBadge({ decision }) {
  const meta = DECISION_META[decision] ?? DECISION_META.REVIEW;
  const Icon = meta.icon;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 animate-stamp-in ${meta.tone}`}
    >
      <Icon className="w-3.5 h-3.5" strokeWidth={2} />
      <span className="font-mono text-[11px] uppercase tracking-wider font-medium">
        {meta.label}
      </span>
    </span>
  );
}

function extractLatestToolPayload(result, toolName) {
  // Each AgentStep carries the full parsed tool payload (`raw_output`)
  // alongside the condensed summary shown in the trace feed — this pulls
  // the most recent call to `toolName` so ScoreCard/ShapChart render exact
  // model output rather than re-deriving values from display text.
  const step = [...(result?.steps ?? [])].reverse().find((s) => s.tool_name === toolName);
  return step?.raw_output ?? null;
}
