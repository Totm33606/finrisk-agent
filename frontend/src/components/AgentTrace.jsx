import { CheckCircle2, Loader2, TerminalSquare, XCircle } from "lucide-react";

const TOOL_LABELS = {
  get_credit_score: "SCORE",
  get_shap_explanation: "EXPLAIN",
  simulate_financial_scenario: "SIMULATE",
};

/**
 * Renders the agent's tool-call trajectory as a stamped teletype feed —
 * each MCP tool call "prints" a line with a sequence number, a monospace
 * timestamp-style tag, its inputs, and a condensed result. This is the
 * literal "chain of thought" surface: what the agent asked the MCP server,
 * in the order it asked it, updated live as steps arrive.
 */
export default function AgentTrace({ steps, isRunning }) {
  return (
    <div className="rounded-lg border border-ink-600 bg-ink-900 h-full flex flex-col overflow-hidden">
      <div className="flex items-center gap-2 px-6 py-4 border-b border-ink-700">
        <TerminalSquare className="w-4 h-4 text-amber-500" strokeWidth={1.75} />
        <h3 className="font-display text-lg text-paper-100">Agent reasoning trace</h3>
        {isRunning && <Loader2 className="w-4 h-4 text-amber-500 animate-spin ml-auto" />}
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3 ledger-bg">
        {steps.length === 0 && !isRunning && (
          <p className="font-body text-sm text-paper-500">
            No tool calls yet — ask a question to watch the agent consult the MCP server live.
          </p>
        )}

        {steps.map((step) => (
          <TraceLine key={step.step_index} step={step} />
        ))}

        {isRunning && (
          <div className="flex items-center gap-2 font-mono text-xs text-paper-500 animate-print-in">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            awaiting next tool call…
          </div>
        )}
      </div>
    </div>
  );
}

function TraceLine({ step }) {
  const ok = step.status === "success";
  const label = TOOL_LABELS[step.tool_name] ?? step.tool_name.toUpperCase();

  return (
    <div className="animate-print-in border-l-2 border-ink-700 pl-3 py-1">
      <div className="flex items-center gap-2 font-mono text-[11px] text-paper-500">
        <span className="text-amber-500">#{String(step.step_index + 1).padStart(2, "0")}</span>
        <span className="uppercase tracking-wider">{label}</span>
        {ok ? (
          <CheckCircle2 className="w-3 h-3 text-signal-safe" />
        ) : (
          <XCircle className="w-3 h-3 text-signal-critical" />
        )}
      </div>
      <p className="font-mono text-[12px] text-paper-300 mt-1 break-words">
        {formatArgs(step.tool_input)}
      </p>
      <p className="font-body text-sm text-paper-100 mt-1">{step.tool_output_summary}</p>
    </div>
  );
}

function formatArgs(input) {
  const entries = Object.entries(input ?? {});
  if (entries.length === 0) return "()";
  return "(" + entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ") + ")";
}
