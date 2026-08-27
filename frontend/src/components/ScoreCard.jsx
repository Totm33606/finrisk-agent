import { CircleDashed } from "lucide-react";
import { DECISION_META } from "../lib/decisionMeta.js";

const BAND_STYLES = {
  LOW: { text: "text-signal-safe", ring: "stroke-signal-safe", label: "Low" },
  MEDIUM: { text: "text-signal-watch", ring: "stroke-signal-watch", label: "Medium" },
  HIGH: { text: "text-signal-risk", ring: "stroke-signal-risk", label: "High" },
  CRITICAL: { text: "text-signal-critical", ring: "stroke-signal-critical", label: "Critical" },
};

/**
 * The primary "instrument reading" for a client: probability of default
 * rendered as a partial-ring gauge, the discretized risk band, and the
 * model's recommendation. Deliberately reads like a single precise
 * measurement rather than a dashboard KPI tile.
 */
export default function ScoreCard({ score, loading }) {
  if (loading) {
    return (
      <div className="rounded-lg border border-ink-600 bg-ink-900 p-6 animate-pulse">
        <div className="h-4 w-24 bg-ink-700 rounded mb-6" />
        <div className="h-32 w-32 rounded-full bg-ink-700 mx-auto" />
      </div>
    );
  }

  if (!score) {
    return (
      <div className="rounded-lg border border-dashed border-ink-600 bg-ink-900/50 p-8 flex flex-col items-center gap-3 text-paper-500">
        <CircleDashed className="w-8 h-8" strokeWidth={1.5} />
        <p className="font-body text-sm">Select a client to pull the current instrument reading.</p>
      </div>
    );
  }

  const pct = score.probability_default * 100;
  const band = BAND_STYLES[score.risk_band] ?? BAND_STYLES.MEDIUM;
  const decision = DECISION_META[score.recommendation] ?? DECISION_META.REVIEW;
  const DecisionIcon = decision.icon;

  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const dash = (Math.min(pct, 100) / 100) * circumference;

  return (
    <div className="rounded-lg border border-ink-600 bg-ink-900 p-6">
      <div className="flex items-baseline justify-between mb-6">
        <h2 className="font-display text-lg text-paper-100">{score.client_id}</h2>
        <span className="font-mono text-xs text-paper-500">v{score.model_version}</span>
      </div>

      <div className="relative w-36 h-36 mx-auto mb-6">
        <svg viewBox="0 0 130 130" className="w-full h-full -rotate-90">
          <circle
            cx="65"
            cy="65"
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className="text-ink-700"
          />
          <circle
            cx="65"
            cy="65"
            r={radius}
            fill="none"
            strokeWidth="8"
            strokeLinecap="round"
            className={band.ring}
            strokeDasharray={`${dash} ${circumference}`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-3xl text-paper-100 tabular-nums">{pct.toFixed(1)}%</span>
          <span className="font-body text-[11px] uppercase tracking-wider text-paper-500">
            probability of default
          </span>
        </div>
      </div>

      <div className="flex items-center justify-between mb-4">
        <span className="font-body text-xs uppercase tracking-wider text-paper-500">Risk band</span>
        <span className={`font-mono text-sm font-medium ${band.text}`}>{band.label}</span>
      </div>

      <div className={`flex items-center gap-2 rounded-md border px-3 py-2 ${decision.tone}`}>
        <DecisionIcon className="w-4 h-4 shrink-0" strokeWidth={2} />
        <span className="font-body text-sm font-medium">{decision.label}</span>
        <span className="ml-auto font-mono text-[11px] text-paper-500">
          threshold {(score.decision_threshold * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
