import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";

// Shared APPROVE/REVIEW/DECLINE → icon/label/tone mapping, used by both
// ScoreCard (the model's raw recommendation) and the Analyst summary (the
// agent's final decision) so the two badges never visually drift apart.
export const DECISION_META = {
  APPROVE: {
    icon: CheckCircle2,
    label: "Approve",
    tone: "text-signal-safe border-signal-safe/40 bg-signal-safe/10",
  },
  REVIEW: {
    icon: AlertTriangle,
    label: "Review",
    tone: "text-signal-watch border-signal-watch/40 bg-signal-watch/10",
  },
  DECLINE: {
    icon: XCircle,
    label: "Decline",
    tone: "text-signal-critical border-signal-critical/40 bg-signal-critical/10",
  },
};
