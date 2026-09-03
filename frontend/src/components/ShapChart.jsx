import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/**
 * Renders local SHAP contributions as a horizontal diverging bar chart:
 * bars pushing right (amber) increase predicted default risk, bars
 * pushing left (teal) decrease it — the sign of the underlying log-odds
 * contribution the tree ensemble assigned to that feature for this client.
 */
export default function ShapChart({ explanation, loading }) {
  if (loading) {
    return (
      <div className="rounded-lg border border-ink-600 bg-ink-900 p-6 h-full animate-pulse">
        <div className="h-4 w-40 bg-ink-700 rounded mb-6" />
        <div className="h-64 bg-ink-700/50 rounded" />
      </div>
    );
  }

  if (!explanation) {
    return (
      <div className="rounded-lg border border-dashed border-ink-600 bg-ink-900/50 p-8 h-full flex items-center justify-center text-paper-500 text-sm">
        SHAP attribution will render once a client is scored.
      </div>
    );
  }

  const data = [...explanation.contributions]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 10)
    .map((c) => ({
      feature: c.feature.replace(/_/g, " "),
      value: c.shap_value,
    }))
    .reverse();

  return (
    <div className="rounded-lg border border-ink-600 bg-ink-900 p-6 h-full flex flex-col">
      <div className="flex items-baseline justify-between mb-1">
        <h3 className="font-display text-lg text-paper-100">
          Attribution — {explanation.client_id}
        </h3>
      </div>
      <p className="font-body text-xs text-paper-500 mb-4">
        Base value {explanation.base_value.toFixed(2)} · top {data.length} drivers by magnitude
      </p>

      <div className="flex-1 min-h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="2 4" stroke="#1C2230" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: "#8D8878", fontSize: 11, fontFamily: "JetBrains Mono" }}
              axisLine={{ stroke: "#2A3142" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="feature"
              width={150}
              tick={{ fill: "#C9C3B3", fontSize: 11, fontFamily: "Inter" }}
              axisLine={false}
              tickLine={false}
            />
            <ReferenceLine x={0} stroke="#2A3142" />
            <Tooltip
              cursor={{ fill: "rgba(237,232,218,0.04)" }}
              contentStyle={{
                background: "#0F131A",
                border: "1px solid #2A3142",
                borderRadius: 6,
                fontFamily: "JetBrains Mono",
                fontSize: 12,
              }}
              formatter={(value) => [Number(value).toFixed(4), "SHAP contribution"]}
            />
            <Bar dataKey="value" radius={3}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.value >= 0 ? "#E8A33D" : "#3FA796"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center gap-4 mt-4 pt-4 border-t border-ink-700">
        <Legend swatch="bg-amber-500" label="Increases risk" />
        <Legend swatch="bg-signal-safe" label="Decreases risk" />
      </div>
    </div>
  );
}

function Legend({ swatch, label }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-2.5 h-2.5 rounded-sm ${swatch}`} />
      <span className="font-body text-[11px] text-paper-500">{label}</span>
    </div>
  );
}
