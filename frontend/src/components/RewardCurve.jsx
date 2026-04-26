import { useMemo } from "react";
import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  const val = payload[0]?.value ?? 0;
  const color = val >= 0 ? "#1D9E75" : "#E24B4A";
  return (
    <div
      style={{
        background: "rgba(22,27,34,0.97)",
        border: `1px solid ${color}55`,
        borderRadius: "8px",
        padding: "8px 12px",
        fontSize: "12px",
        color: "#e6edf3",
        backdropFilter: "blur(8px)",
      }}
    >
      <div style={{ color: "#8b949e", marginBottom: "2px" }}>Step {label}</div>
      <div style={{ color, fontWeight: "600" }}>
        {val >= 0 ? "+" : ""}
        {val.toFixed(3)} reward
      </div>
    </div>
  );
};

export default function RewardCurve({ history }) {
  const data = useMemo(
    () =>
      (history || []).map((h, i) => ({
        step: h.step ?? i,
        reward: parseFloat((h.reward || 0).toFixed(4)),
      })),
    [history]
  );

  const mean = useMemo(() => {
    if (!data.length) return 0;
    return parseFloat(
      (data.reduce((s, d) => s + d.reward, 0) / data.length).toFixed(3)
    );
  }, [data]);

  const positive = useMemo(() => data.filter((d) => d.reward >= 0).length, [data]);

  return (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "12px",
        overflow: "hidden",
        background: "rgba(22,27,34,0.8)",
        backdropFilter: "blur(8px)",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <span style={{ fontSize: "13px", fontWeight: "700", color: "#e6edf3", letterSpacing: "-0.01em" }}>
          Reward Curve
        </span>
        <div style={{ marginLeft: "auto", display: "flex", gap: "12px", alignItems: "center" }}>
          {data.length > 0 && (
            <>
              <span style={{ fontSize: "11px", color: "#8b949e" }}>
                {positive}/{data.length} positive
              </span>
              <span
                style={{
                  fontSize: "12px",
                  fontWeight: "700",
                  color: mean >= 0 ? "#1D9E75" : "#E24B4A",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                μ = {mean >= 0 ? "+" : ""}
                {mean}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Chart */}
      <div style={{ padding: "12px 12px 8px" }}>
        {data.length === 0 ? (
          <div
            style={{
              height: "160px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#8b949e",
              fontSize: "13px",
              flexDirection: "column",
              gap: "6px",
            }}
          >
            <span style={{ fontSize: "24px", opacity: 0.4 }}>📈</span>
            No reward data yet. Run a step to begin.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={180}>
            <ComposedChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="step"
                tick={{ fontSize: 9, fill: "#8b949e" }}
                tickLine={false}
                axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
              />
              <YAxis
                tick={{ fontSize: 9, fill: "#8b949e" }}
                tickLine={false}
                axisLine={false}
                domain={[-1, 1]}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine
                y={0}
                stroke="rgba(255,255,255,0.15)"
                strokeWidth={1}
                strokeDasharray="4 4"
              />
              <Bar
                dataKey="reward"
                isAnimationActive={false}
                shape={(props) => {
                  const { x, y, width, height, value } = props;
                  const color = (value ?? 0) >= 0 ? "#1D9E75" : "#E24B4A";
                  const rectH = Math.max(Math.abs(height ?? 0), 1);
                  const rectY = (value ?? 0) >= 0 ? y : y + (height ?? 0);
                  return (
                    <rect
                      x={x}
                      y={rectY}
                      width={width}
                      height={rectH}
                      fill={color}
                      opacity={0.85}
                      rx={1}
                    />
                  );
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Axis labels */}
      {data.length > 0 && (
        <div
          style={{
            padding: "0 16px 10px",
            display: "flex",
            justifyContent: "space-between",
            fontSize: "10px",
            color: "#8b949e",
          }}
        >
          <span>
            <span style={{ color: "#1D9E75" }}>■</span> Positive (calibrating)
          </span>
          <span>
            <span style={{ color: "#E24B4A" }}>■</span> Negative (Zone C penalty)
          </span>
        </div>
      )}
    </div>
  );
}
