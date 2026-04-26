import { useMemo, useRef, useState, useEffect } from "react";

// ── Zone colours ────────────────────────────────────────────────────────────
const ZONE_COLORS = {
  zone_c: {
    bg:     "linear-gradient(135deg, rgba(226,75,74,0.18) 0%, rgba(163,45,45,0.12) 100%)",
    border: "rgba(226,75,74,0.45)",
    dot:    "#E24B4A",
    label:  "#f7a0a0",
    badge:  "rgba(226,75,74,0.15)",
  },
  zone_b: {
    bg:     "linear-gradient(135deg, rgba(239,159,39,0.18) 0%, rgba(186,117,23,0.12) 100%)",
    border: "rgba(239,159,39,0.45)",
    dot:    "#EF9F27",
    label:  "#f5cc7e",
    badge:  "rgba(239,159,39,0.15)",
  },
  green: {
    bg:     "linear-gradient(135deg, rgba(29,158,117,0.20) 0%, rgba(59,109,17,0.14) 100%)",
    border: "rgba(29,158,117,0.55)",
    dot:    "#1D9E75",
    label:  "#6ee7b7",
    badge:  "rgba(29,158,117,0.15)",
  },
  // fallback alias — backend may send 'calibrated'
  calibrated: {
    bg:     "linear-gradient(135deg, rgba(29,158,117,0.20) 0%, rgba(59,109,17,0.14) 100%)",
    border: "rgba(29,158,117,0.55)",
    dot:    "#1D9E75",
    label:  "#6ee7b7",
    badge:  "rgba(29,158,117,0.15)",
  },
};

const ZONE_LABELS = {
  zone_c:     "Zone C — High confidence, wrong",
  zone_b:     "Zone B — Uncertain, wrong",
  green:      "Calibrated — Correct",
  calibrated: "Calibrated — Correct",
};

const zoneKey = (z) => (z === "calibrated" ? "green" : z) || "zone_b";

// ── Pulse animation (injected once) ─────────────────────────────────────────
const PULSE_STYLE_ID = "evoai-pulse-keyframes";
if (typeof document !== "undefined" && !document.getElementById(PULSE_STYLE_ID)) {
  const style = document.createElement("style");
  style.id = PULSE_STYLE_ID;
  style.textContent = `
    @keyframes evoai-pulse-green {
      0%, 100% { box-shadow: 0 0 0 0 rgba(29,158,117,0.55), 0 0 8px rgba(29,158,117,0.3); }
      50%       { box-shadow: 0 0 0 4px rgba(29,158,117,0.0), 0 0 18px rgba(29,158,117,0.7); }
    }
  `;
  document.head.appendChild(style);
}

// ── Card style ──────────────────────────────────────────────────────────────
const cardStyle = (colors, isPulsing) => ({
  background:     colors.bg,
  border:         `1px solid ${colors.border}`,
  borderRadius:   "10px",
  padding:        "10px 12px",
  minWidth:       "130px",
  cursor:         "default",
  transition:     "transform 0.18s ease, box-shadow 0.18s ease",
  backdropFilter: "blur(4px)",
  animation:      isPulsing ? "evoai-pulse-green 1.5s ease-in-out infinite" : "none",
});

// ── Snapshot history max ─────────────────────────────────────────────────────
const MAX_SNAPSHOTS = 30;

export default function CalibrationMap({ nodes }) {
  // ── Timeline scrubber state ───────────────────────────────────────────────
  const snapshotsRef = useRef([]);          // rolling array of node snapshots
  const [scrubPos, setScrubPos]   = useState(MAX_SNAPSHOTS - 1);
  const [isScrubbing, setIsScrubbing] = useState(false);

  // Track which node keys just transitioned to green this render
  const prevNodesRef = useRef({});
  const [justGreen, setJustGreen] = useState(new Set());

  // Push latest live snapshot into history
  useEffect(() => {
    if (!nodes || nodes.length === 0) return;
    snapshotsRef.current = [
      ...snapshotsRef.current.slice(-(MAX_SNAPSHOTS - 1)),
      nodes,
    ];
    // If not scrubbing, keep scrubber at live end
    if (!isScrubbing) {
      setScrubPos(snapshotsRef.current.length - 1);
    }

    // Detect newly-green nodes for pulse
    const newGreen = new Set();
    const prevMap = prevNodesRef.current;
    nodes.forEach((n) => {
      const nz = zoneKey(n.zone);
      if (nz === "green" && prevMap[n.key] !== "green") {
        newGreen.add(n.key);
      }
    });
    prevNodesRef.current = Object.fromEntries(nodes.map((n) => [n.key, zoneKey(n.zone)]));
    if (newGreen.size > 0) {
      setJustGreen(newGreen);
      // Clear pulse after 6 seconds (4 cycles)
      setTimeout(() => setJustGreen(new Set()), 6000);
    }
  }, [nodes, isScrubbing]);

  // Nodes to display: live or historical
  const displayNodes = useMemo(() => {
    const snaps = snapshotsRef.current;
    if (snaps.length === 0) return nodes || [];
    const idx = Math.min(scrubPos, snaps.length - 1);
    return snaps[idx] || nodes || [];
  }, [scrubPos, nodes]);

  const sorted = useMemo(
    () =>
      [...(displayNodes || [])].sort((a, b) => {
        const order = { zone_c: 0, zone_b: 1, calibrated: 2, green: 2 };
        return (order[a.zone] ?? 2) - (order[b.zone] ?? 2);
      }),
    [displayNodes]
  );

  const counts = useMemo(
    () => ({
      zone_c: (displayNodes || []).filter((n) => n.zone === "zone_c").length,
      zone_b: (displayNodes || []).filter((n) => n.zone === "zone_b").length,
      green:  (displayNodes || []).filter((n) => n.zone === "green" || n.zone === "calibrated").length,
    }),
    [displayNodes]
  );

  const isLive = scrubPos >= snapshotsRef.current.length - 1;

  return (
    <div
      style={{
        border:          "1px solid rgba(255,255,255,0.08)",
        borderRadius:    "12px",
        overflow:        "hidden",
        background:      "rgba(22,27,34,0.8)",
        backdropFilter:  "blur(8px)",
        height:          "100%",
        display:         "flex",
        flexDirection:   "column",
      }}
    >
      {/* ── Header ── */}
      <div
        style={{
          padding:       "14px 16px",
          borderBottom:  "1px solid rgba(255,255,255,0.07)",
          display:       "flex",
          alignItems:    "center",
          gap:           "10px",
          flexShrink:    0,
        }}
      >
        <span style={{ fontSize: "13px", fontWeight: "700", color: "#e6edf3", letterSpacing: "-0.01em" }}>
          Calibration Map
        </span>
        {!isLive && (
          <span style={{ fontSize: "10px", color: "#EF9F27", background: "rgba(239,159,39,0.15)", padding: "1px 7px", borderRadius: "10px", border: "1px solid rgba(239,159,39,0.3)" }}>
            ⏪ History
          </span>
        )}
        <div style={{ marginLeft: "auto", display: "flex", gap: "8px" }}>
          {[["zone_c", "#E24B4A", "rgba(226,75,74,0.15)", `C: ${counts.zone_c}`],
            ["zone_b", "#EF9F27", "rgba(239,159,39,0.15)", `B: ${counts.zone_b}`],
            ["green",  "#1D9E75", "rgba(29,158,117,0.15)", `✓ ${counts.green}`]
          ].map(([z, color, bg, label]) => (
            <span key={z} style={{ fontSize: "11px", fontWeight: "600", padding: "2px 8px", borderRadius: "20px", background: bg, color, border: `1px solid ${color}33` }}>
              {label}
            </span>
          ))}
        </div>
      </div>

      {/* ── Zone legend ── */}
      <div style={{ padding: "8px 16px", borderBottom: "1px solid rgba(255,255,255,0.05)", display: "flex", gap: "16px", flexShrink: 0 }}>
        {Object.entries(ZONE_LABELS).filter(([z]) => z !== "calibrated").map(([zone, label]) => {
          const c = ZONE_COLORS[zone];
          return (
            <div key={zone} style={{ display: "flex", alignItems: "center", gap: "5px" }}>
              <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: c.dot, flexShrink: 0 }} />
              <span style={{ fontSize: "10px", color: "#8b949e" }}>{label}</span>
            </div>
          );
        })}
      </div>

      {/* ── Nodes grid ── */}
      <div
        style={{
          padding:      "12px",
          display:      "flex",
          flexWrap:     "wrap",
          gap:          "8px",
          overflowY:    "auto",
          flex:         1,
          alignContent: "flex-start",
        }}
      >
        {sorted.length === 0 && (
          <div style={{ width: "100%", padding: "40px 20px", textAlign: "center", color: "#8b949e", fontSize: "13px" }}>
            No calibration nodes yet.
            <br />
            <span style={{ fontSize: "11px", opacity: 0.7 }}>Click "Run step" to begin training.</span>
          </div>
        )}
        {sorted.map((node) => {
          const nz     = zoneKey(node.zone);
          const colors = ZONE_COLORS[nz] || ZONE_COLORS.zone_b;
          const isPulsing = justGreen.has(node.key) && nz === "green";
          const visits = node.visits ?? node.visit_count ?? 0;
          const confAvg = typeof node.avg_confidence === "number"
            ? node.avg_confidence.toFixed(1)
            : typeof node.confidence_avg === "number"
            ? node.confidence_avg.toFixed(1)
            : "—";
          const streak = node.correct_streak ?? 0;

          return (
            <div
              key={node.key}
              id={`node-${node.key?.replace(/::/g, "-")}`}
              title={`${ZONE_LABELS[nz] || nz}\nKey: ${node.key}\nVisits: ${visits}\nAvg confidence: ${confAvg}\nCorrect streak: ${streak}`}
              style={cardStyle(colors, isPulsing)}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "scale(1.04)";
                if (!isPulsing) e.currentTarget.style.boxShadow = `0 4px 20px ${colors.dot}33`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "scale(1)";
                if (!isPulsing) e.currentTarget.style.boxShadow = "none";
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: colors.dot, flexShrink: 0, boxShadow: `0 0 6px ${colors.dot}` }} />
                <span style={{ fontSize: "11px", fontWeight: "700", color: colors.label }}>
                  {node.topic}
                </span>
              </div>
              <div style={{ fontSize: "10px", color: colors.label, opacity: 0.75, marginBottom: "2px" }}>{node.question_type}</div>
              <div style={{ fontSize: "10px", color: colors.label, opacity: 0.65, marginBottom: "4px" }}>{node.difficulty_tier}</div>
              {visits > 0 && (
                <div style={{ display: "flex", gap: "6px", alignItems: "center", marginTop: "2px" }}>
                  <span style={{ fontSize: "9px", color: colors.label, opacity: 0.55, background: colors.badge, padding: "1px 5px", borderRadius: "4px" }}>
                    {visits} visit{visits !== 1 ? "s" : ""}
                  </span>
                  <span style={{ fontSize: "9px", color: colors.label, opacity: 0.5 }}>conf {confAvg}</span>
                  {streak >= 1 && (
                    <span style={{ fontSize: "9px", color: colors.dot, opacity: 0.8 }}>🔥{streak}</span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Timeline scrubber ── */}
      <div
        style={{
          padding:      "8px 16px 10px",
          borderTop:    "1px solid rgba(255,255,255,0.05)",
          flexShrink:   0,
          display:      "flex",
          alignItems:   "center",
          gap:          "10px",
        }}
      >
        <span style={{ fontSize: "9px", color: "#8b949e", whiteSpace: "nowrap" }}>
          t=1
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, snapshotsRef.current.length - 1)}
          value={scrubPos}
          onChange={(e) => {
            const v = Number(e.target.value);
            setScrubPos(v);
            setIsScrubbing(v < snapshotsRef.current.length - 1);
          }}
          style={{ flex: 1, accentColor: "#1D9E75", cursor: "pointer" }}
        />
        <span style={{ fontSize: "9px", color: isLive ? "#1D9E75" : "#8b949e", whiteSpace: "nowrap", minWidth: "28px", textAlign: "right" }}>
          {isLive ? "live" : `t=${scrubPos + 1}`}
        </span>
        {!isLive && (
          <button
            onClick={() => { setScrubPos(snapshotsRef.current.length - 1); setIsScrubbing(false); }}
            style={{ fontSize: "9px", padding: "2px 7px", borderRadius: "6px", background: "rgba(29,158,117,0.15)", color: "#1D9E75", border: "1px solid rgba(29,158,117,0.3)", cursor: "pointer" }}
          >
            Live
          </button>
        )}
      </div>
    </div>
  );
}
