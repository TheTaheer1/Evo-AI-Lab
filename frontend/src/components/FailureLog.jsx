// ── Zone / difficulty colour maps ──────────────────────────────────────────
const ZONE_PILL = {
  zone_c:     { color: "#E24B4A", bg: "rgba(226,75,74,0.15)",    border: "rgba(226,75,74,0.3)" },
  zone_b:     { color: "#EF9F27", bg: "rgba(239,159,39,0.15)",   border: "rgba(239,159,39,0.3)" },
  green:      { color: "#1D9E75", bg: "rgba(29,158,117,0.15)",   border: "rgba(29,158,117,0.3)" },
  calibrated: { color: "#1D9E75", bg: "rgba(29,158,117,0.15)",   border: "rgba(29,158,117,0.3)" },
};

const truncate = (text, n = 120) => {
  if (text == null || text === "") return "";
  const s = String(text);
  return s.length > n ? s.slice(0, n) + "..." : s;
};

/** Non-empty row for the learning panel (after same rules as card body). */
function rowHasContent(f) {
  const ghRaw = (f.gold_hint || "").trim();
  const ghBad =
    !ghRaw ||
    ["UNVERIFIABLE", "NONE", "NULL", "MAJORITY_CORRECT"].includes(
      ghRaw.toUpperCase()
    );
  const correction =
    (f.correction || f.chosen || "").trim() || (ghBad ? "" : ghRaw);
  const studentSaid = (f.student_answer || "").trim();
  const contrastWrong = (f.failure_answer || f.rejected || "").trim();
  const believed = studentSaid || contrastWrong;
  return !!(believed || correction);
}

const BADGE_BG = {
  math:     { color: "#93c5fd", bg: "rgba(96,165,250,0.15)" },
  code:     { color: "#c4b5fd", bg: "rgba(167,139,250,0.15)" },
  logic:    { color: "#f9a8d4", bg: "rgba(236,72,153,0.15)" },
  factual:  { color: "#6ee7b7", bg: "rgba(52,211,153,0.15)" },
  planning: { color: "#fcd34d", bg: "rgba(251,191,36,0.15)" },
};

function SkillBadge({ topic, questionType, difficulty }) {
  const bStyle = BADGE_BG[topic?.toLowerCase()] || BADGE_BG.factual;
  const parts = [topic, questionType, difficulty]
    .filter(Boolean)
    .map((s) => s.toUpperCase())
    .join(" · ");
  return (
    <span
      style={{
        fontSize:       "9px",
        fontWeight:     "700",
        padding:        "2px 7px",
        borderRadius:   "10px",
        background:     bStyle.bg,
        color:          bStyle.color,
        letterSpacing:  "0.04em",
        border:         `1px solid ${bStyle.color}33`,
      }}
    >
      {parts}
    </span>
  );
}

function StatusBadge({ learned }) {
  return learned ? (
    <span
      style={{
        fontSize:     "9px",
        fontWeight:   "700",
        padding:      "2px 7px",
        borderRadius: "10px",
        background:   "rgba(29,158,117,0.15)",
        color:        "#1D9E75",
        border:       "1px solid rgba(29,158,117,0.3)",
        letterSpacing: "0.04em",
      }}
    >
      LEARNED ✓
    </span>
  ) : (
    <span
      style={{
        fontSize:     "9px",
        fontWeight:   "700",
        padding:      "2px 7px",
        borderRadius: "10px",
        background:   "rgba(239,159,39,0.12)",
        color:        "#EF9F27",
        border:       "1px solid rgba(239,159,39,0.3)",
        letterSpacing: "0.04em",
      }}
    >
      PENDING ⏳
    </span>
  );
}

export default function FailureLog({ failures, totalMoments }) {
  const safeFailures = Array.isArray(failures) ? failures : [];
  const total = typeof totalMoments === "number" ? totalMoments : safeFailures.length;

  return (
    <div
      style={{
        border:          "1px solid rgba(255,255,255,0.08)",
        borderRadius:    "12px",
        overflow:        "hidden",
        background:      "rgba(22,27,34,0.8)",
        backdropFilter:  "blur(8px)",
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
        }}
      >
        <span style={{ fontSize: "13px", fontWeight: "700", color: "#e6edf3", letterSpacing: "-0.01em" }}>
          What the Model Learned
        </span>
        <span
          style={{
            marginLeft:   "auto",
            fontSize:     "11px",
            color:        "#8b949e",
            background:   "rgba(139,148,158,0.1)",
            padding:      "2px 8px",
            borderRadius: "10px",
            border:       "1px solid rgba(139,148,158,0.15)",
          }}
        >
          {total} moment{total !== 1 ? "s" : ""}
        </span>
      </div>

      {/* ── Cards ── */}
      <div
        style={{
          maxHeight:      "260px",
          overflowY:      "auto",
          padding:        "8px",
          scrollbarWidth: "thin",
          scrollbarColor: "rgba(255,255,255,0.08) transparent",
        }}
      >
        {safeFailures.length === 0 ? (
          <div style={{ padding: "28px 16px", textAlign: "center", color: "#8b949e", fontSize: "13px" }}>
            <div style={{ fontSize: "24px", marginBottom: "8px", opacity: 0.4 }}>🧠</div>
            No learning moments yet.
            <br />
            <span style={{ fontSize: "11px", opacity: 0.7 }}>
              Run steps to see Zone C/B → calibrated corrections.
            </span>
          </div>
        ) : (
          safeFailures.filter(rowHasContent).map((f, i) => {
            const zStyle = ZONE_PILL[f.zone] || ZONE_PILL.zone_b;
            const conf =
              typeof f.confidence === "number"
                ? f.confidence
                : f.confidence != null
                  ? Number(f.confidence)
                  : null;
            const confLabel =
              conf != null && !Number.isNaN(conf) ? conf : "—";
            const ghRaw = (f.gold_hint || "").trim();
            const ghBad =
              !ghRaw ||
              ["UNVERIFIABLE", "NONE", "NULL", "MAJORITY_CORRECT"].includes(
                ghRaw.toUpperCase()
              );
            const correction =
              (f.correction || f.chosen || "").trim() ||
              (ghBad ? "" : ghRaw);
            const studentSaid = (f.student_answer || "").trim();
            const contrastWrong = (f.failure_answer || f.rejected || "").trim();
            const believed =
              studentSaid || contrastWrong;
            const showContrastNote =
              studentSaid &&
              contrastWrong &&
              studentSaid !== contrastWrong;

            const qPreview = (f.question || "").trim();

            return (
              <div
                key={`m-${f.step ?? i}-${f.difficulty_tier ?? ""}-${f.topic ?? ""}-${f.question_type ?? ""}-${i}`}
                id={`failure-${i}`}
                style={{
                  borderLeft:   `3px solid ${zStyle.color}`,
                  padding:      "10px 12px",
                  marginBottom: "8px",
                  background:   `${zStyle.bg}`,
                  borderRadius: "0 8px 8px 0",
                  transition:   "background 0.2s",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.filter = "brightness(1.12)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.filter = "none"; }}
              >
                {/* Skill tag row */}
                <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "7px", flexWrap: "wrap" }}>
                  <SkillBadge
                    topic={f.topic}
                    questionType={f.question_type}
                    difficulty={f.difficulty_tier}
                  />
                  <StatusBadge learned={!!f.learned} />
                </div>

                {qPreview ? (
                  <p
                    style={{
                      margin: "0 0 8px",
                      fontSize: "10px",
                      color: "#8b949e",
                      lineHeight: 1.45,
                    }}
                  >
                    <strong style={{ color: "#6e7681" }}>Q:</strong> {truncate(qPreview, 160)}
                  </p>
                ) : null}

                <div>
                  <p style={{ margin: "0 0 4px", fontSize: "11px", color: "#e6edf3", lineHeight: 1.45 }}>
                    <strong>
                      {studentSaid
                        ? `❌ Student said (conf ${confLabel})`
                        : `❌ Contrastive mistake (conf ${confLabel})`}
                      :
                    </strong>
                  </p>
                  <p
                    className="text-red"
                    style={{
                      margin: "0 0 10px",
                      fontSize: "11px",
                      lineHeight: 1.5,
                      color: "#f87171",
                    }}
                  >
                    {truncate(believed) || "—"}
                  </p>
                  {showContrastNote && (
                    <p
                      style={{
                        margin: "-6px 0 10px",
                        fontSize: "9px",
                        color: "#8b949e",
                        lineHeight: 1.4,
                      }}
                    >
                      DPO negative (wrong teacher line): {truncate(contrastWrong, 80)}
                    </p>
                  )}

                  <p style={{ margin: "0 0 4px", fontSize: "11px", color: "#e6edf3", lineHeight: 1.45 }}>
                    <strong>✅ Correct reasoning:</strong>
                  </p>
                  <p
                    className="text-green"
                    style={{
                      margin: 0,
                      fontSize: "11px",
                      lineHeight: 1.5,
                      color: "#4ade80",
                    }}
                  >
                    {truncate(correction) || "—"}
                  </p>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
