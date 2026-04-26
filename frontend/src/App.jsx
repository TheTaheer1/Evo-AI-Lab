import { useState, useEffect } from "react";
import { connect, disconnect } from "./api/socket";
import { authHeaders } from "./api/authHeaders";
import CalibrationMap from "./components/CalibrationMap";
import RewardCurve from "./components/RewardCurve";
import FailureLog from "./components/FailureLog";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const styles = {
  root: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%)",
    color: "#e6edf3",
    fontFamily: "'Inter', system-ui, sans-serif",
    padding: "0",
  },
  header: {
    borderBottom: "1px solid rgba(255,255,255,0.07)",
    background: "rgba(13,17,23,0.95)",
    backdropFilter: "blur(12px)",
    position: "sticky",
    top: 0,
    zIndex: 100,
    padding: "0 24px",
  },
  headerInner: {
    maxWidth: "1280px",
    margin: "0 auto",
    height: "56px",
    display: "flex",
    alignItems: "center",
    gap: "16px",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  logoIcon: {
    width: "28px",
    height: "28px",
    borderRadius: "8px",
    background: "linear-gradient(135deg, #1D9E75, #0d7a5c)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "14px",
    fontWeight: "700",
    color: "#fff",
    flexShrink: 0,
  },
  title: {
    fontSize: "16px",
    fontWeight: "700",
    color: "#e6edf3",
    letterSpacing: "-0.01em",
  },
  subtitle: {
    fontSize: "12px",
    color: "#8b949e",
    fontWeight: "400",
  },
  pills: {
    display: "flex",
    gap: "8px",
    marginLeft: "auto",
    alignItems: "center",
  },
  pill: (color, bg) => ({
    fontSize: "11px",
    fontWeight: "600",
    padding: "3px 10px",
    borderRadius: "20px",
    background: bg,
    color: color,
    letterSpacing: "0.02em",
    border: `1px solid ${color}33`,
  }),
  btnRow: {
    display: "flex",
    gap: "8px",
    marginLeft: "16px",
  },
  btnRun: (running) => ({
    padding: "7px 18px",
    fontSize: "12px",
    fontWeight: "600",
    cursor: running ? "not-allowed" : "pointer",
    background: running ? "#21262d" : "linear-gradient(135deg, #1D9E75, #0d7a5c)",
    color: running ? "#8b949e" : "#fff",
    border: "none",
    borderRadius: "8px",
    transition: "all 0.2s",
    letterSpacing: "0.01em",
  }),
  btnReset: {
    padding: "7px 14px",
    fontSize: "12px",
    fontWeight: "500",
    cursor: "pointer",
    background: "transparent",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: "8px",
    color: "#8b949e",
    transition: "all 0.2s",
  },
  main: {
    maxWidth: "1280px",
    margin: "0 auto",
    padding: "20px 24px",
  },
  tagline: {
    fontSize: "12px",
    color: "#8b949e",
    textAlign: "center",
    padding: "10px 0 18px",
    fontStyle: "italic",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gridTemplateRows: "auto auto",
    gap: "16px",
  },
  leftPanel: {
    gridRow: "span 2",
  },
  metricCard: {
    padding: "6px 14px",
    borderRadius: "10px",
    display: "flex",
    flexDirection: "column",
    gap: "2px",
    marginLeft: "8px",
    flexShrink: 0,
  },
  metricCardTitle: {
    margin: 0,
    fontSize: "10px",
    fontWeight: "600",
    color: "#8b949e",
    letterSpacing: "0.04em",
    textTransform: "uppercase",
  },
  metricCardValue: (color) => ({
    margin: 0,
    fontSize: "17px",
    fontWeight: "700",
    color,
    letterSpacing: "-0.02em",
    lineHeight: 1.2,
  }),
};

export default function App() {
  const [nodes, setNodes] = useState([]);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [failures, setFailures] = useState([]);
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [zoneCounts, setZoneCounts] = useState({ zone_c: 0, zone_b: 0, green: 0 });
  const [calibrationScore, setCalibrationScore] = useState(0);
  const [difficultyTier, setDifficultyTier] = useState(2);   // adversary tier 1-5
  const [totalMoments, setTotalMoments]     = useState(0);   // cumulative learning moments
  const [wsConnected, setWsConnected] = useState(false);
  const [wsError, setWsError] = useState(null);
  const [wsPaused, setWsPaused] = useState(false);
  const [wsStopped, setWsStopped] = useState(false);

  useEffect(() => {
    const c = zoneCounts.zone_c || 0;
    const b = zoneCounts.zone_b || 0;
    const g = zoneCounts.green || 0;
    const totalNodes = c + b + g;
    setCalibrationScore(
      totalNodes ? Math.round((g / totalNodes) * 100) : 0
    );
  }, [zoneCounts]);

  useEffect(() => {
    // Load initial state from REST API
    fetch(`${API_BASE}/api/state`, { headers: authHeaders() })
      .then((r) => r.json())
      .then((data) => {
        setNodes(data.calibration_map?.nodes || []);
        setRewardHistory(data.reward_history || []);
        setStep(data.step || 0);
        setZoneCounts({
          zone_c: data.zone_c_count || 0,
          zone_b: data.zone_b_count || 0,
          green: data.green_count || 0,
        });
        if (typeof data.difficulty_tier === "number") {
          setDifficultyTier(data.difficulty_tier);
        }
        if (typeof data.total_moments === "number") {
          setTotalMoments(data.total_moments);
        }
      })
      .then(() =>
        fetch(`${API_BASE}/api/failures?n=50`, { headers: authHeaders() })
          .then((r) => r.json())
          .then((failData) => {
            if (Array.isArray(failData)) {
              setFailures(failData);
            }
          })
      )
      .catch((e) => console.warn("[App] Could not fetch initial state:", e));

    // Connect WebSocket for live updates
    connect((data) => {
      setWsConnected(true);

      if (data.error) {
        setWsError(data.error);
        setWsPaused(!!data.paused);
        setWsStopped(!!data.stopped);
        return;
      }

      setWsError(null);
      setWsPaused(false);
      setWsStopped(false);

      if (data.calibration_map) setNodes(data.calibration_map.nodes || []);
      if (data.reward !== undefined) {
        setRewardHistory((prev) => [
          ...prev,
          { step: data.step ?? prev.length, reward: data.reward },
        ]);
      }
      // Learning moments: prefer WS push for real-time feel
      const _failureHasContent =
        data.failure &&
        (data.failure.student_answer ||
          data.failure.failure_answer ||
          data.failure.correction ||
          data.failure.gold_hint);
      if (_failureHasContent) {
        setFailures((prev) => {
          const dupKey = (x) =>
            `${x.step}-${x.topic}-${x.question_type}-${x.difficulty_tier ?? ""}`;
          const isDuplicate = prev.some(
            (f) => dupKey(f) === dupKey(data.failure)
          );
          if (isDuplicate) return prev;
          const updated = prev.map((f) =>
            f.topic === data.failure.topic &&
            f.question_type === data.failure.question_type &&
            (f.difficulty_tier || "") === (data.failure.difficulty_tier || "") &&
            data.failure.is_correct === true
              ? { ...f, learned: true }
              : f
          );
          return [data.failure, ...updated].slice(0, 50);
        });
      }
      if (data.step !== undefined)            setStep(data.step);
      if (data.zone_counts)                   setZoneCounts(data.zone_counts);
      if (data.difficulty_tier !== undefined) setDifficultyTier(data.difficulty_tier);
      // Always prefer server-sent totalMoments; fall back to incrementing
      if (data.total_moments !== undefined) {
        setTotalMoments(data.total_moments);
      } else if (_failureHasContent) {
        setTotalMoments((n) => n + 1);
      }
    }, (status) => {
      if (status === "connecting") {
        setWsConnected(false);
      } else if (status === "open") {
        setWsConnected(true);
      } else if (status === "closed") {
        setWsConnected(false);
      }
    });

    return () => disconnect();
  }, []);

  const handleRun = async () => {
    setRunning(true);
    try {
      const response = await fetch(`${API_BASE}/api/run-steps`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ n: 1 }),
      });
      if (response.ok) {
        const data = await response.json();
        const last = Array.isArray(data.results) && data.results.length > 0
          ? data.results[data.results.length - 1]
          : null;
        const obs = last?.observation || {};
        const info = last?.info || {};

        if (
          last &&
          !last.info?.skipped &&
          typeof last.reward === "number"
        ) {
          setRewardHistory((prev) => [
            ...prev,
            { step: obs.step ?? prev.length, reward: last.reward },
          ]);
        }

        if (obs.calibration_map) setNodes(obs.calibration_map.nodes || []);
        if (typeof obs.step === "number") setStep(obs.step);
        setZoneCounts({
          zone_c: obs.zone_c_count || 0,
          zone_b: obs.zone_b_count || 0,
          green: obs.green_count || 0,
        });
        if (typeof obs.difficulty_tier === "number") {
          setDifficultyTier(obs.difficulty_tier);
        } else if (typeof info.difficulty_tier === "number") {
          setDifficultyTier(info.difficulty_tier);
        }
        if (typeof data.total_moments === "number") {
          setTotalMoments(data.total_moments);
        } else if (typeof obs.total_moments === "number") {
          setTotalMoments(obs.total_moments);
        }
      }
      const failData = await fetch(`${API_BASE}/api/failures?n=50`, {
        headers: authHeaders(),
      }).then((r) => r.json());
      if (Array.isArray(failData)) {
        setFailures(failData);
      }
    } catch (e) {
      console.error("[App] run-steps error:", e);
    } finally {
      setRunning(false);
    }
  };

  const handleReset = async () => {
    try {
      await fetch(`${API_BASE}/api/reset`, { method: "POST", headers: authHeaders() });
      setRewardHistory([]);
      setFailures([]);
      setStep(0);
      setDifficultyTier(2);
      setTotalMoments(0);
      setZoneCounts({ zone_c: 0, zone_b: 0, green: 0 });
      // Reload calibration map seed
      const state = await fetch(`${API_BASE}/api/state`, { headers: authHeaders() }).then(
        (r) => r.json()
      );
      setNodes(state.calibration_map?.nodes || []);
    } catch (e) {
      console.error("[App] reset error:", e);
    }
  };

  const calibrationScoreColor =
    calibrationScore < 30
      ? "#E24B4A"
      : calibrationScore <= 60
        ? "#EF9F27"
        : "#1D9E75";

  return (
    <div style={styles.root}>
      {/* ── Header ── */}
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div style={styles.logo}>
            <div style={styles.logoIcon}>E</div>
            <div>
              <div style={styles.title}>EvoAI Lab</div>
            </div>
          </div>

          <div
            className="metric-card"
            style={{
              ...styles.metricCard,
              border: `1px solid ${calibrationScoreColor}40`,
              background: `${calibrationScoreColor}14`,
            }}
          >
            <h3 style={styles.metricCardTitle}>Calibration Score</h3>
            <p style={styles.metricCardValue(calibrationScoreColor)}>{calibrationScore}%</p>
          </div>

          <div style={styles.pills}>
            <span style={styles.pill("#E24B4A", "rgba(226,75,74,0.12)")}>
              Zone C: {zoneCounts.zone_c}
            </span>
            <span style={styles.pill("#EF9F27", "rgba(239,159,39,0.12)")}>
              Zone B: {zoneCounts.zone_b}
            </span>
            <span style={styles.pill("#1D9E75", "rgba(29,158,117,0.12)")}>
              Green: {zoneCounts.green}
            </span>
            <span style={styles.pill("#8b949e", "rgba(139,148,158,0.1)")}>
              Step {step}
            </span>
            <span style={styles.pill("#a78bfa", "rgba(167,139,250,0.1)")}>
              Tier {difficultyTier}/5
            </span>
            <span
              style={styles.pill(
                wsConnected
                  ? wsStopped
                    ? "#E24B4A"
                    : wsPaused
                    ? "#EF9F27"
                    : "#1D9E75"
                  : "#8b949e",
                wsConnected
                  ? wsStopped
                    ? "rgba(226,75,74,0.15)"
                    : wsPaused
                    ? "rgba(239,159,39,0.15)"
                    : "rgba(29,158,117,0.1)"
                  : "rgba(139,148,158,0.08)"
              )}
            >
              {wsConnected
                ? wsStopped
                  ? "⛔ Stopped"
                  : wsPaused
                  ? "⚠ Paused"
                  : "● Live"
                : "○ Connecting"}
            </span>
          </div>

          <div style={styles.btnRow}>
            <button
              id="btn-run-step"
              onClick={handleRun}
              disabled={running}
              style={styles.btnRun(running)}
              onMouseEnter={(e) => {
                if (!running) e.currentTarget.style.opacity = "0.88";
              }}
              onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
            >
              {running ? "Running…" : "▶ Run step"}
            </button>
            <button
              id="btn-reset"
              onClick={handleReset}
              style={styles.btnReset}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.25)";
                e.currentTarget.style.color = "#e6edf3";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.12)";
                e.currentTarget.style.color = "#8b949e";
              }}
            >
              Reset
            </button>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main style={styles.main}>
        {/* ── Error / paused banner ── */}
        {wsError && (
          <div
            style={{
              margin: "0 0 14px 0",
              padding: "10px 16px",
              borderRadius: "8px",
              background: wsStopped
                ? "rgba(226,75,74,0.12)"
                : "rgba(239,159,39,0.10)",
              border: `1px solid ${
                wsStopped ? "rgba(226,75,74,0.3)" : "rgba(239,159,39,0.3)"
              }`,
              fontSize: "12px",
              color: wsStopped ? "#E24B4A" : "#EF9F27",
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <span style={{ fontWeight: 700 }}>
              {wsStopped ? "⛔ Training stopped" : "⚠ Training paused"}
            </span>
            <span style={{ color: "#8b949e", flex: 1 }}>{wsError}</span>
            {wsStopped && (
              <span style={{ color: "#8b949e" }}>
                Restart the backend to resume.
              </span>
            )}
          </div>
        )}

        <p style={styles.tagline}>
          "Most AI training asks if the model is right or wrong. We train for something harder:{" "}
          <strong style={{ color: "#e6edf3" }}>is it right about being right?</strong>"
        </p>

        <div style={styles.grid}>
          {/* Left — Calibration Map spans both rows */}
          <div style={styles.leftPanel}>
            <CalibrationMap nodes={nodes} />
          </div>

          {/* Top right — Reward Curve */}
          <div>
            <RewardCurve history={rewardHistory} />
          </div>

          <div>
            <FailureLog failures={failures} totalMoments={totalMoments} />
          </div>
        </div>
      </main>
    </div>
  );
}
