"""
app.py
FastAPI server for EvoAI Lab.
Exposes REST endpoints and WebSocket for the React frontend.
"""
import asyncio
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

# Load project-root .env before backend imports (HF_API_URL, HF_TOKEN, GROQ_API_KEY, etc.).
_root = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(_root / ".env")

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.env.evoai_env import EvoAIEnv


def _cors_allow_origins() -> list:
    raw = os.environ.get("EVOAI_CORS_ORIGINS", "").strip()
    if not raw:
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    if raw == "*":
        return ["*"]
    parts = [o.strip() for o in raw.split(",") if o.strip()]
    return parts or ["http://localhost:5173"]


def require_api_key_if_set(request: Request) -> None:
    """When EVOAI_API_KEY is set, require Bearer token or X-API-Key on REST routes."""
    expected = os.environ.get("EVOAI_API_KEY", "").strip()
    if not expected:
        return
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {expected}":
        return
    if request.headers.get("X-API-Key", "") == expected:
        return
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="EvoAI Lab", description="Self-improving LLM calibration environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Environment singleton (lazy-initialised at startup) ────────────────────
env: EvoAIEnv | None = None
_step_lock = asyncio.Lock()
_WS_STEP_INTERVAL_SECS = float(os.environ.get("EVOAI_WS_STEP_INTERVAL_SECS", "6.0"))
_WS_AUTOSTEP = os.environ.get("EVOAI_WS_AUTOSTEP", "false").strip().lower() in {"1", "true", "yes", "on"}


def _require_env() -> EvoAIEnv:
    if env is None:
        raise HTTPException(status_code=503, detail="Environment is not initialized")
    return env


@app.on_event("startup")
async def startup_event():
    global env
    env = EvoAIEnv()


@app.on_event("shutdown")
async def shutdown_event():
    if env is not None:
        env.close()


# ── Request models ─────────────────────────────────────────────────────────
class RunStepsRequest(BaseModel):
    n: int = 1


# ── REST endpoints ─────────────────────────────────────────────────────────

@app.get("/api/state")
async def get_state(_: None = Depends(require_api_key_if_set)):
    """Return the current environment state."""
    return _require_env().state()


@app.get("/api/reward-curve")
async def get_reward_curve(_: None = Depends(require_api_key_if_set)):
    """Return the full reward history."""
    current_env = _require_env()
    return current_env.pipeline.dataset_builder.get_reward_curve()


@app.get("/api/failures")
async def get_failures(n: int = 10, _: None = Depends(require_api_key_if_set)):
    """Return the n most recent learning moments."""
    current_env = _require_env()
    return current_env.pipeline.dataset_builder.get_recent_failures(n)


@app.get("/api/calibration-map")
async def get_calibration_map(_: None = Depends(require_api_key_if_set)):
    """Return the current calibration map as serialisable dict."""
    current_env = _require_env()
    return current_env.pipeline.calibration_map.to_dict()


@app.post("/api/reset")
async def reset_env(_: None = Depends(require_api_key_if_set)):
    """Reset the environment and return the initial state."""
    current_env = _require_env()
    async with _step_lock:
        return current_env.reset()


@app.post("/api/run-steps")
async def run_steps(body: RunStepsRequest, _: None = Depends(require_api_key_if_set)):
    """Run n training steps and return the final state."""
    current_env = _require_env()
    n = max(1, min(int(body.n), 200))
    results = []
    async with _step_lock:
        for _ in range(n):
            result = await current_env.step()
            results.append(result)
    return {
        "steps_run": n,
        "final_state": current_env.state(),
        "results": results,
        "total_moments": current_env.pipeline.dataset_builder.get_total_moments(),
    }


# ── WebSocket — live step streaming ───────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Stream training step results in real time.
    On connect: send current state.
    Then run steps continuously; send each step result as JSON.
    On disconnect: stop cleanly.
    """
    expected_key = os.environ.get("EVOAI_API_KEY", "").strip()
    await websocket.accept()
    if expected_key:
        token = websocket.query_params.get("token", "")
        auth_h = websocket.headers.get("authorization", "")
        if token != expected_key and auth_h != f"Bearer {expected_key}":
            await websocket.close(code=1008)
            return

    current_env = env
    if current_env is None:
        await websocket.send_text(json.dumps({"error": "Environment is not initialized"}))
        await websocket.close(code=1011)
        return

    # ── Failure tracking ───────────────────────────────────────────────────
    MAX_BACKOFF_SECS = 60        # cap on wait between retries
    PAUSE_AFTER     = 5         # consecutive failures → pause + notify frontend
    STOP_AFTER      = 10        # consecutive failures → stop loop + notify frontend
    consecutive_failures = 0

    async def _send(payload: dict):
        await websocket.send_text(json.dumps(payload))

    try:
        # Send initial state immediately on connect
        initial_state = current_env.state()
        await _send(initial_state)

        # Continuous step loop with failure-aware backoff
        while True:
            try:
                if not _WS_AUTOSTEP:
                    await _send({
                        **current_env.state(),
                        "auto_step": False,
                        "error": None,
                    })
                    await asyncio.sleep(_WS_STEP_INTERVAL_SECS)
                    continue

                async with _step_lock:
                    step_result = await current_env.step()
                info = step_result.get("info", {})
                skipped = info.get("skipped", False)

                # Successful step — reset failure counter
                consecutive_failures = 0

                payload = {
                    "step": step_result.get("observation", {}).get("step", 0),
                    "calibration_map": step_result.get("observation", {}).get("calibration_map", {}),
                    "zone_counts": {
                        "zone_c": step_result.get("observation", {}).get("zone_c_count", 0),
                        "zone_b": step_result.get("observation", {}).get("zone_b_count", 0),
                        "green":  step_result.get("observation", {}).get("green_count", 0),
                    },
                    "skipped":          skipped,
                    "failure":          info.get("failure"),
                    "difficulty_tier":  info.get("difficulty_tier", 2),
                    "total_moments":    current_env.pipeline.dataset_builder.get_total_moments(),
                    "error":            None,
                }
                if not skipped:
                    payload["reward"] = step_result.get("reward", 0.0)

                await _send(payload)
                await asyncio.sleep(_WS_STEP_INTERVAL_SECS)

            except WebSocketDisconnect:
                break

            except Exception as e:
                consecutive_failures += 1
                # Exponential backoff: 3s, 6s, 12s, 24s, capped at MAX_BACKOFF_SECS
                backoff = min(3 * (2 ** (consecutive_failures - 1)), MAX_BACKOFF_SECS)
                err_msg = str(e)[:200]
                print(f"[WS] Step error #{consecutive_failures}: {err_msg} — waiting {backoff}s")

                # Notify the frontend of the error so the UI can show a warning
                try:
                    await _send({
                        "error": err_msg,
                        "consecutive_failures": consecutive_failures,
                        "paused": consecutive_failures >= PAUSE_AFTER,
                        "stopped": consecutive_failures >= STOP_AFTER,
                    })
                except Exception:
                    pass  # websocket may have closed

                if consecutive_failures >= STOP_AFTER:
                    print(f"[WS] {STOP_AFTER} consecutive failures — stopping loop. Restart backend or reset.")
                    break

                if consecutive_failures >= PAUSE_AFTER:
                    print(f"[WS] {PAUSE_AFTER} consecutive failures — pausing {backoff}s before retry.")

                await asyncio.sleep(backoff)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Connection error: {e}")
    finally:
        print("[WS] Client disconnected cleanly.")


# ── Serve React frontend (production build) ────────────────────────────────
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
