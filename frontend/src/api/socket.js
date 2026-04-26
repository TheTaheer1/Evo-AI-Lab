// WebSocket connection to the FastAPI backend
// Provides connect(), disconnect(), and sendAction(action)

const _wsBase = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/live";
const _wsToken = import.meta.env.VITE_EVOAI_API_TOKEN || "";
const WS_URL = _wsToken
  ? `${_wsBase}${_wsBase.includes("?") ? "&" : "?"}token=${encodeURIComponent(_wsToken)}`
  : _wsBase;

let socket = null;
let messageHandler = null;
let statusHandler = null;
let reconnectTimer = null;
let shouldReconnect = true;
const RECONNECT_DELAY_MS = 2000;

function clearReconnectTimer() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function scheduleReconnect() {
  if (!shouldReconnect) return;
  clearReconnectTimer();
  reconnectTimer = setTimeout(() => {
    if (shouldReconnect) openSocket();
  }, RECONNECT_DELAY_MS);
}

function openSocket() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    if (statusHandler) statusHandler("open");
    console.log("[WS] connected to", WS_URL);
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (messageHandler) messageHandler(data);
    } catch (e) {
      console.error("[WS] parse error", e);
    }
  };

  socket.onerror = (e) => {
    if (statusHandler) statusHandler("error");
    console.error("[WS] error", e);
  };

  socket.onclose = (e) => {
    if (statusHandler) statusHandler("closed");
    console.log("[WS] disconnected", e.code, e.reason);
    socket = null;
    scheduleReconnect();
  };
}

export function connect(onMessage, onStatus) {
  shouldReconnect = true;
  clearReconnectTimer();
  messageHandler = onMessage;
  statusHandler = onStatus || null;
  if (statusHandler) statusHandler("connecting");
  openSocket();
}

export function disconnect() {
  shouldReconnect = false;
  clearReconnectTimer();
  if (socket) {
    socket.close();
    socket = null;
  }
  if (statusHandler) statusHandler("closed");
}

export function sendAction(action) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(action));
  }
}
