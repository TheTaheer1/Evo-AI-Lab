/** Merge optional Bearer token for EVOAI_API_KEY–protected backends. */
export function authHeaders(base = {}) {
  const token = import.meta.env.VITE_EVOAI_API_TOKEN || "";
  if (!token) return { ...base };
  return { ...base, Authorization: `Bearer ${token}` };
}
