"""
test_key.py  — Run this FIRST to verify your Groq API key works.
Usage:  python3 test_key.py
"""
import os, json
import urllib.request, urllib.error

KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = os.environ.get("GROQ_TEST_MODEL", "llama-3.1-8b-instant")
if not KEY:
    print("❌ GROQ_API_KEY is not set in this shell.")
    print("   Fix: export GROQ_API_KEY=your_key_here")
    raise SystemExit(1)

print("Testing GROQ_API_KEY from environment (value not printed).")

req = urllib.request.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    method="POST",
    headers={
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    },
    data=json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 5,
    }).encode(),
)

try:
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
        reply = body["choices"][0]["message"]["content"]
        print(f"✅ Key works! Model replied: {reply!r}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"❌ HTTP {e.code}: {body}")
    if e.code == 401:
        print("   → Key is invalid or revoked. Get a new one at https://console.groq.com/keys")
    elif e.code == 400:
        print("   → Bad request (likely model name issue or malformed payload)")
        if "model_decommissioned" in body:
            print("   → The selected model was retired. Try:")
            print("      $env:GROQ_TEST_MODEL='llama-3.1-8b-instant'")
            print("      python test_key.py")
    elif e.code == 429:
        print("   → Rate limit hit. Wait 60 seconds and try again.")
    elif e.code == 403 and "1010" in body:
        print("   → Cloudflare blocked this request signature (Error 1010).")
        print("   → Try from a normal home network (no VPN/proxy) and verify Groq project permissions.")
