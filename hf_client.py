import requests
import os
import time
import threading
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = (os.getenv("HF_API_URL") or "").strip()
HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()

HF_LOCK = threading.Semaphore(3)  # limit parallel calls


def hf_generate(prompt, temperature=0.7, max_tokens=300):
    if not HF_API_URL or not HF_TOKEN:
        print(
            "[HF CONFIG MISSING] Set HF_API_URL and HF_TOKEN "
            "(e.g. copy .env.example to .env in the project root)."
        )
        return ""

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
        },
    }

    print(f"[HF CALL] prompt_len={len(prompt)}")

    for attempt in range(3):  # retry up to 3 times
        try:
            with HF_LOCK:
                response = requests.post(
                    HF_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=20,
                )

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "")

                return str(data)

            print(f"[HF ERROR] {response.status_code} {response.text}")

        except Exception as e:
            print(f"[HF RETRY {attempt + 1}] {e}")
            time.sleep(1)

    return ""
