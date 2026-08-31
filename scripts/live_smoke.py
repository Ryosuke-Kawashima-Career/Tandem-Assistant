"""
Summary:
    live_smoke.py is the repeatable Phase 8.2 verification: it exercises the Convo AI
    voice path against real Agora credentials and the public Custom LLM tunnel, in all
    three session languages (hi / ja / en).

Checks, per language:
    1. /api/rtc/token issues a real (non-simulated) RTC + RTM token pair.
    2. The Convo AI Engine's Custom LLM contract answers over the PUBLIC base URL and
       replies in the session's own script - the check that caught the Hindi session
       being told "Target Language: Japanese".
    3. A real Convo AI agent starts, reaches RUNNING, and stops cleanly.

Prerequisites:
    - .env holds real AGORA_* / CONVOAI_* credentials
    - `uv run python -m src.server` is running on PORT
    - `ngrok http 8000 --url=<CONVOAI_LLM_BASE_URL>` is up

Not covered: audible playback through a browser mic and speakers, which needs a human.

Usage:
    PYTHONIOENCODING=utf-8 uv run python scripts/live_smoke.py
"""
import json, os, sys, time, urllib.request, urllib.error

try:  # .env is authoritative for the tunnel URL, exactly as it is for the server.
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

LOCAL = os.environ.get("LOCAL_BASE", f"http://localhost:{os.environ.get('PORT', '8000')}")
# Read from the environment so no tunnel hostname is hardcoded into the repo.
PUBLIC = os.environ.get("PUBLIC_BASE") or os.environ.get("CONVOAI_LLM_BASE_URL") or ""
if not PUBLIC.startswith("https://"):
    sys.exit("Set CONVOAI_LLM_BASE_URL (or PUBLIC_BASE) to the public https tunnel URL.")
LANGS = ["en", "ja", "hi"]
results = []

def req(url, body=None, method=None, timeout=60, stream=False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    r.add_header("Content-Type", "application/json")
    r.add_header("ngrok-skip-browser-warning", "1")
    return urllib.request.urlopen(r, timeout=timeout)

def record(name, ok, detail):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + " :: " + detail, flush=True)

# --- 1. token issuance (real credentials, not simulated) ---
for i, lang in enumerate(LANGS):
    ch = f"smoke-{lang}-{int(time.time())}"
    try:
        tok = json.load(req(f"{LOCAL}/api/rtc/token?channel={ch}&uid={1000+i}"))
        ok = (not tok["simulated"]) and len(tok.get("token") or "") > 50 and len(tok.get("rtm_token") or "") > 50
        record(f"rtc-token[{lang}]", ok,
               f"simulated={tok['simulated']} app_id={tok['app_id'][:6]}... rtc_token_len={len(tok.get('token') or '')} rtm_token_len={len(tok.get('rtm_token') or '')}")
    except Exception as e:
        record(f"rtc-token[{lang}]", False, repr(e))

# --- 2. Custom LLM bridge over the PUBLIC url (what the Engine actually calls) ---
def script_ok(lang, text):
    """The reply must be spoken in the session language, not merely produced."""
    if lang == "ja":
        return any("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in text)
    if lang == "hi":
        return any("ऀ" <= c <= "ॿ" for c in text)
    return any(c.isascii() and c.isalpha() for c in text)

PROMPTS = {"en": "Hello, can you help me practice English?",
           "ja": "こんにちは、日本語を練習したいです。",
           "hi": "नमस्ते, मैं हिंदी सीखना चाहता हूँ।"}
for lang in LANGS:
    try:
        t0 = time.time()
        resp = req(f"{PUBLIC}/chat/completions", {
            "model": "echosphere-teaching-agent",
            "language": lang,
            "speaker_id": "SmokeLearner",
            "stream": True,
            "messages": [{"role": "user", "content": PROMPTS[lang]}],
        }, timeout=90)
        ttfb = None; text = ""
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            if ttfb is None:
                ttfb = time.time() - t0
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content") or ""
            except Exception:
                delta = ""
            text += delta
        ok = len(text.strip()) > 0 and script_ok(lang, text)
        record(f"llm-bridge[{lang}]", ok,
               f"script_ok={script_ok(lang, text)} " +
               f"ttfb={ttfb:.2f}s total={time.time()-t0:.2f}s chars={len(text)} reply={text[:120]!r}")
    except Exception as e:
        record(f"llm-bridge[{lang}]", False, repr(e))

# --- 3. Convo AI agent lifecycle against the real Agora Engine ---
for lang in LANGS:
    ch = f"smoke-{lang}-{int(time.time())}"
    agent_id = None
    try:
        start = json.load(req(f"{LOCAL}/api/convoai/start", {"channel": ch, "language": lang, "speaker_id": "SmokeLearner"}, timeout=60))
        agent = start.get("agent") or {}
        agent_id = agent.get("agent_id")
        record(f"convoai-start[{lang}]", bool(start.get("success") and agent_id),
               f"channel={ch} agent_id={agent_id} status={agent.get('status')} simulated={agent.get('simulated')}")
        state = None
        for _ in range(12):
            time.sleep(2)
            st = json.load(req(f"{LOCAL}/api/convoai/status?channel={ch}" + (f"&agent_id={agent_id}" if agent_id else "")))
            a = st.get("agent") or {}
            state = a.get("status") or a.get("state")
            if str(state).upper() in ("RUNNING", "IDLE", "FAILED", "STOPPED"):
                break
        record(f"convoai-running[{lang}]", str(state).upper() in ("RUNNING", "IDLE"), f"state={state}")
    except urllib.error.HTTPError as e:
        record(f"convoai-start[{lang}]", False, f"HTTP {e.code}: {e.read()[:300]!r}")
    except Exception as e:
        record(f"convoai-start[{lang}]", False, repr(e))
    finally:
        try:
            stop = req(f"{LOCAL}/api/convoai/stop", {"channel": ch, "agent_id": agent_id}, timeout=30)
            record(f"convoai-stop[{lang}]", stop.status == 200, f"http={stop.status}")
        except urllib.error.HTTPError as e:
            record(f"convoai-stop[{lang}]", False, f"HTTP {e.code}")
        except Exception as e:
            record(f"convoai-stop[{lang}]", False, repr(e))

print("\n=== SUMMARY ===")
for n, ok, d in results:
    print(("PASS " if ok else "FAIL ") + n)
print(f"{sum(1 for _,ok,_ in results if ok)}/{len(results)} passed")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
