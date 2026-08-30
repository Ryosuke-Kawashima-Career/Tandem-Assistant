"""
Summary:
    conftest.py isolates the test suite from real Agora credentials on disk.

    src/server.py calls load_dotenv(override=True) at import time so a real .env is
    always authoritative for a running server, even over a stale same-name variable
    left in the shell (e.g. `$env:CONVOAI_LLM_BASE_URL` outliving an edit to .env).
    override=True means it will happily clobber mock values this file sets via
    os.environ - a plain os.environ.setdefault() guard is not enough once
    override=True is in play.

    pytest imports this file before collecting any test module, so neutralizing
    dotenv.load_dotenv here - before `src.server` is ever imported - guarantees the
    test suite exercises the simulated Convo AI path even on a developer machine
    whose `.env` holds real, working Agora credentials. Without this, every test run
    would attempt a real POST to Agora's REST API in /api/convoai/start.

    ECHOSPHERE_ENGINE / OPENAI_API_KEY / GEMINI_API_KEY are neutralized for the same
    reason and by the same mechanism: `TeachingAgent` reads them from the real process
    environment too, independently of the .env file this module already blocks. A shell
    that has previously exported real values (e.g. from a manual `uv run python -m
    src.server` session) makes the suite silently place real, billed calls to a live
    LLM provider - slow, non-deterministic, and network-dependent. This was the
    condition that turned a narrow, near-instant race in the Convo AI scaffolding
    background task (dev/tasks/task_plans/implementation_plan_latency_improvement.md
    Phase 7) into an easily-hit one: real provider latency gave a leaked future far
    more time to land inside an unrelated, later test's mock.
"""

import os
import dotenv

# Patched on the module object, not just this file's local name: src/server.py does
# `from dotenv import load_dotenv`, which resolves dotenv.load_dotenv at import time -
# since that import happens later (on the test suite's first `from src.server import
# ...`), it picks up this no-op instead of the real function.
dotenv.load_dotenv = lambda *args, **kwargs: False

os.environ["AGORA_APP_ID"] = "mock_app_id"
os.environ["AGORA_APP_CERTIFICATE"] = "mock_certificate"
os.environ["AGORA_CUSTOMER_ID"] = ""
os.environ["AGORA_CUSTOMER_SECRET"] = ""
os.environ["CONVOAI_LLM_BASE_URL"] = "http://localhost:8000"

# Force the mock reasoning engine and blank both provider keys, regardless of what a
# developer's shell happens to have exported. Tests that need a specific engine pass
# engine=/*_api_key= explicitly as constructor arguments (which override these), or
# scope their own patch.dict override - neither is weakened by this default.
os.environ["ECHOSPHERE_ENGINE"] = "mock"
os.environ["OPENAI_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
