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
