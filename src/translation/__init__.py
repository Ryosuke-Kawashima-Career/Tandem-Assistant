"""
Summary:
    `src/translation/` owns EchoSphere's Gemini Live Translate legs (REQ-17): the
    WebSocket transport (`gemini_live.py`), the PCM adaptation on either side of it
    (`audio.py`), and the mode-dependent routing that decides which participant's audio
    reaches which leg and whose ears the result lands in (`router.py`).

    The package is an interpreter transport only. Pedagogical and work reasoning stays in
    `src/agent/`, and the REQ-17 `TeachingAgent` language-role policy is deliberately
    implemented there, not here: the two halves of REQ-17 must remain independently
    changeable.
"""
