# ============================================================
# LANGUAGE (en / hi / hinglish)
# ============================================================

ALLOWED_LANGS = {"en", "hi", "hinglish"}

def normalize_language(value: Any) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()

    # common variants
    if v in {"english", "eng", "en-us", "en_in", "en-gb"}:
        return "en"
    if v in {"hindi", "hin", "hi-in"}:
        return "hi"
    if v in {"hinglish", "hi-en", "hin-eng", "hindi+english"}:
        return "hinglish"

    if v in ALLOWED_LANGS:
        return v
    return None

def resolve_teaching_language(req_language: Any, signals: Dict[str, Any], brain: Dict[str, Any]) -> str:
    """
    Priority:
    1) signals.language / signals.preferred_language
    2) request.language
    3) brain.preferred_language
    Default: 'hinglish'

    Rule: If student chooses Hindi => teach in Hinglish.
    """
    signals = signals or {}
    brain = brain or {}

    raw = (
        signals.get("language")
        or signals.get("preferred_language")
        or req_language
        or brain.get("preferred_language")
    )

    lang = normalize_language(raw) or "hinglish"

    # Your rule: Hindi -> Hinglish teaching
    if lang == "hi":
        return "hinglish"

    return lang