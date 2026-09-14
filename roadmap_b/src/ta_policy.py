"""Fixed H1 authored response routes; estimates never become observed history."""

import math

from .responses import respond

LABELS6 = ("neutral", "happy", "sad", "angry", "fear", "disgust")
ROUTES = {
    "angry": "Would you like to vent, or focus on what to do next?",
    "fear": "Would it help to slow down and talk through your concern?",
    "disgust": "Would you like to say more about this, or change the subject?",
}
POLICIES = ("evidence_only", "role_specific_h1")


def select_response(text, state, evidence, *, policy="evidence_only"):
    """Keep B state untouched and return a separately owned proposed action."""
    if policy not in POLICIES:
        raise ValueError("Unknown authored response policy")
    reference = respond(text, state["emotion"], state["confidence"])
    result = {"policy": policy, "route_id": "b_policy", "evidence_label": None,
              "response": reference, "reference_response": reference,
              "reason": "evidence_only"}
    if policy == "evidence_only":
        return result
    lower = text.strip().lower()
    if lower in {"bye", "bye.", "goodbye", "goodbye."} or lower.startswith(("thank you", "thanks")):
        result["reason"] = "lexical_precedence"
        return result
    if state["emotion"] != "neutral" and state["confidence"] >= 0.5:
        result["reason"] = "contextual_non_neutral"
        return result
    if not evidence.get("available"):
        result["reason"] = "audio_unavailable"
        return result
    values = evidence.get("distribution")
    if evidence.get("namespace") != "crema6_audio_votes_v1" or not isinstance(values, dict):
        raise ValueError("Expected genuine CREMA6 evidence")
    if set(values) != set(LABELS6):
        raise ValueError("Six-class namespace label mismatch")
    p = [float(values[k]) for k in LABELS6]
    if any(not math.isfinite(v) or v < 0 or v > 1 for v in p) or not math.isclose(sum(p), 1.0, abs_tol=1e-6):
        raise ValueError("Invalid six-class probability distribution")
    order = sorted(range(6), key=lambda i: (-p[i], i))
    label = LABELS6[order[0]]
    if label not in ROUTES:
        result["reason"] = "unsupported_vocal_route"
    elif p[order[0]] < 0.60 or p[order[0]] - p[order[1]] < 0.15:
        result["reason"] = "low_vocal_confidence_or_margin"
    else:
        result.update(route_id="vocal_" + label, evidence_label=label,
                      response=ROUTES[label], reason="supported_vocal_offer")
    return result
