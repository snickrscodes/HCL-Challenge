"""Authored response realization, separate from perception and its metrics."""


def respond(text, emotion, confidence):
    lower = text.strip().lower()
    if lower in {"bye", "bye.", "goodbye", "goodbye."}:
        return "Take care. We can talk again whenever you like."
    if lower.startswith(("thank you", "thanks")):
        return "You're welcome. Is there anything else you'd like to talk about?"
    if confidence < 0.5:
        return "I'm not sure how that felt for you. Would you tell me a little more?"
    if "?" in text and emotion == "neutral":
        return "Could you tell me a little more about what you'd like help with?"
    phrases = {
        "neutral": "I'm listening. What would you like to add?",
        "joy": "That sounds encouraging. What was the best part?",
        "sadness": "That sounds difficult. Want to tell me more?",
        "anger": "That sounds frustrating. Want to tell me what happened?",
        "surprise": "That sounds unexpected. What happened next?",
        "fear": "That sounds unsettling. Would it help to talk it through?",
        "disgust": "That sounds unpleasant. Want to say more about it?",
    }
    return phrases[emotion]
