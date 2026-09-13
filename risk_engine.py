from scam_detector import detect_scam


def calculate_risk(
    voice_is_suspicious,
    voice_confidence,
    scam_result
):

    scam_score = scam_result["risk_score"]

    if voice_is_suspicious and scam_score >= 0.45:

        risk_level = "high"
        trigger_challenge = True

        combined_score = (
            (voice_confidence * 0.6) +
            (scam_score * 0.4)
        )

    elif voice_is_suspicious:

        risk_level = "medium"
        trigger_challenge = False

        combined_score = voice_confidence * 0.6

    elif scam_score >= 0.45:

        risk_level = "medium"
        trigger_challenge = False

        combined_score = scam_score * 0.6

    else:

        risk_level = "low"
        trigger_challenge = False

        combined_score = 0.0

    return {
        "risk_level": risk_level,
        "combined_score": round(combined_score, 2),
        "urgency_detected": scam_result["urgency_detected"],
        "trigger_challenge": trigger_challenge,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_cases = [

        {
            "name": "Normal call",
            "voice_fake": False,
            "voice_confidence": 0.10,
            "text": "Hey, are you coming home for dinner?"
        },

        {
            "name": "Suspicious voice only",
            "voice_fake": True,
            "voice_confidence": 0.85,
            "text": "Hey, are you coming home for dinner?"
        },

        {
            "name": "Scam conversation only",
            "voice_fake": False,
            "voice_confidence": 0.15,
            "text": "Please send me the money right now."
        },

        {
            "name": "Voice clone + scam",
            "voice_fake": True,
            "voice_confidence": 0.92,
            "text": "Please send me the money right now."
        },

        {
            "name": "Voice clone + OTP scam",
            "voice_fake": True,
            "voice_confidence": 0.95,
            "text": "I need your OTP immediately."
        },
    ]


    print("\n===== VOICEGUARD RISK ENGINE =====\n")


    for case in test_cases:

        scam_result = detect_scam(case["text"])

        result = calculate_risk(
            case["voice_fake"],
            case["voice_confidence"],
            scam_result
        )

        print("CASE:", case["name"])
        print("TEXT:", case["text"])
        print("VOICE SUSPICIOUS:", case["voice_fake"])
        print("VOICE CONFIDENCE:", case["voice_confidence"])
        print("SCAM SCORE:", scam_result["risk_score"])
        print("FINAL RISK:", result["risk_level"])
        print("COMBINED SCORE:", result["combined_score"])
        print("TRIGGER CHALLENGE:", result["trigger_challenge"])
        print("-" * 70)