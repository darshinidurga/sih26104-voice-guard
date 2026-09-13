import re


# ============================================================
# VOICEGUARD SOCIAL-ENGINEERING SIGNALS
# ============================================================

SIGNAL_PATTERNS = {

    "money_request": [
        r"\bsend (me )?(the )?money\b",
        r"\btransfer (me )?(the )?money\b",
        r"\btransfer funds\b",
        r"\bsend (me )?(the )?payment\b",
        r"\bmake (a )?payment\b",
        r"\bpay (me|us)\b",
        r"\bhelp me (with )?(a )?payment\b",
        r"\bneed.*money\b",
        r"\bneed.*payment\b",
        r"\bborrow.*money\b",
    ],

    "credential_request": [
        r"\botp\b",
        r"\bone time password\b",
        r"\bverification code\b",
        r"\bsecurity code\b",
        r"\bpasscode\b",
        r"\bpassword\b",
        r"\bpin\b",
        r"\bcvv\b",
        r"\bcode\b.*\bsend\b",
        r"\bsend\b.*\bcode\b",
    ],

    "financial_information": [
        r"\bbank account\b",
        r"\bbank details\b",
        r"\baccount number\b",
        r"\bcredit card\b",
        r"\bdebit card\b",
        r"\bcard details\b",
        r"\baccount\b.*\bblocked\b",
        r"\baccount\b.*\bclosed\b",
    ],

    "gift_card_request": [
        r"\bgift card\b",
        r"\bgift cards\b",
        r"\bbuy.*gift card\b",
        r"\bsend.*gift card\b",
    ],

    "urgency": [
        r"\bright now\b",
        r"\bimmediately\b",
        r"\burgent\b",
        r"\bas soon as possible\b",
        r"\bdon'?t wait\b",
        r"\bdo not wait\b",
        r"\bwithin \d+ minutes?\b",
        r"\byou only have\b",
        r"\bbefore.*closes?\b",
        r"\bbefore.*ends?\b",
        r"\bquickly\b",
        r"\basap\b",
    ],

    "secrecy": [
        r"\bdon'?t tell anyone\b",
        r"\bdo not tell anyone\b",
        r"\bkeep this secret\b",
        r"\bkeep it secret\b",
        r"\bdon'?t tell\b",
        r"\bdo not tell\b",
        r"\bkeep this between us\b",
        r"\btell no one\b",
    ],

    "threat_or_pressure": [
        r"\bpolice\b",
        r"\barrest\b",
        r"\byou'?ll be arrested\b",
        r"\blegal action\b",
        r"\byour account will be blocked\b",
        r"\baccount will be blocked\b",
        r"\baccount will be closed\b",
        r"\bpay immediately\b",
        r"\byou will lose\b",
        r"\byou'?ll lose\b",
    ],
}

# ============================================================
# WEIGHTS
# ============================================================

SIGNAL_WEIGHTS = {
    "money_request": 0.35,
    "credential_request": 0.40,
    "financial_information": 0.30,
    "gift_card_request": 0.35,
    "urgency": 0.20,
    "secrecy": 0.25,
    "threat_or_pressure": 0.30,
}


# ============================================================
# DETECTOR
# ============================================================

def detect_scam(text):

    text = text.lower().strip()

    detected_signals = {}
    matched_phrases = []

    # Check every signal category
    for category, patterns in SIGNAL_PATTERNS.items():

        category_detected = False

        for pattern in patterns:

            match = re.search(pattern, text)

            if match:
                category_detected = True
                matched_phrases.append(match.group())
                break

        detected_signals[category] = category_detected

    # --------------------------------------------------------
    # Calculate weighted risk
    # --------------------------------------------------------

    score = 0.0

    for category, detected in detected_signals.items():

        if detected:
            score += SIGNAL_WEIGHTS[category]

    # Cap score at 1.0
    score = min(score, 1.0)

    # --------------------------------------------------------
    # Special combinations
    # --------------------------------------------------------

    money = detected_signals["money_request"]
    credential = detected_signals["credential_request"]
    financial = detected_signals["financial_information"]
    gift_card = detected_signals["gift_card_request"]
    urgency = detected_signals["urgency"]
    secrecy = detected_signals["secrecy"]
    threat = detected_signals["threat_or_pressure"]

    # Strong social-engineering combinations
    if (money or credential or financial or gift_card) and urgency:
        score = max(score, 0.75)

    if (money or credential or financial or gift_card) and secrecy:
        score = max(score, 0.80)

    if threat and (money or credential or financial):
        score = max(score, 0.85)

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if score >= 0.75:
        risk_level = "high"

    elif score >= 0.45:
        risk_level = "medium"

    elif score > 0:
        risk_level = "low"

    else:
        risk_level = "low"

    return {
        "urgency_detected": urgency,
        "signals": detected_signals,
        "matched_phrases": matched_phrases,
        "risk_score": round(score, 2),
        "risk_level": risk_level,
    }


# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":

    test_sentences = [

        # Normal conversations
        "Hey, are you coming home for dinner?",
        "Can you call me when you reach home?",
        "We should meet tomorrow.",

        # Money scams
        "Please send me the money right now.",
        "Transfer the money immediately.",

        # OTP scams
        "I need your OTP immediately.",
        "Tell me the verification code right now.",

        # Bank scams
        "Your bank account has a problem. Give me your verification code immediately.",

        # Gift card
        "Buy a gift card and send me the code as soon as possible.",

        # Secrecy
        "Don't tell anyone about this. Transfer the money right now.",

        # Threat
        "Your account will be blocked unless you make the payment immediately.",

        # More natural sounding scam
        "I'm in trouble and I need you to help me with a payment before the bank closes.",
    ]


    print("\n===== VOICEGUARD SCAM DETECTOR =====\n")


    for sentence in test_sentences:

        result = detect_scam(sentence)

        print("TEXT:", sentence)
        print("URGENCY:", result["urgency_detected"])
        print("SIGNALS:", result["signals"])
        print("MATCHED:", result["matched_phrases"])
        print("RISK:", result["risk_level"])
        print("SCORE:", result["risk_score"])
        print("-" * 70)