from PIL import Image

# ---------- IMAGE READER ----------
def read_image(file):
    return Image.open(file).convert("RGB")


# ---------- TEXT ANALYSIS ----------
def analyze_text(text):
    text = text.lower()

    score_map = {}

    # ---------- KEYWORD RULES ----------

    rules = {
        "Bacterial Blight": ["yellow", "wilting", "dry edge", "leaf drying"],
        "Brown Spot": ["brown spot", "brown spots", "small spots", "dark spots"],
        "Leaf Blast": ["blast", "diamond shape", "gray center", "lesion"],
        "Leaf Scald": ["scald", "burn", "dry tip", "dry edges"],
        "Leaf Smut": ["black powder", "soot", "black spots"],
        "Narrow Brown Spot": ["narrow", "thin spots", "linear spots"],
        "Sheath Blight": ["sheath", "stem rot", "fungus on stem"],
        "Healthy": ["healthy", "no issue", "green", "normal"]
    }

    # ---------- SCORING ----------
    for disease, keywords in rules.items():
        score = 0
        for word in keywords:
            if word in text:
                score += 1
        if score > 0:
            score_map[disease] = score

    # ---------- DECISION ----------
    if score_map:
        best_disease = max(score_map, key=score_map.get)
        confidence = min(0.6 + (score_map[best_disease] * 0.1), 0.95)
        return best_disease, confidence

    # ---------- FALLBACK ----------
    return "General Crop Stress", 0.5