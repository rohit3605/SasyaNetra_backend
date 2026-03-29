def get_advice(disease_name):

    disease_name = disease_name.lower()

    advice_dict = {
        "brown spot": "Use fungicide spray. Remove infected leaves. Improve drainage.",
        "bacterial leaf blight": "Avoid excess nitrogen. Use resistant varieties. Maintain field hygiene.",
        "leaf blast": "Apply fungicide. Avoid water stress.",
        "leaf scald": "Ensure proper irrigation and avoid nutrient deficiency.",
        "leaf smut": "Use certified seeds and maintain field sanitation.",
        "narrow brown spot": "Apply balanced fertilizer and fungicide.",
        "sheath blight": "Improve air circulation and apply fungicide.",
        "healthy leaf": "Your crop is healthy. Maintain proper care."
    }

    return advice_dict.get(disease_name, "Consult an agricultural expert for proper guidance.")