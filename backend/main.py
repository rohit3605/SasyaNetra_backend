from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.ai_advisor import get_advice
from PIL import Image
import io

from backend.model import predict_image
from backend.utils import read_image
from backend.utils import analyze_text

app = FastAPI(title="🌾 Rice Disease Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ROOT (CHECK SERVER) ----------
@app.get("/")
def home():
    return {"message": "API is running successfully 🚀"}

# ---------- IMAGE API ----------
@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):
    try:
        # Read image using utils
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Predict
        label, confidence = predict_image(image)
        advice = get_advice(label)  
        return JSONResponse(content={
            "disease": label,
            "confidence": round(confidence * 100, 2) , 
            "advice": advice
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# ---------- TEXT API ----------
@app.post("/analyze-text")
async def analyze_text_api(text: str = Form(...)):
    try:
        # ✅ USE YOUR UTILS FUNCTION
        label, confidence = analyze_text(text)
        advice = get_advice(label)
        return JSONResponse(content={
            "result": label,
            "confidence": round(confidence * 100, 2),
            "advice": advice
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )