# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io, os, json
# Ensure Keras uses TensorFlow backend (required for loading .keras models trained with TF)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
from PIL import Image
import numpy as np
import cv2
import keras
from keras.models import load_model



app = FastAPI(title="Traffic Sign Classifier")


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_traffic_model.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)
LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.json")

# load labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)
# normalize to list indexable by int id
if isinstance(labels, dict):
    # ensure ordering by int key
    labels = [labels[str(i)] for i in range(len(labels))]

# Lazy-load model to avoid crashing on startup and to provide clearer errors
model = None

def get_model():
    global model
    if model is not None:
        return model
    # Check existence first for clearer message
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Model dosyası bulunamadı: {MODEL_PATH}. Lütfen dosyanın mevcut olduğundan ve uygulamanın bu yola erişebildiğinden emin olun."
        )
    try:
        # compile=False: faster load, prevents unnecessary compile errors
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model yüklenemedi: {e}. Yol: {MODEL_PATH}"
        )

def compute_status(conf):
    if conf >= 0.85:
        return "HIGH"
    elif conf >= 0.65:
        return "MEDIUM"
    else:
        return "LOW"

def preprocess_image_bytes(file_bytes):
    # Image -> numpy, aynı Colab pipeline'ı (48x48, CLAHE, normalize)
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = np.array(image)
    img_resized = cv2.resize(img, (48, 48))
    img_yuv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    img_norm = img_enhanced.astype("float32") / 255.0
    return np.expand_dims(img_norm, axis=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # basic validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece resim dosyası yükleyin.")
    contents = await file.read()
    try:
        x = preprocess_image_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {e}")
    # Ensure model is loaded and provide clearer error if not
    mdl = get_model()
    preds = mdl.predict(x)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    top3_probs = preds[top3_idx].tolist()
    top3 = [
        {"class_id": int(i), "label": labels[int(i)], "probability": float(p)}
        for i, p in zip(top3_idx, top3_probs)
    ]
    class_id = int(top3_idx[0])
    confidence = float(top3_probs[0])
    status = compute_status(confidence)
    return JSONResponse({
        "predicted_class_id": class_id,
        "predicted_label": labels[class_id],
        "confidence": confidence,
        "status": status,
        "top3": top3
    })
