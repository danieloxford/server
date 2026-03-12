from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import os
from groq import Groq
from dotenv import load_dotenv
import io
import json
import httpx
import traceback
import socket
import base64 as b64lib

load_dotenv()

# =========================
# AI CHAT (Groq)
# =========================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# GEMINI CONFIG
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash-lite"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# =========================
# Load TFLite model (offline fallback + pre-screening)
# =========================
interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

app = FastAPI(title="DermAware Backend v4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# Thresholds
# =========================
CONFIDENCE_THRESHOLD = 0.40
NOT_SKIN_THRESHOLD   = 0.30

HEALTHY_LABELS = {
    "normal skin", "healthy", "healthy skin", "normal",
    "clear skin", "no disease"
}

NOT_SKIN_LABELS = {
    "not skin", "not_skin", "invalid", "background",
    "no skin", "other"
}

# =========================
# Utils
# =========================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def preprocess_image(image: Image.Image):
    image       = image.resize((224, 224), Image.LANCZOS)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def normalize_label(label: str) -> str:
    clean = label.lower().strip()
    if clean in HEALTHY_LABELS or any(h in clean for h in ["healthy", "normal", "clear"]):
        return "healthy"
    if clean in NOT_SKIN_LABELS or any(n in clean for n in ["not skin", "not_skin", "invalid"]):
        return "not_skin"
    return label.replace("-", " ").replace("_", " ").title()

# =========================
# GEMINI — Online Classification
# =========================
@app.post("/classify/gemini")
async def classify_gemini(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not configured on server")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(400, "Empty file received")

        try:
            Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"Invalid image: {str(e)}")

        image_b64 = b64lib.b64encode(contents).decode("utf-8")

        prompt = (
            "You are a professional AI dermatology screening assistant. "
            "Analyze this image carefully and respond ONLY in one of the following JSON formats — no markdown, no extra text.\n\n"

            "━━━ FORMAT 1: Not human skin ━━━\n"
            "{\"type\":\"NOT_SKIN\",\"reason\":\"brief reason why this is not skin\"}\n\n"
            "Use ONLY when the image clearly shows: animals, food, objects, plants, text, or non-human material.\n\n"

            "━━━ FORMAT 2: Healthy human skin ━━━\n"
            "{\"type\":\"HEALTHY\"}\n\n"
            "Use when you can clearly see human skin with NO visible condition, rash, lesion, discoloration, or abnormality.\n\n"

            "━━━ FORMAT 3: Skin condition detected ━━━\n"
            "{\n"
            "  \"type\": \"CONDITION\",\n"
            "  \"label\": \"Precise medical condition name\",\n"
            "  \"alsoKnownAs\": \"Most popular common name regular people use\",\n"
            "  \"confidence\": 0.85,\n"
            "  \"severity\": \"mild|moderate|severe\",\n"
            "  \"explanation\": \"2-3 sentence patient-friendly explanation\",\n"
            "  \"symptoms\": [\"symptom 1\", \"symptom 2\", \"symptom 3\", \"symptom 4\", \"symptom 5\"],\n"
            "  \"causes\": [\"cause 1\", \"cause 2\", \"cause 3\", \"cause 4\"],\n"
            "  \"dos\": [\"actionable do 1\", \"actionable do 2\", \"actionable do 3\", \"actionable do 4\"],\n"
            "  \"donts\": [\"actionable don't 1\", \"actionable don't 2\", \"actionable don't 3\", \"actionable don't 4\"],\n"
            "  \"whenToSeeDoctor\": \"One clear sentence about when this specifically needs professional medical attention\"\n"
            "}\n\n"

            "━━━ RULES ━━━\n"
            "• ANY human body part → never NOT_SKIN\n"
            "• When unsure between HEALTHY and CONDITION → always CONDITION\n"
            "• alsoKnownAs: use the most popular everyday name (e.g. 'Ringworm' not 'Tinea Corporis')\n"
            "• confidence: 0.0–1.0\n"
            "• severity: mild / moderate / severe\n"
            "• symptoms/causes/dos/donts: 3–6 items each\n"
            "• explanation: simple language, no heavy jargon\n"
            "• Reply ONLY valid JSON — no markdown fences, no preamble"
        )

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": 700,
                "temperature": 0.1,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(GEMINI_URL, json=payload)

        if response.status_code != 200:
            print(f"❌ Gemini HTTP error: {response.status_code} — {response.text[:300]}")
            raise HTTPException(response.status_code, f"Gemini API error: {response.text[:200]}")

        data  = response.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])

        raw_text = (
            next((p["text"] for p in reversed(parts) if not p.get("thought") and "text" in p), None)
            or (parts[-1].get("text", "") if parts else "")
        ).strip()

        if not raw_text:
            raise HTTPException(500, "Empty response from Gemini")

        parsed = json.loads(raw_text.replace("```json", "").replace("```", "").strip())

        if parsed.get("type") == "NOT_SKIN":
            return JSONResponse({"type": "NOT_SKIN", "reason": parsed.get("reason", "Not human skin")})

        if parsed.get("type") == "HEALTHY":
            return JSONResponse({"type": "HEALTHY"})

        if parsed.get("type") == "CONDITION":
            return JSONResponse({
                "type":            "CONDITION",
                "label":           parsed.get("label", "Skin Condition"),
                "alsoKnownAs":     parsed.get("alsoKnownAs", parsed.get("label", "")),
                "confidence":      float(parsed.get("confidence", 0.70)),
                "severity":        parsed.get("severity", "mild") if parsed.get("severity") in ["mild","moderate","severe"] else "mild",
                "explanation":     parsed.get("explanation", ""),
                "symptoms":        parsed.get("symptoms", [])  if isinstance(parsed.get("symptoms"), list) else [],
                "causes":          parsed.get("causes", [])    if isinstance(parsed.get("causes"),   list) else [],
                "dos":             parsed.get("dos", [])        if isinstance(parsed.get("dos"),       list) else [],
                "donts":           parsed.get("donts", [])      if isinstance(parsed.get("donts"),     list) else [],
                "whenToSeeDoctor": parsed.get("whenToSeeDoctor", "")
            })

        return JSONResponse({"type": "ERROR"})

    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error from Gemini: {e}")
        return JSONResponse({"type": "ERROR"})
    except httpx.TimeoutException:
        raise HTTPException(504, "Gemini request timed out")
    except httpx.RequestError as e:
        raise HTTPException(503, f"Network error reaching Gemini: {str(e)}")
    except Exception as e:
        print(f"❌ Gemini classify error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Gemini classification failed: {str(e)}")


# =========================
# OFFLINE — TFLite Direct Classifier
# =========================
@app.post("/classify/offline")
async def classify_offline(file: UploadFile = File(...)):
    try:
        print(f"📥 Offline: {file.filename}")

        contents   = await file.read()
        image      = Image.open(io.BytesIO(contents)).convert("RGB")
        input_data = preprocess_image(image)

        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])[0]

        top3_idx = np.argsort(output_data)[::-1][:3]
        print("Top 3 predictions:")
        for idx in top3_idx:
            print(f"  [{idx}] {labels[idx]}: {output_data[idx]:.2%}")

        max_idx    = int(np.argmax(output_data))
        raw_label  = labels[max_idx]
        confidence = float(output_data[max_idx])
        normalized = normalize_label(raw_label)

        if confidence < NOT_SKIN_THRESHOLD:
            return JSONResponse({"label": "Not Skin", "confidence": confidence, "error": "low_confidence"})
        if normalized == "not_skin":
            return JSONResponse({"label": "Not Skin", "confidence": confidence, "error": "not_skin"})
        if normalized == "healthy":
            return JSONResponse({"label": "Healthy Skin", "confidence": confidence})
        if confidence < CONFIDENCE_THRESHOLD:
            return JSONResponse({"label": normalized, "confidence": confidence, "warning": "low_confidence_result"})

        return JSONResponse({"label": normalized, "confidence": confidence})

    except Exception as e:
        print(f"❌ Offline error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Offline failed: {str(e)}")


# =========================
# UNIFIED ENDPOINT
# =========================
@app.post("/classify")
async def classify_unified(file: UploadFile = File(...), mode: str = Form("gemini")):
    print(f"📍 Mode: {mode}")
    if mode.lower() == "offline":
        return await classify_offline(file)
    return await classify_gemini(file)


# =========================
# AI CHAT — With Conversation History
# =========================
class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        print(f"💬 Chat: {req.message[:50]}...")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are DermAware's friendly dermatology assistant.\n\n"
                    "YOUR EXPERTISE:\n"
                    "- Skin health, skincare, and dermatology\n"
                    "- Skin conditions, symptoms, and general care\n"
                    "- Sun protection, moisturizing, and basic skincare routines\n"
                    "- When to see a doctor for skin issues\n\n"
                    "CONVERSATION GUIDELINES:\n"
                    "- Allow natural conversation flow (greetings, thanks, follow-ups)\n"
                    "- Respond naturally to 'okay', 'thanks', 'I see', 'ahhh'\n"
                    "- Be friendly and conversational\n\n"
                    "TOPIC RESTRICTIONS:\n"
                    "- REFUSE: politics, sports, cooking, math, coding, general knowledge\n"
                    "- For off-topic: \"I'm DermAware's skin health assistant and can only help with skin-related topics.\"\n\n"
                    "MEDICAL DISCLAIMERS:\n"
                    "- Never diagnose — only provide general information\n"
                    "- Always recommend consulting healthcare professionals for serious concerns"
                )
            }
        ]

        for msg in req.history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        messages.append({"role": "user", "content": req.message})

        response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
        reply    = response.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(500, f"Chat failed: {str(e)}")


# =========================
# EXPLAIN RESULT
# =========================
class ExplainRequest(BaseModel):
    label: str

@app.post("/explain_result")
def explain_result(req: ExplainRequest):
    try:
        condition = req.label.lower().strip()

        if "not skin" in condition:
            return {
                "explanation": "The image doesn't appear to contain skin. Please take a clear photo of skin.",
                "causes": [],
                "dos":    ["Ensure good lighting", "Focus on skin area", "Keep camera steady"],
                "donts":  ["Don't photograph non-skin areas", "Avoid blurry images"]
            }

        if "healthy" in condition:
            return {
                "explanation": "Your skin appears healthy! Continue good skincare habits.",
                "causes": [],
                "dos":    ["Maintain skincare routine", "Stay hydrated", "Use SPF 30+ daily", "Get 7-9 hours sleep"],
                "donts":  ["Don't skip sunscreen", "Don't over-wash", "Don't pick at skin"]
            }

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant. Use SIMPLE, EVERYDAY language.\n"
                        "Provide info in JSON:\n"
                        "- explanation: 2-3 sentences simple English\n"
                        "- causes: 3-5 common causes\n"
                        "- dos: 3-5 care recommendations\n"
                        "- donts: 3-5 things to avoid\n"
                        "Return ONLY valid JSON."
                    )
                },
                {"role": "user", "content": f"Explain in simple terms: {req.label}"}
            ],
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)
        return {
            "explanation": data.get("explanation", "No info available"),
            "causes":      data.get("causes", []),
            "dos":         data.get("dos", []),
            "donts":       data.get("donts", [])
        }

    except Exception as e:
        print(f"❌ Explain error: {e}")
        raise HTTPException(500, f"Explanation failed: {str(e)}")


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {
        "status":  "ok",
        "service": "DermAware Backend v4.0",
        "gemini":  "✅" if GEMINI_API_KEY else "❌ Missing GEMINI_API_KEY",
        "groq":    "✅" if os.getenv("GROQ_API_KEY") else "❌ Missing GROQ_API_KEY",
        "tflite":  f"✅ ({len(labels)} classes)",
        "endpoints": {
            "gemini":  "POST /classify/gemini",
            "offline": "POST /classify/offline",
            "unified": "POST /classify",
            "chat":    "POST /chat",
            "explain": "POST /explain_result",
        }
    }

@app.get("/debug")
def debug():
    return {
        "gemini_key": "✅ Set" if GEMINI_API_KEY else "❌ Missing",
        "groq_key":   "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Missing",
        "labels_count": len(labels),
        "sample_labels": labels[:5],
    }

@app.get("/ui", response_class=HTMLResponse)
def ui():
    try:
        with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h1>UI not available</h1>"


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    print("DermAware Backend v4.0 Starting...")
    print(f"Local:   http://{local_ip}:8000")
    print(f"Gemini:  {'✅' if GEMINI_API_KEY else '❌ Missing GEMINI_API_KEY'}")
    print(f"Groq:    {'✅' if os.getenv('GROQ_API_KEY') else '❌ Missing GROQ_API_KEY'}")
    print(f"TFLite:  ✅ ({len(labels)} classes)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
