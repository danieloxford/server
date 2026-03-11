from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
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


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash-lite"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

app = FastAPI(title="DermAware Backend v3.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

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

def tflite_prescreen(image: Image.Image) -> dict:
    try:
        input_data = preprocess_image(image)
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])[0]

        top3_idx = np.argsort(output_data)[::-1][:3]
        print("[SCAN] TFLite pre-screen top-3:")
        for idx in top3_idx:
            print(f"       [{idx}] {labels[idx]}: {output_data[idx]:.2%}")

        max_idx    = int(np.argmax(output_data))
        raw_label  = labels[max_idx]
        confidence = float(output_data[max_idx])
        normalized = normalize_label(raw_label)

        if confidence < NOT_SKIN_THRESHOLD:
            return {
                "passed":     False,
                "reason":     "low_confidence",
                "message":    "Image is too blurry or unclear. Please take a clearer photo with better lighting.",
                "confidence": confidence
            }

        if normalized == "not_skin":
            return {
                "passed":     False,
                "reason":     "not_skin",
                "message":    "This doesn't appear to be a skin image. Please upload a photo of the affected skin area.",
                "confidence": confidence
            }

        print(f"[OK]   Pre-screen PASSED ({confidence:.2%})")
        return { "passed": True, "confidence": confidence }

    except Exception as e:
        print(f"[WARN] TFLite pre-screen error (allowing API anyway): {e}")
        return { "passed": True, "confidence": 0.0 }


def get_skin_info_from_groq(label: str):
    try:
        print(f"[AI]   Asking Groq for details about: {label}")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant. "
                        "Provide info in JSON format:\n"
                        "- alsoKnownAs: The SHORTEST, SIMPLEST single name only — no slashes, no extra words, no medical terms. "
                        "One word or two at most. Examples: 'Acne', 'Ringworm', 'Warts', 'Moles', 'Hives', 'Eczema', 'Psoriasis'.\n"
                        "- explanation: 2-3 sentences simple definition in everyday English\n"
                        "- causes: 3-5 common causes in simple language\n"
                        "- dos: 3-5 care recommendations in simple, actionable language\n"
                        "- donts: 3-5 things to avoid in simple language\n"
                        "Return ONLY valid JSON. Use simple, everyday language throughout."
                    )
                },
                {
                    "role": "user",
                    "content": f"What is the simplest common name for '{label}' and explain this skin condition in simple terms."
                }
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data    = json.loads(content)
        return {
            "alsoKnownAs": data.get("alsoKnownAs", label),
            "explanation": data.get("explanation", "Information currently unavailable."),
            "causes":      data.get("causes", []),
            "dos":         data.get("dos", []),
            "donts":       data.get("donts", [])
        }
    except Exception as e:
        print(f"[ERROR] Groq info fetch failed: {e}")
        return {
            "alsoKnownAs": label,
            "explanation": "Information currently unavailable.",
            "causes":      [],
            "dos":         [],
            "donts":       []
        }

class GcashVerifyRequest(BaseModel):
    image_base64:    str
    mime_type:       str = "image/jpeg"
    expected_amount: str

@app.post("/verify/gcash-screenshot")
async def verify_gcash_screenshot(req: GcashVerifyRequest):
    """
    Verifies a GCash payment screenshot using Gemini Vision.
    The mobile app sends the image here; the API key never leaves the server.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not configured on server")

    SUPPORTED_MIME = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    safe_mime = req.mime_type if req.mime_type in SUPPORTED_MIME else "image/jpeg"

    raw_b64 = req.image_base64
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[-1]

    print(f"[PAY]  GCash verify request | amount={req.expected_amount} | mime={safe_mime}")

    try:
        payload = {
            "contents": [{"parts": [
                {
                    "inline_data": {
                        "mime_type": safe_mime,
                        "data": raw_b64
                    }
                },
                {
                    "text": (
                        f"You are a payment verification assistant. "
                        f"Analyze this GCash screenshot. Expected amount: {req.expected_amount}. "
                        f"Check: (1) Is this a genuine GCash transaction receipt? "
                        f"(2) What is the reference number? "
                        f"(3) What amount was sent? "
                        f"(4) Does it match {req.expected_amount}? "
                        f"Respond ONLY with a raw JSON object — no markdown, no backticks, no extra text: "
                        f'{{ "isGcash": boolean, "confidence": "high"|"medium"|"low", '
                        f'"extractedRef": string|null, "extractedAmount": string|null, '
                        f'"amountMatches": boolean|null, "reason": string }}'
                    )
                },
            ]}],
            "generationConfig": {
                "temperature":     0,
                "maxOutputTokens": 300
            }
        }

        async with httpx.AsyncClient(timeout=30.0) as http:
            response = await http.post(GEMINI_URL, json=payload)

        if response.status_code != 200:
            print(f"[ERROR] Gemini responded {response.status_code} — {response.text[:300]}")
            raise HTTPException(response.status_code, f"Gemini API error: {response.text[:200]}")

        data    = response.json()
        raw     = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        result  = json.loads(cleaned)

        print(
            f"[OK]   GCash verify done | "
            f"isGcash={result.get('isGcash')} | "
            f"confidence={result.get('confidence')} | "
            f"ref={result.get('extractedRef')} | "
            f"amount={result.get('extractedAmount')}"
        )

        return JSONResponse({
            "isGcash":         bool(result.get("isGcash", False)),
            "confidence":      result.get("confidence", "low"),
            "extractedRef":    result.get("extractedRef"),
            "extractedAmount": result.get("extractedAmount"),
            "amountMatches":   result.get("amountMatches"),
            "reason":          result.get("reason", "")
        })

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed (GCash verify): {e}")
        return JSONResponse({
            "isGcash":         False,
            "confidence":      "low",
            "extractedRef":    None,
            "extractedAmount": None,
            "amountMatches":   None,
            "reason":          "Could not auto-verify. Admin will review manually."
        })
    except httpx.TimeoutException:
        raise HTTPException(504, "Gemini request timed out")
    except httpx.RequestError as e:
        raise HTTPException(503, f"Network error reaching Gemini: {str(e)}")
    except Exception as e:
        print(f"[ERROR] GCash verify exception: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Verification failed: {str(e)}")

@app.post("/classify/gemini")
async def classify_gemini(file: UploadFile = File(...)):
    """
    Gemini-powered skin analysis.
    The mobile app sends the image file here; this endpoint calls Gemini
    so the API key stays securely on the server.
    """
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
        print(f"[SCAN] Gemini classify | file={file.filename} | size={len(contents)} bytes")

        prompt = (
            "You are a professional AI dermatology screening assistant. "
            "Analyze this image carefully and respond ONLY in one of the following JSON formats — no markdown, no extra text.\n\n"

            "--- FORMAT 1: Not human skin ---\n"
            "{\"type\":\"NOT_SKIN\",\"reason\":\"brief reason why this is not skin\"}\n\n"
            "Use ONLY when the image clearly shows: animals, food, objects, plants, text, or non-human material.\n\n"

            "--- FORMAT 2: Healthy human skin ---\n"
            "{\"type\":\"HEALTHY\"}\n\n"
            "Use when you can clearly see human skin with NO visible condition, rash, lesion, discoloration, or abnormality.\n\n"

            "--- FORMAT 3: Skin condition detected ---\n"
            "{\n"
            "  \"type\": \"CONDITION\",\n"
            "  \"label\": \"Precise medical condition name\",\n"
            "  \"alsoKnownAs\": \"Shortest simplest everyday word only\",\n"
            "  \"confidence\": 0.85,\n"
            "  \"severity\": \"mild|moderate|severe\",\n"
            "  \"explanation\": \"2-3 sentence patient-friendly explanation\",\n"
            "  \"symptoms\": [\"symptom 1\", \"symptom 2\", \"symptom 3\", \"symptom 4\", \"symptom 5\"],\n"
            "  \"causes\": [\"cause 1\", \"cause 2\", \"cause 3\", \"cause 4\"],\n"
            "  \"dos\": [\"actionable do 1\", \"actionable do 2\", \"actionable do 3\", \"actionable do 4\"],\n"
            "  \"donts\": [\"actionable don't 1\", \"actionable don't 2\", \"actionable don't 3\", \"actionable don't 4\"],\n"
            "  \"whenToSeeDoctor\": \"One clear sentence about when this specifically needs professional medical attention\"\n"
            "}\n\n"

            "--- RULES ---\n"
            "- ANY human body part -> never NOT_SKIN\n"
            "- When unsure between HEALTHY and CONDITION -> always CONDITION\n"
            "- alsoKnownAs: use ONLY the shortest, simplest everyday word people say — just the condition name, nothing else. "
            "Examples: 'Acne' not 'Acne & Rosacea', 'Moles' not 'Melanoma / Suspicious Mole', 'Ringworm' not 'Ringworm / Fungal Infection', "
            "'Warts' not 'Warts / Viral Infection', 'Eczema' not 'Atopic Dermatitis', 'Hives' not 'Urticaria'. "
            "No slashes, no extra context, no medical terms. One or two words maximum.\n"
            "- confidence: 0.0-1.0 (your honest estimate)\n"
            "- severity: mild (minor, self-manageable), moderate (needs attention), severe (urgent care needed)\n"
            "- symptoms/causes/dos/donts: 3-6 items each, specific and actionable\n"
            "- explanation: simple language, avoid heavy jargon\n"
            "- whenToSeeDoctor: be specific to this condition, not generic advice\n"
            "- Reply ONLY valid JSON — no markdown fences, no preamble"
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
                "temperature":     0.1,
                "thinkingConfig":  {"thinkingBudget": 0}
            }
        }

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(GEMINI_URL, json=payload)

        if response.status_code != 200:
            print(f"[ERROR] Gemini HTTP {response.status_code} — {response.text[:300]}")
            raise HTTPException(response.status_code, f"Gemini API error: {response.text[:200]}")

        data  = response.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])

        raw_text = (
            next((p["text"] for p in reversed(parts) if not p.get("thought") and "text" in p), None)
            or (parts[-1].get("text", "") if parts else "")
        ).strip()

        if not raw_text:
            raise HTTPException(500, "Empty response from Gemini")

        parsed      = json.loads(raw_text.replace("```json", "").replace("```", "").strip())
        result_type = parsed.get("type", "ERROR")
        print(f"[OK]   Gemini classify result: type={result_type}")

        if result_type == "NOT_SKIN":
            return JSONResponse({
                "type":   "NOT_SKIN",
                "reason": parsed.get("reason", "Not human skin")
            })

        if result_type == "HEALTHY":
            return JSONResponse({"type": "HEALTHY"})

        if result_type == "CONDITION":
            return JSONResponse({
                "type":            "CONDITION",
                "label":           parsed.get("label", "Skin Condition"),
                "alsoKnownAs":     parsed.get("alsoKnownAs", parsed.get("label", "")),
                "confidence":      float(parsed.get("confidence", 0.70)),
                "severity":        parsed.get("severity", "mild") if parsed.get("severity") in ["mild", "moderate", "severe"] else "mild",
                "explanation":     parsed.get("explanation", ""),
                "symptoms":        parsed.get("symptoms", []) if isinstance(parsed.get("symptoms"), list) else [],
                "causes":          parsed.get("causes", [])   if isinstance(parsed.get("causes"),   list) else [],
                "dos":             parsed.get("dos", [])       if isinstance(parsed.get("dos"),       list) else [],
                "donts":           parsed.get("donts", [])     if isinstance(parsed.get("donts"),     list) else [],
                "whenToSeeDoctor": parsed.get("whenToSeeDoctor", "")
            })

        return JSONResponse({"type": "ERROR"})

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed (Gemini classify): {e}")
        return JSONResponse({"type": "ERROR"})
    except httpx.TimeoutException:
        raise HTTPException(504, "Gemini request timed out")
    except httpx.RequestError as e:
        raise HTTPException(503, f"Network error reaching Gemini: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Gemini classify exception: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Gemini classification failed: {str(e)}")

@app.post("/classify/offline")
async def classify_offline(file: UploadFile = File(...)):
    try:
        print(f"[SCAN] Offline classify | file={file.filename}")

        contents   = await file.read()
        image      = Image.open(io.BytesIO(contents)).convert("RGB")
        input_data = preprocess_image(image)

        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details["index"])[0]

        top3_idx = np.argsort(output_data)[::-1][:3]
        print("[SCAN] Top 3 predictions:")
        for idx in top3_idx:
            print(f"       [{idx}] {labels[idx]}: {output_data[idx]:.2%}")

        max_idx    = int(np.argmax(output_data))
        raw_label  = labels[max_idx]
        confidence = float(output_data[max_idx])
        normalized = normalize_label(raw_label)

        if confidence < NOT_SKIN_THRESHOLD:
            return JSONResponse({ "label": "Not Skin", "confidence": confidence, "error": "low_confidence" })
        if normalized == "not_skin":
            return JSONResponse({ "label": "Not Skin", "confidence": confidence, "error": "not_skin" })
        if normalized == "healthy":
            return JSONResponse({ "label": "Healthy Skin", "confidence": confidence })
        if confidence < CONFIDENCE_THRESHOLD:
            return JSONResponse({ "label": normalized, "confidence": confidence, "warning": "low_confidence_result" })

        return JSONResponse({ "label": normalized, "confidence": confidence })

    except Exception as e:
        print(f"[ERROR] Offline classify failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(500, f"Offline failed: {str(e)}")


@app.post("/classify")
async def classify_unified(file: UploadFile = File(...), mode: str = Form("gemini")):
    print(f"[ROUTE] Mode: {mode}")
    if mode.lower() == "gemini":
        return await classify_gemini(file)
    else:
        return await classify_offline(file)


class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        print(f"[CHAT] Message: {req.message[:50]}...")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are DermAware's friendly dermatology assistant.\n\n"
                    "YOUR EXPERTISE:\n"
                    "- Skin health, skincare, and dermatology\n"
                    "- Skin conditions, symptoms, and general care\n"
                    "- Sun protection, moisturizing, and basic skincare routines\n"
                    "- When to see a doctor for skin issues\n"
                    "- General skin anatomy and function\n\n"
                    "CONVERSATION GUIDELINES:\n"
                    "- ALLOW natural conversation flow (greetings, thanks, acknowledgments, follow-ups)\n"
                    "- Respond naturally to casual responses like 'okay', 'thanks', 'I see', 'ahhh'\n"
                    "- Be friendly and conversational when discussing skin topics\n"
                    "- Remember conversation context - if discussing a topic, stay engaged\n"
                    "- Accept follow-up questions about topics already being discussed\n\n"
                    "TOPIC RESTRICTIONS (only enforce for NEW topic requests):\n"
                    "- REFUSE questions about: politics, sports, cooking, math, coding, general knowledge\n"
                    "- REFUSE medical advice for non-skin conditions\n"
                    "- For off-topic questions, say: \"I'm DermAware's skin health assistant and can only help "
                    "with questions about skin, skincare, and dermatology.\"\n\n"
                    "MEDICAL DISCLAIMERS:\n"
                    "- Never diagnose - only provide general information\n"
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
        print(f"[ERROR] Chat failed: {e}")
        raise HTTPException(500, f"Chat failed: {str(e)}")


class ExplainRequest(BaseModel):
    label: str

@app.post("/explain_result")
def explain_result(req: ExplainRequest):
    try:
        condition = req.label.lower().strip()
        print(f"[EXPLAIN] Label: {req.label}")

        if "not skin" in condition:
            return {
                "explanation": "The image doesn't appear to contain skin. Please take a clear photo of skin.",
                "causes": [],
                "dos":    ["Ensure good lighting", "Focus on skin area", "Keep camera steady", "Take from 6-12 inches away"],
                "donts":  ["Don't take photos of non-skin", "Avoid blurry images", "Don't include too much background", "Avoid extreme close-ups"]
            }

        if "healthy" in condition:
            return {
                "explanation": "Your skin appears healthy! Continue good skincare habits.",
                "causes": [],
                "dos":    ["Maintain skincare routine", "Stay hydrated", "Use SPF 30+ daily", "Get 7-9 hours sleep"],
                "donts":  ["Don't skip sunscreen", "Don't over-wash", "Don't pick at skin", "Avoid harsh products"]
            }

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dermatology assistant. Use SIMPLE, EVERYDAY language.\n"
                        "Provide info in JSON format:\n"
                        "- explanation: 2-3 sentences in simple English with reminder to see doctor\n"
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
        print(f"[ERROR] Explain failed: {e}")
        raise HTTPException(500, f"Explanation failed: {str(e)}")


@app.get("/")
def health():
    return {
        "status":  "ok",
        "service": "DermAware Backend v3.3",
        "features": [
            "Gemini Vision - skin classification (server-side key)",
            "Gemini Vision - GCash screenshot verification (server-side key)",
            "TFLite Pre-screen (offline fallback)",
            "DermNet 23 Classes",
            "Scientific + Popular Names via Groq",
            "AI Chat (Groq)",
        ],
        "gemini": "OK"      if GEMINI_API_KEY            else "MISSING",
        "groq":   "OK"      if os.getenv("GROQ_API_KEY") else "MISSING",
        "tflite": "OK",
        "model_classes": len(labels),
        "endpoints": {
            "health":          "GET  /",
            "debug":           "GET  /debug",
            "classify_gemini": "POST /classify/gemini    - Gemini skin analysis",
            "classify_offline":"POST /classify/offline   - TFLite offline",
            "classify_unified":"POST /classify           - Auto-routes (default: gemini)",
            "gcash_verify":    "POST /verify/gcash-screenshot - GCash receipt AI check",
            "chat":            "POST /chat",
            "explain":         "POST /explain_result",
        }
    }


@app.get("/ui", response_class=HTMLResponse)
def ui():
    try:
        with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h1>UI not available</h1>"


@app.get("/debug")
def debug():
    return {
        "gemini_key":   "OK - Set" if GEMINI_API_KEY            else "MISSING",
        "groq_key":     "OK - Set" if os.getenv("GROQ_API_KEY") else "MISSING",
        "labels_count": len(labels),
        "sample_labels": labels[:5],
    }



if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    print("=" * 52)
    print("  DermAware Backend v3.3")
    print("=" * 52)
    print(f"  Local Network : http://{local_ip}:8000")
    print(f"  Localhost     : http://127.0.0.1:8000")
    print(f"  Gemini        : {'[OK]' if GEMINI_API_KEY else '[MISSING] GEMINI_API_KEY not set'}")
    print(f"  Groq          : {'[OK]' if os.getenv('GROQ_API_KEY') else '[MISSING] GROQ_API_KEY not set'}")
    print(f"  TFLite        : [OK] ({len(labels)} classes)")
    print("=" * 52)
    uvicorn.run(app, host="0.0.0.0", port=8000)
