"""
image_analyzer.py — Medicine / Tablet Image Analysis
Uses google-genai SDK (Python 3.14 compatible) with Gemini Vision.
"""

import os
import base64
from dotenv import load_dotenv
from PIL import Image
import io

from google import genai
from google.genai import types

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = "gemini-3-flash-preview"

client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


def analyze_medicine_image(image_bytes: bytes) -> dict:
    """
    Analyze a tablet or medicine image using Gemini Vision.
    Returns identification, common use, safety notes, expiry reminder, confidence.
    """
    if not client:
        return _image_error("API key not configured. Add GOOGLE_API_KEY to .env file.")
    if not image_bytes:
        return _image_error("No image data received.")

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        image_data = buffer.getvalue()
    except Exception as e:
        return _image_error(f"Could not process image: {str(e)}")

    system_prompt = """You are a medicine identification assistant under strict safety rules.

RULES:
1. Analyze ONLY what is visibly in the image (text, imprints, packaging, shape, color).
2. Identify a medicine ONLY IF the imprint, brand, or generic name is clearly readable.
3. If unclear or unreadable → say exactly: "The medicine cannot be reliably identified from this image."
4. DO NOT suggest dosages or advise starting/stopping medication.
5. ALWAYS remind to check expiry date and consult a pharmacist.
6. If not a medicine image → say so clearly.

RESPONSE FORMAT (use exactly these headers):
**Medicine Identification:**
**Common Use (if identified):**
**Safety Notes:**
**Expiry Reminder:**
**Confidence Level:**"""

    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=[
                types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                "Analyze this medicine image following all rules strictly.",
            ],
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        raw_text = response.text.strip()
        parsed = _parse_image_response(raw_text)
        parsed["success"] = True
        parsed["raw"] = raw_text
        return parsed
    except Exception as e:
        return _image_error(f"Image analysis failed: {str(e)}")


def _parse_image_response(text: str) -> dict:
    result = {
        "identified_medicine": None,
        "common_use": "",
        "safety_notes": "",
        "expiry_reminder": "Always check the expiry date on medicine packaging before use. Do not use expired medicines.",
        "confidence": "Low",
        "message": text,
    }
    headers = {
        "identified_medicine": "Medicine Identification:",
        "common_use": "Common Use (if identified):",
        "safety_notes": "Safety Notes:",
        "expiry_reminder": "Expiry Reminder:",
        "confidence": "Confidence Level:",
    }
    for key, header in headers.items():
        for marker in [f"**{header}**", header]:
            idx = text.find(marker)
            if idx != -1:
                start = idx + len(marker)
                next_idx = len(text)
                for ok, oh in headers.items():
                    if ok == key:
                        continue
                    for m in [f"**{oh}**", oh]:
                        ni = text.find(m, start)
                        if ni != -1 and ni < next_idx:
                            next_idx = ni
                result[key] = text[start:next_idx].strip()
                break

    # If medicine wasn't identified
    med = result.get("identified_medicine") or ""
    if not med or any(p in med.lower() for p in ["cannot be reliably", "unclear", "not a medicine", "no medicine"]):
        result["identified_medicine"] = None

    return result


def _image_error(message: str) -> dict:
    return {
        "success": False,
        "identified_medicine": None,
        "common_use": "",
        "safety_notes": "",
        "expiry_reminder": "Always check the expiry date on medicine packaging before use.",
        "confidence": "Low",
        "message": message,
        "raw": message,
    }
