"""MediClarity AI - Streamlit app that explains prescriptions in simple Bangla."""

import hashlib
import html
import io
import os
import json

import httpx
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types, errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"

CONFIDENCE_BADGES = {
    "high": ("উচ্চ নির্ভরযোগ্যতা", "#166534", "#dcfce7"),
    "medium": ("মাঝারি নির্ভরযোগ্যতা", "#92400e", "#fef3c7"),
    "low": ("কম নির্ভরযোগ্যতা", "#991b1b", "#fee2e2"),
}

MEAL_RELATION_LABELS = {
    "before": "খাবারের আগে",
    "after": "খাবারের পরে",
    "with": "খাবারের সাথে",
    "unspecified": "নির্দিষ্ট নয়",
}

st.set_page_config(
    page_title="MediClarity AI",
    page_icon="💊",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main { background-color: #f7f9fb; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        color: #1f2937;
    }
    .card h3, .card p, .card strong {
        color: #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def prepare_image_part(image: Image.Image, max_dimension: int = 2048, quality: int = 85) -> types.Part:
    """Downscale and JPEG-encode the image before sending it to Gemini.

    PIL drops an image's format after .convert("RGB"), so the SDK's own PIL-to-blob
    logic always re-encodes as lossless PNG - on a large photo that can balloon a
    6 MB JPEG into 25+ MB, risking slow uploads and hitting request size limits.
    """
    resized = image.copy()
    resized.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=quality)
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def explain_prescription(client: genai.Client, image: Image.Image) -> list:
    system_prompt = (
        "You are a medical assistant that reads prescription images and explains them to "
        "patients in simple, easy-to-understand Bangla. You must always respond ONLY with "
        "valid JSON, no extra text. The JSON must be a list of objects, one per medicine, "
        "each with these exact keys: \"medicine_name\", \"purpose\", \"dosage\", "
        "\"side_effects\", \"precautions\", \"confidence\", \"timing\". All text values must "
        "be written in simple Bangla, understandable by someone with no medical background. "
        "The \"confidence\" value must be exactly one of \"high\", \"medium\", or \"low\" "
        "(in English, lowercase), reflecting how confident you are that you read this "
        "medicine's name and dosage correctly from the image. Use \"low\" whenever the "
        "handwriting or print is unclear, and in that case make reasonable, clearly-labeled "
        "assumptions in the other fields and mention that the reading may be uncertain. "
        "The \"timing\" value must be an object with exactly these keys: \"morning\" "
        "(boolean, true if this medicine should be taken in the morning), \"afternoon\" "
        "(boolean, for noon/afternoon), \"night\" (boolean, for night), and "
        "\"meal_relation\" (one of \"before\", \"after\", \"with\", or \"unspecified\", in "
        "English lowercase, describing the medicine's relation to meals). Infer these from "
        "the dosage instructions on the prescription; if the timing is not specified, set "
        "all three time booleans to false and \"meal_relation\" to \"unspecified\"."
    )

    user_prompt = (
        "Below is an image of a handwritten/printed prescription. "
        "Identify each medicine and explain it in simple Bangla as instructed."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prepare_image_part(image), user_prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )

    content = response.text.strip()

    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    data = json.loads(content)

    if isinstance(data, dict):
        for key in ("medicines", "result", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    return data


st.title("💊 MediClarity AI")
st.caption("প্রেসক্রিপশনের ছবি আপলোড করুন — সহজ বাংলায় বুঝে নিন আপনার ওষুধ সম্পর্কে")

step_cols = st.columns(3)
STEPS = [
    ("📤", "ছবি আপলোড করুন", "প্রেসক্রিপশনের স্পষ্ট ছবি দিন"),
    ("🤖", "AI বিশ্লেষণ করবে", "Gemini Vision ছবিটি পড়ে বুঝবে"),
    ("📋", "ফলাফল দেখুন", "সহজ বাংলায় বিস্তারিত পাবেন"),
]
for col, (icon, step_title, step_desc) in zip(step_cols, STEPS):
    with col:
        st.markdown(
            f"<div style='text-align:center; font-size:1.75rem;'>{icon}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<p style='text-align:center;'><strong>{step_title}</strong></p>", unsafe_allow_html=True)
        st.caption(step_desc)

st.divider()

gemini_client = get_gemini_client()

if gemini_client is None:
    st.error(
        "GEMINI_API_KEY পাওয়া যায়নি। `.env` ফাইলে আপনার Gemini API key সেট করুন "
        "(দেখুন `.env.example`)।"
    )

uploaded_file = st.file_uploader(
    "প্রেসক্রিপশনের ছবি আপলোড করুন",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("ছবিটি পড়া যায়নি। দয়া করে একটি সঠিক JPG/JPEG/PNG ফাইল আপলোড করুন।")
        st.stop()

    st.image(image, caption="আপলোড করা প্রেসক্রিপশন", use_container_width=True)

    analyze_clicked = st.button(
        "🔍 প্রেসক্রিপশন বিশ্লেষণ করুন",
        disabled=gemini_client is None,
        use_container_width=True,
    )

    already_cached = st.session_state.get("cached_file_hash") == file_hash

    if analyze_clicked and not already_cached:
        with st.spinner("Gemini দিয়ে প্রেসক্রিপশন বিশ্লেষণ করা হচ্ছে..."):
            try:
                medicines = explain_prescription(gemini_client, image)
                st.session_state["cached_file_hash"] = file_hash
                st.session_state["cached_medicines"] = medicines
            except genai_errors.APIError as exc:
                st.error(f"Gemini API-তে সমস্যা হয়েছে: {exc}")
                st.stop()
            except httpx.TransportError:
                st.error(
                    "ইন্টারনেট সংযোগে সমস্যা হয়েছে। দয়া করে আপনার সংযোগ পরীক্ষা করে "
                    "আবার চেষ্টা করুন।"
                )
                st.stop()
            except json.JSONDecodeError:
                st.error(
                    "AI-এর উত্তর সঠিকভাবে বোঝা যায়নি। দয়া করে আবার চেষ্টা করুন।"
                )
                st.stop()
            except Exception as exc:
                st.error(f"একটি অপ্রত্যাশিত সমস্যা হয়েছে: {exc}")
                st.stop()

    if st.session_state.get("cached_file_hash") == file_hash:
        medicines = st.session_state["cached_medicines"]
        valid_medicines = [med for med in medicines if isinstance(med, dict)]

        if not medicines:
            st.warning("কোনো ওষুধ শনাক্ত করা যায়নি।")
        elif not valid_medicines:
            st.error(
                "AI-এর উত্তর প্রত্যাশিত ফরম্যাটে পাওয়া যায়নি। দয়া করে আবার চেষ্টা করুন।"
            )
        else:
            st.success(f"✅ বিশ্লেষণ সম্পন্ন — {len(valid_medicines)} টি ওষুধ শনাক্ত হয়েছে")

            st.divider()
            st.subheader("🕒 ওষুধ সেবনের সময়সূচি")
            timing_rows = []
            for med in valid_medicines:
                timing = med.get("timing")
                if not isinstance(timing, dict):
                    timing = {}
                meal_key = str(timing.get("meal_relation", "unspecified")).strip().lower()
                timing_rows.append(
                    {
                        "ওষুধ": str(med.get("medicine_name", "অজানা ওষুধ")),
                        "সকাল": "✔️" if timing.get("morning") else "—",
                        "দুপুর": "✔️" if timing.get("afternoon") else "—",
                        "রাত": "✔️" if timing.get("night") else "—",
                        "খাবারের সময়": MEAL_RELATION_LABELS.get(
                            meal_key, MEAL_RELATION_LABELS["unspecified"]
                        ),
                    }
                )
            st.table(timing_rows)

            st.divider()
            st.subheader("🩺 বিস্তারিত ব্যাখ্যা")

            for med in valid_medicines:
                name = html.escape(str(med.get("medicine_name", "অজানা ওষুধ")))
                purpose = html.escape(str(med.get("purpose", "তথ্য পাওয়া যায়নি")))
                dosage = html.escape(str(med.get("dosage", "তথ্য পাওয়া যায়নি")))
                side_effects = html.escape(str(med.get("side_effects", "তথ্য পাওয়া যায়নি")))
                precautions = html.escape(str(med.get("precautions", "তথ্য পাওয়া যায়নি")))
                confidence_key = str(med.get("confidence", "low")).strip().lower()
                confidence_label, confidence_fg, confidence_bg = CONFIDENCE_BADGES.get(
                    confidence_key, CONFIDENCE_BADGES["low"]
                )
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>🩺 {name}
                            <span style="background-color:{confidence_bg}; color:{confidence_fg};
                                padding:2px 10px; border-radius:9999px; font-size:0.75rem;
                                font-weight:600; vertical-align:middle;">
                                {confidence_label}
                            </span>
                        </h3>
                        <p><strong>উদ্দেশ্য:</strong> {purpose}</p>
                        <p><strong>সেবনবিধি:</strong> {dosage}</p>
                        <p><strong>সম্ভাব্য পার্শ্বপ্রতিক্রিয়া:</strong> {side_effects}</p>
                        <p><strong>সতর্কতা:</strong> {precautions}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div style="background-color:#fef3c7; border-left:4px solid #d97706;
                border-radius:8px; padding:1rem 1.25rem; margin-top:1.25rem; color:#78350f;">
                ⚠️ <strong>দ্রষ্টব্য:</strong> এই তথ্য শুধুমাত্র সহায়ক উদ্দেশ্যে। চূড়ান্ত
                সিদ্ধান্তের জন্য সবসময় আপনার ডাক্তার বা ফার্মাসিস্টের পরামর্শ নিন।
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("শুরু করতে উপরে একটি প্রেসক্রিপশনের ছবি আপলোড করুন।")
