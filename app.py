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


def as_bool(value) -> bool:
    """Normalize a boolean-ish value from AI JSON output.

    LLMs occasionally emit "true"/"false" as quoted strings despite instructions
    to use JSON booleans - a bare `if value:` check would treat the non-empty
    string "false" as truthy, so string values are compared explicitly.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)

st.set_page_config(
    page_title="MediClarity AI",
    page_icon="💊",
    layout="centered",
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    .stApp, .stApp * {
        font-family: 'Hind Siliguri', sans-serif;
    }
    .main { background-color: #0f172a; }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        border: none;
    }
    .stButton>button:hover { background-color: #2563eb; }
    .hero-banner {
        background: linear-gradient(135deg, #3b82f6, #1e3a8a);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
    }
    .hero-banner .hero-icon { font-size: 2.5rem; }
    .hero-banner h1 {
        color: white;
        margin: 0.25rem 0 0.4rem 0;
        font-size: 1.9rem;
    }
    .hero-banner p {
        color: #dbeafe;
        margin: 0;
        font-size: 1rem;
    }
    .card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.35);
        color: #e2e8f0;
    }
    .card h3, .card p, .card strong {
        color: #e2e8f0;
    }
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        padding: 1rem 0 0.5rem 0;
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


def explain_prescription(client: genai.Client, image: Image.Image) -> dict:
    system_prompt = (
        "You are a medical assistant that reads prescription images and explains them to "
        "patients in simple, easy-to-understand Bangla. You must always respond ONLY with "
        "valid JSON, no extra text. The JSON must be a single object with exactly these "
        "top-level keys: \"overall_summary\", \"overall_confidence\", \"doctor_info\", "
        "\"diagnosis\", \"drug_interactions\", \"medicines\".\n"
        "- \"overall_summary\": a 2-3 sentence plain-Bangla summary of the whole prescription.\n"
        "- \"overall_confidence\": one of \"high\", \"medium\", or \"low\" (English, "
        "lowercase), reflecting your overall confidence in reading the entire image.\n"
        "- \"doctor_info\": the doctor's name/qualification/chamber info in simple Bangla if "
        "visible on the prescription; otherwise exactly \"শনাক্ত করা যায়নি\".\n"
        "- \"diagnosis\": the patient's diagnosis or condition in simple Bangla if mentioned; "
        "otherwise exactly \"উল্লেখ করা হয়নি\".\n"
        "- \"drug_interactions\": a plain-Bangla note about any potentially significant "
        "interaction between the identified medicines; if none are apparent, exactly "
        "\"উল্লেখযোগ্য কোনো ইন্টারঅ্যাকশন শনাক্ত হয়নি\". This is informational only, not a "
        "substitute for a pharmacist's or doctor's advice.\n"
        "- \"medicines\": a list of objects, one per medicine, each with these exact keys: "
        "\"medicine_name\", \"purpose\", \"dosage\", \"side_effects\", \"precautions\", "
        "\"confidence\", \"timing\", \"high_risk\".\n"
        "All text values must be written in simple Bangla, understandable by someone with no "
        "medical background. Each medicine's \"confidence\" value must be exactly one of "
        "\"high\", \"medium\", or \"low\" (English, lowercase), reflecting how confident you "
        "are that you read that medicine's name and dosage correctly. Use \"low\" whenever "
        "the handwriting or print is unclear, and in that case make reasonable, "
        "clearly-labeled assumptions and mention that the reading may be uncertain. "
        "\"high_risk\" must be a boolean - true if the medicine requires special caution, is "
        "a controlled/narrow-safety-margin drug, or commonly causes serious side effects; "
        "otherwise false. \"timing\" must be an object with exactly these keys: \"morning\" "
        "(boolean), \"afternoon\" (boolean), \"night\" (boolean), and \"meal_relation\" (one "
        "of \"before\", \"after\", \"with\", or \"unspecified\", English lowercase). Infer "
        "timing from the dosage instructions; if unspecified, set all three booleans to "
        "false and meal_relation to \"unspecified\"."
    )

    user_prompt = (
        "Below is an image of a handwritten/printed prescription. "
        "Analyze it fully and respond in simple Bangla as instructed."
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

    if isinstance(data, list):
        data = {"medicines": data}
    elif not isinstance(data, dict):
        data = {}

    medicines = data.get("medicines")
    if not isinstance(medicines, list):
        medicines = []

    return {
        "overall_summary": str(data.get("overall_summary") or "").strip(),
        "overall_confidence": str(data.get("overall_confidence", "low")).strip().lower(),
        "doctor_info": str(data.get("doctor_info") or "শনাক্ত করা যায়নি").strip(),
        "diagnosis": str(data.get("diagnosis") or "উল্লেখ করা হয়নি").strip(),
        "drug_interactions": str(
            data.get("drug_interactions") or "উল্লেখযোগ্য কোনো ইন্টারঅ্যাকশন শনাক্ত হয়নি"
        ).strip(),
        "medicines": medicines,
    }


with st.sidebar:
    st.markdown("### ℹ️ MediClarity AI সম্পর্কে")
    st.markdown(
        "প্রেসক্রিপশনের হাতের লেখা বোঝা প্রায়ই কঠিন। **MediClarity AI** একটি ছবি থেকেই "
        "Google Gemini Vision দিয়ে আপনার ওষুধের নাম, উদ্দেশ্য, সেবনবিধি, পার্শ্বপ্রতিক্রিয়া "
        "ও সতর্কতা সহজ বাংলায় বুঝিয়ে দেয়।"
    )
    st.markdown("**🛠️ ব্যবহৃত প্রযুক্তি**")
    st.markdown("- Streamlit\n- Google Gemini 2.5 Flash (Vision)\n- Python")
    st.divider()
    st.caption("⚠️ এটি একটি সহায়ক টুল, চিকিৎসা পরামর্শের বিকল্প নয়।")

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-icon">💊</div>
        <h1>MediClarity AI</h1>
        <p>প্রেসক্রিপশনের ছবি আপলোড করুন — সহজ বাংলায় বুঝে নিন আপনার ওষুধ সম্পর্কে</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        with st.status("🔍 প্রেসক্রিপশন বিশ্লেষণ করা হচ্ছে...", expanded=True) as status:
            st.write("📤 ছবি প্রস্তুত করা হচ্ছে...")
            try:
                st.write("🤖 Gemini Vision দিয়ে পড়া হচ্ছে... এটা কয়েক সেকেন্ড সময় নিতে পারে।")
                result = explain_prescription(gemini_client, image)
                st.session_state["cached_file_hash"] = file_hash
                st.session_state["cached_result"] = result
                status.update(label="✅ বিশ্লেষণ সম্পন্ন হয়েছে!", state="complete", expanded=False)
            except genai_errors.APIError as exc:
                status.update(label="❌ Gemini API-তে সমস্যা হয়েছে", state="error")
                st.error(f"Gemini API-তে সমস্যা হয়েছে: {exc}")
                st.stop()
            except httpx.TransportError:
                status.update(label="❌ ইন্টারনেট সংযোগে সমস্যা", state="error")
                st.error(
                    "ইন্টারনেট সংযোগে সমস্যা হয়েছে। দয়া করে আপনার সংযোগ পরীক্ষা করে "
                    "আবার চেষ্টা করুন।"
                )
                st.stop()
            except json.JSONDecodeError:
                status.update(label="❌ AI-এর উত্তর বোঝা যায়নি", state="error")
                st.error(
                    "AI-এর উত্তর সঠিকভাবে বোঝা যায়নি। দয়া করে আবার চেষ্টা করুন।"
                )
                st.stop()
            except Exception as exc:
                status.update(label="❌ অপ্রত্যাশিত সমস্যা হয়েছে", state="error")
                st.error(f"একটি অপ্রত্যাশিত সমস্যা হয়েছে: {exc}")
                st.stop()

    if st.session_state.get("cached_file_hash") == file_hash:
        result = st.session_state["cached_result"]
        medicines = result.get("medicines", [])
        valid_medicines = [med for med in medicines if isinstance(med, dict)]

        if not medicines:
            st.warning("কোনো ওষুধ শনাক্ত করা যায়নি।")
        elif not valid_medicines:
            st.error(
                "AI-এর উত্তর প্রত্যাশিত ফরম্যাটে পাওয়া যায়নি। দয়া করে আবার চেষ্টা করুন।"
            )
        else:
            st.success(f"✅ বিশ্লেষণ সম্পন্ন — {len(valid_medicines)} টি ওষুধ শনাক্ত হয়েছে")

            overall_confidence_key = str(result.get("overall_confidence", "low")).strip().lower()
            oc_label, oc_fg, oc_bg = CONFIDENCE_BADGES.get(
                overall_confidence_key, CONFIDENCE_BADGES["low"]
            )
            overall_summary = html.escape(
                result.get("overall_summary") or "কোনো সারসংক্ষেপ পাওয়া যায়নি।"
            )
            st.markdown(
                f"""
                <div class="card">
                    <h3>🧠 AI সারসংক্ষেপ
                        <span style="background-color:{oc_bg}; color:{oc_fg};
                            padding:2px 10px; border-radius:9999px; font-size:0.75rem;
                            font-weight:600; vertical-align:middle;">
                            সামগ্রিক {oc_label}
                        </span>
                    </h3>
                    <p>{overall_summary}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            doctor_info = html.escape(result.get("doctor_info") or "শনাক্ত করা যায়নি")
            diagnosis = html.escape(result.get("diagnosis") or "উল্লেখ করা হয়নি")
            info_cols = st.columns(2)
            with info_cols[0]:
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>👨‍⚕️ ডাক্তারের তথ্য</h3>
                        <p>{doctor_info}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with info_cols[1]:
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>🩻 রোগ নির্ণয়</h3>
                        <p>{diagnosis}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            drug_interactions = html.escape(
                result.get("drug_interactions") or "উল্লেখযোগ্য কোনো ইন্টারঅ্যাকশন শনাক্ত হয়নি"
            )
            st.markdown(
                f"""
                <div style="background-color:#450a0a; border-left:4px solid #dc2626;
                    border-radius:8px; padding:1rem 1.25rem; margin-bottom:1rem; color:#fecaca;">
                    💊⚠️ <strong>ড্রাগ ইন্টারঅ্যাকশন:</strong> {drug_interactions}
                </div>
                """,
                unsafe_allow_html=True,
            )

            high_risk_meds = [med for med in valid_medicines if as_bool(med.get("high_risk"))]
            if high_risk_meds:
                risk_names = ", ".join(
                    html.escape(str(m.get("medicine_name", "অজানা ওষুধ"))) for m in high_risk_meds
                )
                st.markdown(
                    f"""
                    <div style="background-color:#450a0a; border-left:4px solid #dc2626;
                        border-radius:8px; padding:1rem 1.25rem; margin-bottom:1rem; color:#fecaca;">
                        🚨 <strong>উচ্চ-ঝুঁকিপূর্ণ ওষুধ:</strong> {risk_names} — এই ওষুধ(গুলো)
                        বিশেষ সতর্কতার সাথে ও ডাক্তারের নির্দেশনা মেনে গ্রহণ করুন।
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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
                        "💊 ওষুধ": str(med.get("medicine_name", "অজানা ওষুধ")),
                        "☀️ সকাল": "✔️" if as_bool(timing.get("morning")) else "—",
                        "🌤️ দুপুর": "✔️" if as_bool(timing.get("afternoon")) else "—",
                        "🌙 রাত": "✔️" if as_bool(timing.get("night")) else "—",
                        "🍽️ খাবারের সময়": MEAL_RELATION_LABELS.get(
                            meal_key, MEAL_RELATION_LABELS["unspecified"]
                        ),
                    }
                )
            st.table(timing_rows)

            st.divider()
            st.subheader("🩺 বিস্তারিত ব্যাখ্যা")

            card_cols = st.columns(2)
            for idx, med in enumerate(valid_medicines):
                name = html.escape(str(med.get("medicine_name", "অজানা ওষুধ")))
                purpose = html.escape(str(med.get("purpose", "তথ্য পাওয়া যায়নি")))
                dosage = html.escape(str(med.get("dosage", "তথ্য পাওয়া যায়নি")))
                side_effects = html.escape(str(med.get("side_effects", "তথ্য পাওয়া যায়নি")))
                precautions = html.escape(str(med.get("precautions", "তথ্য পাওয়া যায়নি")))
                confidence_key = str(med.get("confidence", "low")).strip().lower()
                confidence_label, confidence_fg, confidence_bg = CONFIDENCE_BADGES.get(
                    confidence_key, CONFIDENCE_BADGES["low"]
                )
                high_risk_badge = (
                    """<span style="background-color:#450a0a; color:#fecaca;
                        padding:2px 10px; border-radius:9999px; font-size:0.75rem;
                        font-weight:600; vertical-align:middle;">
                        🚨 উচ্চ ঝুঁকি
                    </span>"""
                    if as_bool(med.get("high_risk"))
                    else ""
                )
                with card_cols[idx % 2]:
                    st.markdown(
                        f"""
                        <div class="card">
                            <h3>🩺 {name}
                                <span style="background-color:{confidence_bg}; color:{confidence_fg};
                                    padding:2px 10px; border-radius:9999px; font-size:0.75rem;
                                    font-weight:600; vertical-align:middle;">
                                    {confidence_label}
                                </span>
                                {high_risk_badge}
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
            <div style="background-color:#451a03; border-left:4px solid #d97706;
                border-radius:8px; padding:1rem 1.25rem; margin-top:1.25rem; color:#fde68a;">
                ⚠️ <strong>দ্রষ্টব্য:</strong> এই তথ্য শুধুমাত্র সহায়ক উদ্দেশ্যে। চূড়ান্ত
                সিদ্ধান্তের জন্য সবসময় আপনার ডাক্তার বা ফার্মাসিস্টের পরামর্শ নিন।
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("শুরু করতে উপরে একটি প্রেসক্রিপশনের ছবি আপলোড করুন।")

st.markdown(
    """
    <div class="app-footer" style="text-align:center; padding:1rem; color:#6b7280; font-size:0.85rem;">
        MediClarity AI — Hackathon Project
    </div>
    """,
    unsafe_allow_html=True,
)