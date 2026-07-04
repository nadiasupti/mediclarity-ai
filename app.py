"""MediClarity AI - Streamlit app that explains prescriptions in simple Bangla."""

import os
import json

import streamlit as st
from PIL import Image
import pytesseract
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()

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
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def extract_text_from_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image).strip()


def explain_prescription(client: OpenAI, ocr_text: str) -> dict:
    system_prompt = (
        "You are a medical assistant that explains prescriptions to patients in simple, "
        "easy-to-understand Bangla. You must always respond ONLY with valid JSON, no extra text. "
        "The JSON must be a list of objects, one per medicine, each with these exact keys: "
        "\"medicine_name\", \"purpose\", \"dosage\", \"side_effects\", \"precautions\". "
        "All values must be written in simple Bangla, understandable by someone with no medical "
        "background. If the OCR text is unclear or incomplete, make reasonable, clearly-labeled "
        "assumptions and mention that the reading may be uncertain in the relevant field."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Below is the OCR-extracted text from a handwritten/printed prescription. "
                    "Identify each medicine and explain it in simple Bangla as instructed.\n\n"
                    f"OCR TEXT:\n{ocr_text}"
                ),
            },
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content.strip()

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

client = get_openai_client()
if client is None:
    st.error(
        "OPENAI_API_KEY পাওয়া যায়নি। `.env` ফাইলে আপনার OpenAI API key সেট করুন "
        "(দেখুন `.env.example`)।"
    )

uploaded_file = st.file_uploader(
    "প্রেসক্রিপশনের ছবি আপলোড করুন",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("ছবিটি পড়া যায়নি। দয়া করে একটি সঠিক JPG/JPEG/PNG ফাইল আপলোড করুন।")
        st.stop()

    st.image(image, caption="আপলোড করা প্রেসক্রিপশন", use_container_width=True)

    analyze_clicked = st.button("প্রেসক্রিপশন বিশ্লেষণ করুন", disabled=client is None)

    if analyze_clicked:
        with st.spinner("ছবি থেকে লেখা পড়া হচ্ছে..."):
            try:
                ocr_text = extract_text_from_image(image)
            except pytesseract.TesseractNotFoundError:
                st.error(
                    "Tesseract OCR engine পাওয়া যায়নি। এটি ইনস্টল করুন এবং সিস্টেম PATH-এ যোগ করুন। "
                    "(README.md দেখুন বিস্তারিত নির্দেশনার জন্য।)"
                )
                st.stop()
            except Exception as exc:
                st.error(f"ছবি থেকে লেখা বের করতে সমস্যা হয়েছে: {exc}")
                st.stop()

        if not ocr_text:
            st.warning(
                "ছবি থেকে কোনো লেখা শনাক্ত করা যায়নি। দয়া করে আরও স্পষ্ট ছবি আপলোড করুন।"
            )
            st.stop()

        with st.expander("OCR থেকে প্রাপ্ত মূল লেখা দেখুন"):
            st.text(ocr_text)

        with st.spinner("GPT-4o দিয়ে প্রেসক্রিপশন বিশ্লেষণ করা হচ্ছে..."):
            try:
                medicines = explain_prescription(client, ocr_text)
            except OpenAIError as exc:
                st.error(f"OpenAI API-তে সমস্যা হয়েছে: {exc}")
                st.stop()
            except json.JSONDecodeError:
                st.error(
                    "AI-এর উত্তর সঠিকভাবে বোঝা যায়নি। দয়া করে আবার চেষ্টা করুন।"
                )
                st.stop()
            except Exception as exc:
                st.error(f"একটি অপ্রত্যাশিত সমস্যা হয়েছে: {exc}")
                st.stop()

        if not medicines:
            st.warning("কোনো ওষুধ শনাক্ত করা যায়নি।")
        else:
            st.success(f"{len(medicines)} টি ওষুধ শনাক্ত হয়েছে")
            for med in medicines:
                name = med.get("medicine_name", "অজানা ওষুধ")
                with st.container():
                    st.markdown(f'<div class="card">', unsafe_allow_html=True)
                    st.subheader(f"🩺 {name}")
                    st.markdown(f"**উদ্দেশ্য:** {med.get('purpose', 'তথ্য পাওয়া যায়নি')}")
                    st.markdown(f"**সেবনবিধি:** {med.get('dosage', 'তথ্য পাওয়া যায়নি')}")
                    st.markdown(
                        f"**সম্ভাব্য পার্শ্বপ্রতিক্রিয়া:** "
                        f"{med.get('side_effects', 'তথ্য পাওয়া যায়নি')}"
                    )
                    st.markdown(
                        f"**সতর্কতা:** {med.get('precautions', 'তথ্য পাওয়া যায়নি')}"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

        st.info(
            "⚠️ এই তথ্য শুধুমাত্র সহায়ক উদ্দেশ্যে। চূড়ান্ত সিদ্ধান্তের জন্য সবসময় আপনার "
            "ডাক্তার বা ফার্মাসিস্টের পরামর্শ নিন।"
        )
else:
    st.info("শুরু করতে উপরে একটি প্রেসক্রিপশনের ছবি আপলোড করুন।")
