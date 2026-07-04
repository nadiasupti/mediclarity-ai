"""MediClarity AI - Streamlit app that explains prescriptions in simple Bangla."""

import hashlib
import html
import io
import os
import json
import re

import httpx
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from gtts import gTTS
from PIL import Image
from google import genai
from google.genai import types, errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

BENGALI_FONT_PATH = os.path.join(os.path.dirname(__file__), "assets", "fonts", "NotoSansBengali-Regular.ttf")

GEMINI_MODEL = "gemini-2.5-flash"

CONFIDENCE_LABELS = {
    "বাংলা": {"high": "উচ্চ নির্ভরযোগ্যতা", "medium": "মাঝারি নির্ভরযোগ্যতা", "low": "কম নির্ভরযোগ্যতা"},
    "English": {"high": "High Reliability", "medium": "Medium Reliability", "low": "Low Reliability"},
}
CONFIDENCE_EMOJI_DEFAULT = {"high": "🟢", "medium": "🟡", "low": "🔴"}
# Blue/orange/purple instead of red/green/yellow - distinguishable regardless
# of red-green color blindness (the most common form).
CONFIDENCE_EMOJI_COLORBLIND = {"high": "🔵", "medium": "🟠", "low": "🟣"}
CONFIDENCE_COLORS_DEFAULT = {
    "high": ("#166534", "#dcfce7"),
    "medium": ("#92400e", "#fef3c7"),
    "low": ("#991b1b", "#fee2e2"),
}
CONFIDENCE_COLORS_COLORBLIND = {
    "high": ("#1e3a8a", "#dbeafe"),
    "medium": ("#9a3412", "#fed7aa"),
    "low": ("#6b21a8", "#f3e8ff"),
}


def get_confidence_badges(language: str, colorblind_mode: bool) -> dict:
    labels = CONFIDENCE_LABELS[language]
    emojis = CONFIDENCE_EMOJI_COLORBLIND if colorblind_mode else CONFIDENCE_EMOJI_DEFAULT
    colors = CONFIDENCE_COLORS_COLORBLIND if colorblind_mode else CONFIDENCE_COLORS_DEFAULT
    return {
        key: (f"{emojis[key]} {labels[key]}", colors[key][0], colors[key][1])
        for key in ("high", "medium", "low")
    }


MEAL_RELATION_TEXT = {
    "বাংলা": {
        "before": "খাবারের আগে", "after": "খাবারের পরে",
        "with": "খাবারের সাথে", "unspecified": "নির্দিষ্ট নয়",
    },
    "English": {
        "before": "Before meal", "after": "After meal",
        "with": "With meal", "unspecified": "Not specified",
    },
}
MEAL_RELATION_EMOJI = {"before": "🍽️", "after": "🥣", "with": "🍛", "unspecified": "❔"}


def get_meal_label(language: str, meal_key: str) -> str:
    lang_map = MEAL_RELATION_TEXT[language]
    key = meal_key if meal_key in lang_map else "unspecified"
    return f"{MEAL_RELATION_EMOJI[key]} {lang_map[key]}"


# Danger/alert colors (drug interactions, high-risk medicine). The colorblind
# variant swaps red for dark-indigo/amber, since red-on-dark can be hard to
# distinguish from other saturated colors used elsewhere in the UI.
DANGER_COLORS_DEFAULT = {"bg": "#450a0a", "border": "#dc2626", "text": "#fecaca"}
DANGER_COLORS_COLORBLIND = {"bg": "#1e1b4b", "border": "#f59e0b", "text": "#fde68a"}

TRANSLATIONS = {
    "বাংলা": {
        "hero_subtitle": "প্রেসক্রিপশনের ছবি আপলোড করুন — সহজ বাংলায় বুঝে নিন আপনার ওষুধ সম্পর্কে",
        "step1_title": "ছবি আপলোড করুন", "step1_desc": "প্রেসক্রিপশনের স্পষ্ট ছবি দিন",
        "step2_title": "AI বিশ্লেষণ করবে", "step2_desc": "Gemini Vision ছবিটি পড়ে বুঝবে",
        "step3_title": "ফলাফল দেখুন", "step3_desc": "সহজ বাংলায় বিস্তারিত পাবেন",
        "upload_label": "প্রেসক্রিপশনের ছবি আপলোড করুন",
        "upload_caption": "আপলোড করা প্রেসক্রিপশন",
        "analyze_button": "🔍 প্রেসক্রিপশন বিশ্লেষণ করুন",
        "api_key_missing": (
            "GEMINI_API_KEY পাওয়া যায়নি। `.env` ফাইলে আপনার Gemini API key সেট করুন "
            "(দেখুন `.env.example`)।"
        ),
        "image_read_error": "ছবিটি পড়া যায়নি। দয়া করে একটি সঠিক JPG/JPEG/PNG ফাইল আপলোড করুন।",
        "status_analyzing": "🔍 প্রেসক্রিপশন বিশ্লেষণ করা হচ্ছে...",
        "status_preparing": "📤 ছবি প্রস্তুত করা হচ্ছে...",
        "status_reading": "🤖 Gemini Vision দিয়ে পড়া হচ্ছে... এটা কয়েক সেকেন্ড সময় নিতে পারে।",
        "status_done": "✅ বিশ্লেষণ সম্পন্ন হয়েছে!",
        "status_api_error": "❌ Gemini API-তে সমস্যা হয়েছে",
        "status_network_error": "❌ ইন্টারনেট সংযোগে সমস্যা",
        "status_parse_error": "❌ AI-এর উত্তর বোঝা যায়নি",
        "status_unexpected_error": "❌ অপ্রত্যাশিত সমস্যা হয়েছে",
        "error_api_prefix": "Gemini API-তে সমস্যা হয়েছে: ",
        "error_network": (
            "ইন্টারনেট সংযোগে সমস্যা হয়েছে। দয়া করে আপনার সংযোগ পরীক্ষা করে আবার চেষ্টা করুন।"
        ),
        "error_parse": "AI-এর উত্তর সঠিকভাবে বোঝা যায়নি। দয়া করে আবার চেষ্টা করুন।",
        "error_unexpected_prefix": "একটি অপ্রত্যাশিত সমস্যা হয়েছে: ",
        "no_medicines": "কোনো ওষুধ শনাক্ত করা যায়নি।",
        "bad_format": "AI-এর উত্তর প্রত্যাশিত ফরম্যাটে পাওয়া যায়নি। দয়া করে আবার চেষ্টা করুন।",
        "success_prefix": "✅ বিশ্লেষণ সম্পন্ন — ",
        "success_suffix": " টি ওষুধ শনাক্ত হয়েছে",
        "language_switch_notice": (
            "ℹ️ ভাষা পরিবর্তন করেছেন। নতুন ভাষায় ফলাফল পেতে আবার বিশ্লেষণ বাটনে ক্লিক করুন। "
            "নিচে আগের ফলাফল দেখানো হচ্ছে।"
        ),
        "summary_title": "📋 প্রেসক্রিপশনের সারসংক্ষেপ",
        "summary_overall": "সামগ্রিক",
        "patient_name_label": "রোগীর নাম",
        "patient_age_label": "বয়স",
        "no_summary": "কোনো সারসংক্ষেপ পাওয়া যায়নি।",
        "probable_condition_label": "রোগীর সম্ভাব্য সমস্যা",
        "probable_condition_default": "নির্ধারণ করা যায়নি",
        "treatment_purpose_label": "চিকিৎসার উদ্দেশ্য",
        "treatment_purpose_default": "উল্লেখ করা যায়নি",
        "total_medicines_label": "মোট ওষুধ",
        "total_medicines_suffix": "টি",
        "not_identified": "শনাক্ত করা যায়নি",
        "not_mentioned": "উল্লেখ নেই",
        "doctor_title": "👨‍⚕️ ডাক্তারের তথ্য",
        "doctor_name_label": "নাম",
        "doctor_degree_label": "ডিগ্রি",
        "doctor_hospital_label": "হাসপাতাল/চেম্বার",
        "doctor_phone_label": "ফোন",
        "date_title": "🗓️ প্রেসক্রিপশনের তারিখ",
        "date_label": "তারিখ",
        "followup_label": "ফলো-আপ তারিখ",
        "no_interaction": "উল্লেখযোগ্য কোনো ইন্টারঅ্যাকশন শনাক্ত হয়নি",
        "interaction_label": "ড্রাগ ইন্টারঅ্যাকশন",
        "high_risk_label": "উচ্চ-ঝুঁকিপূর্ণ ওষুধ",
        "high_risk_warning": "এই ওষুধ(গুলো) বিশেষ সতর্কতার সাথে ও ডাক্তারের নির্দেশনা মেনে গ্রহণ করুন।",
        "high_risk_badge": "🚨 উচ্চ ঝুঁকি",
        "schedule_title": "📅 দৈনিক ওষুধ সময়সূচি",
        "medicine_col": "ওষুধ",
        "morning_col": "☀️ সকাল",
        "afternoon_col": "🌤️ দুপুর",
        "night_col": "🌙 রাত",
        "meal_col": "🍽️ খাবারের সময়",
        "details_title": "🩺 বিস্তারিত ব্যাখ্যা",
        "unknown_medicine": "অজানা ওষুধ",
        "no_info": "তথ্য পাওয়া যায়নি",
        "purpose_label": "উদ্দেশ্য",
        "dosage_label": "সেবনবিধি",
        "duration_label": "মেয়াদ",
        "side_effects_label": "সম্ভাব্য পার্শ্বপ্রতিক্রিয়া",
        "precautions_label": "সতর্কতা",
        "disclaimer": (
            "⚠️ দ্রষ্টব্য: এই তথ্য শুধুমাত্র সহায়ক উদ্দেশ্যে। চূড়ান্ত সিদ্ধান্তের জন্য "
            "সবসময় আপনার ডাক্তার বা ফার্মাসিস্টের পরামর্শ নিন।"
        ),
        "empty_state": "শুরু করতে উপরে একটি প্রেসক্রিপশনের ছবি আপলোড করুন।",
        "listen_button": "🔊 শুনুন",
        "tts_lang": "bn",
        "content_language": "Bangla",
        "tts_error": "ভয়েস তৈরি করা যায়নি। দয়া করে আপনার ইন্টারনেট সংযোগ পরীক্ষা করে আবার চেষ্টা করুন।",
        "report_title": "MediClarity AI - প্রেসক্রিপশন রিপোর্ট",
        "download_pdf_button": "⬇️ PDF ডাউনলোড",
        "download_txt_button": "⬇️ TXT ডাউনলোড",
        "print_button": "🖨️ প্রিন্ট করুন",
        "copy_button": "📋 কপি করুন",
        "copy_success": "✅ কপি হয়েছে!",
        "copy_failed": "❌ কপি ব্যর্থ হয়েছে",
        "medicines_section_title": "ওষুধের বিস্তারিত",
        "confidence_col_label": "নির্ভরযোগ্যতা",
        "high_risk_col_label": "উচ্চ ঝুঁকি",
        "yes_label": "হ্যাঁ",
        "no_label": "না",
        "timing_label": "সময়",
        "generated_by_label": "MediClarity AI দিয়ে তৈরি — শুধুমাত্র সহায়ক উদ্দেশ্যে, চিকিৎসা পরামর্শের বিকল্প নয়।",
        "chat_title": "💬 প্রেসক্রিপশন নিয়ে জিজ্ঞাসা করুন",
        "chat_intro": "আপনার ওষুধ, ডোজ, সময়, বা পার্শ্বপ্রতিক্রিয়া নিয়ে যেকোনো প্রশ্ন জিজ্ঞাসা করুন।",
        "chat_placeholder": "আপনার প্রশ্ন লিখুন...",
        "chat_suggestion_1": "এই ওষুধটা কেন দিয়েছে?",
        "chat_suggestion_2": "খালি পেটে খাব?",
        "chat_suggestion_3": "রাতে খাব?",
        "chat_thinking": "চিন্তা করছে...",
        "chat_error": "উত্তর দিতে সমস্যা হয়েছে। দয়া করে আবার চেষ্টা করুন।",
    },
    "English": {
        "hero_subtitle": "Upload a prescription image — understand your medicines in plain English",
        "step1_title": "Upload Image", "step1_desc": "Provide a clear prescription photo",
        "step2_title": "AI Analyzes It", "step2_desc": "Gemini Vision reads and understands it",
        "step3_title": "View Results", "step3_desc": "Get plain-English details",
        "upload_label": "Upload a prescription image",
        "upload_caption": "Uploaded prescription",
        "analyze_button": "🔍 Analyze Prescription",
        "api_key_missing": (
            "GEMINI_API_KEY not found. Set your Gemini API key in the `.env` file "
            "(see `.env.example`)."
        ),
        "image_read_error": "Could not read the image. Please upload a valid JPG/JPEG/PNG file.",
        "status_analyzing": "🔍 Analyzing prescription...",
        "status_preparing": "📤 Preparing the image...",
        "status_reading": "🤖 Reading with Gemini Vision... this may take a few seconds.",
        "status_done": "✅ Analysis complete!",
        "status_api_error": "❌ Gemini API error",
        "status_network_error": "❌ Network connection issue",
        "status_parse_error": "❌ Could not parse AI response",
        "status_unexpected_error": "❌ Unexpected error",
        "error_api_prefix": "Gemini API error: ",
        "error_network": "A network error occurred. Please check your connection and try again.",
        "error_parse": "Could not properly understand the AI's response. Please try again.",
        "error_unexpected_prefix": "An unexpected error occurred: ",
        "no_medicines": "No medicines were identified.",
        "bad_format": "The AI's response wasn't in the expected format. Please try again.",
        "success_prefix": "✅ Analysis complete — ",
        "success_suffix": " medicine(s) identified",
        "language_switch_notice": (
            "ℹ️ You changed the language. Click the analyze button again to see the result in "
            "the new language. Showing the previous result below."
        ),
        "summary_title": "📋 Prescription Summary",
        "summary_overall": "Overall",
        "patient_name_label": "Patient Name",
        "patient_age_label": "Age",
        "no_summary": "No summary available.",
        "probable_condition_label": "Probable Condition",
        "probable_condition_default": "Could not be determined",
        "treatment_purpose_label": "Treatment Purpose",
        "treatment_purpose_default": "Not mentioned",
        "total_medicines_label": "Total Medicines",
        "total_medicines_suffix": "",
        "not_identified": "Not identified",
        "not_mentioned": "Not mentioned",
        "doctor_title": "👨‍⚕️ Doctor Information",
        "doctor_name_label": "Name",
        "doctor_degree_label": "Degree",
        "doctor_hospital_label": "Hospital/Chamber",
        "doctor_phone_label": "Phone",
        "date_title": "🗓️ Prescription Date",
        "date_label": "Date",
        "followup_label": "Follow-up Date",
        "no_interaction": "No significant interaction detected",
        "interaction_label": "Drug Interaction",
        "high_risk_label": "High-Risk Medicine",
        "high_risk_warning": "Take this medicine(s) with special caution and follow your doctor's guidance.",
        "high_risk_badge": "🚨 High Risk",
        "schedule_title": "📅 Daily Medicine Schedule",
        "medicine_col": "Medicine",
        "morning_col": "☀️ Morning",
        "afternoon_col": "🌤️ Afternoon",
        "night_col": "🌙 Night",
        "meal_col": "🍽️ Meal Timing",
        "details_title": "🩺 Detailed Explanation",
        "unknown_medicine": "Unknown medicine",
        "no_info": "Information not available",
        "purpose_label": "Purpose",
        "dosage_label": "Dosage",
        "duration_label": "Duration",
        "side_effects_label": "Possible Side Effects",
        "precautions_label": "Precautions",
        "disclaimer": (
            "⚠️ Note: This information is for supportive purposes only. Always consult your "
            "doctor or pharmacist before making final decisions."
        ),
        "empty_state": "Upload a prescription image above to get started.",
        "listen_button": "🔊 Listen",
        "tts_lang": "en",
        "content_language": "English",
        "tts_error": "Could not generate voice audio. Please check your internet connection and try again.",
        "report_title": "MediClarity AI - Prescription Report",
        "download_pdf_button": "⬇️ Download PDF",
        "download_txt_button": "⬇️ Download TXT",
        "print_button": "🖨️ Print",
        "copy_button": "📋 Copy",
        "copy_success": "✅ Copied!",
        "copy_failed": "❌ Copy failed",
        "medicines_section_title": "Medicine Details",
        "confidence_col_label": "Reliability",
        "high_risk_col_label": "High Risk",
        "yes_label": "Yes",
        "no_label": "No",
        "timing_label": "Timing",
        "generated_by_label": "Generated by MediClarity AI - for supportive purposes only, not a substitute for medical advice.",
        "chat_title": "💬 Chat with Your Prescription",
        "chat_intro": "Ask anything about your medicines, dosage, timing, or side effects.",
        "chat_placeholder": "Type your question...",
        "chat_suggestion_1": "Why was this medicine given?",
        "chat_suggestion_2": "Should I take it on an empty stomach?",
        "chat_suggestion_3": "Should I take it at night?",
        "chat_thinking": "Thinking...",
        "chat_error": "Could not get an answer. Please try again.",
    },
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


def render_html(html_str: str) -> None:
    """Render a multi-line HTML string via st.markdown, collapsed onto one line.

    Streamlit's markdown parser treats 4+ leading spaces as an indented code
    block, and a line that's empty after an interpolated value (e.g. an unset
    high_risk badge) reads as a blank line that splits the block mid-way. Both
    are avoided entirely by joining stripped, non-empty lines with a space so
    the parser only ever sees a single line of raw HTML.
    """
    collapsed = " ".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(collapsed, unsafe_allow_html=True)


def speak_button(text: str, label: str, tts_lang: str, error_text: str, key: str) -> None:
    """Render a button that generates and plays speech audio for `text`.

    Uses gTTS (Google Translate's TTS endpoint) to synthesize an MP3 server-side
    and play it with st.audio, instead of the browser's built-in SpeechSynthesis
    API. Browser TTS depends on voices installed at the OS level, and most
    Windows/Android installs don't ship a Bangla voice at all - speechSynthesis
    would then silently substitute a default (English) voice. Generating the
    audio server-side works the same for every listener regardless of what
    voices their device happens to have.
    """
    if st.button(label, key=key):
        with st.spinner("..."):
            try:
                buffer = io.BytesIO()
                gTTS(text=text, lang=tts_lang).write_to_fp(buffer)
                buffer.seek(0)
                st.audio(buffer, format="audio/mp3", autoplay=True)
            except Exception:
                st.error(error_text)


def print_button(label: str) -> None:
    """Render a button that opens the browser's native print dialog for the whole page."""
    components.html(
        f"""
        <html>
        <body style="margin:0; padding:0; background:transparent;">
        <button id="printBtn" style="background-color:#3b82f6; color:white; border:none;
            border-radius:8px; padding:0.6rem 1.5rem; font-size:1rem; cursor:pointer;
            font-family:sans-serif; width:100%;">
            {label}
        </button>
        <script>
        document.getElementById("printBtn").addEventListener("click", function () {{
            window.parent.print();
        }});
        </script>
        </body>
        </html>
        """,
        height=60,
        scrolling=False,
    )


def copy_button(text: str, label: str, success_text: str, failed_text: str, dom_id: str) -> None:
    """Render a button that copies `text` to the clipboard.

    Uses the older document.execCommand("copy") via a hidden textarea instead of
    navigator.clipboard.writeText, because the Clipboard API can be blocked by
    permissions policy inside the iframe that st.components.v1.html renders
    into, while execCommand works without any special permission grant.
    """
    safe_text = json.dumps(text)
    safe_success = json.dumps(success_text)
    safe_failed = json.dumps(failed_text)
    components.html(
        f"""
        <html>
        <body style="margin:0; padding:0; background:transparent; font-family:sans-serif;">
        <button id="copyBtn-{dom_id}" style="background-color:#3b82f6; color:white; border:none;
            border-radius:8px; padding:0.6rem 1.5rem; font-size:1rem; cursor:pointer;
            font-family:sans-serif; width:100%;">
            {label}
        </button>
        <div id="copyStatus-{dom_id}" style="font-size:0.75rem; color:#94a3b8;
            margin-top:2px; text-align:center;"></div>
        <script>
        document.getElementById("copyBtn-{dom_id}").addEventListener("click", function () {{
            var textarea = document.createElement("textarea");
            textarea.value = {safe_text};
            textarea.style.position = "fixed";
            textarea.style.opacity = "0";
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();
            var statusEl = document.getElementById("copyStatus-{dom_id}");
            try {{
                document.execCommand("copy");
                statusEl.textContent = {safe_success};
            }} catch (e) {{
                statusEl.textContent = {safe_failed};
            }}
            document.body.removeChild(textarea);
        }});
        </script>
        </body>
        </html>
        """,
        height=80,
        scrolling=False,
    )


def build_text_report(result: dict, T: dict, valid_medicines: list, language: str) -> str:
    """Build a plain-text version of the full analysis, for TXT download and copy."""
    patient_info = result.get("patient_info", {})
    if not isinstance(patient_info, dict):
        patient_info = {}
    doctor_info = result.get("doctor_info", {})
    if not isinstance(doctor_info, dict):
        doctor_info = {}

    lines = [T["report_title"], "=" * len(T["report_title"]), ""]
    lines.append(f"{T['patient_name_label']}: {patient_info.get('name') or T['not_identified']}")
    lines.append(f"{T['patient_age_label']}: {patient_info.get('age') or T['not_identified']}")
    lines.append("")
    lines.append(result.get("overall_summary") or T["no_summary"])
    lines.append("")
    lines.append(
        f"{T['probable_condition_label']}: "
        f"{result.get('probable_condition') or T['probable_condition_default']}"
    )
    lines.append(
        f"{T['treatment_purpose_label']}: "
        f"{result.get('treatment_purpose') or T['treatment_purpose_default']}"
    )
    lines.append(f"{T['total_medicines_label']}: {len(valid_medicines)} {T['total_medicines_suffix']}")
    lines.append("")

    lines.append(T["doctor_title"])
    lines.append(f"{T['doctor_name_label']}: {doctor_info.get('name') or T['not_identified']}")
    lines.append(f"{T['doctor_degree_label']}: {doctor_info.get('degree') or T['not_identified']}")
    lines.append(f"{T['doctor_hospital_label']}: {doctor_info.get('hospital') or T['not_identified']}")
    lines.append(f"{T['doctor_phone_label']}: {doctor_info.get('phone') or T['not_mentioned']}")
    lines.append("")

    lines.append(T["date_title"])
    lines.append(f"{T['date_label']}: {result.get('prescription_date') or T['not_mentioned']}")
    lines.append(f"{T['followup_label']}: {result.get('follow_up_date') or T['not_mentioned']}")
    lines.append("")

    lines.append(f"{T['interaction_label']}: {result.get('drug_interactions') or T['no_interaction']}")
    lines.append("")

    lines.append(T["medicines_section_title"])
    lines.append("-" * len(T["medicines_section_title"]))
    for idx, med in enumerate(valid_medicines, 1):
        name = med.get("medicine_name") or T["unknown_medicine"]
        lines.append(f"{idx}. {name}")
        lines.append(f"   {T['purpose_label']}: {med.get('purpose') or T['no_info']}")
        lines.append(f"   {T['dosage_label']}: {med.get('dosage') or T['no_info']}")
        lines.append(f"   {T['duration_label']}: {med.get('duration') or T['not_mentioned']}")
        lines.append(f"   {T['side_effects_label']}: {med.get('side_effects') or T['no_info']}")
        lines.append(f"   {T['precautions_label']}: {med.get('precautions') or T['no_info']}")
        confidence_key = str(med.get("confidence", "low")).strip().lower()
        lines.append(f"   {T['confidence_col_label']}: {confidence_key}")
        high_risk_text = T["yes_label"] if as_bool(med.get("high_risk")) else T["no_label"]
        lines.append(f"   {T['high_risk_col_label']}: {high_risk_text}")

        timing = med.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        times = [
            label for flag_key, label in (
                ("morning", T["morning_col"]),
                ("afternoon", T["afternoon_col"]),
                ("night", T["night_col"]),
            ) if as_bool(timing.get(flag_key))
        ]
        meal_key = str(timing.get("meal_relation", "unspecified")).strip().lower()
        meal_text = get_meal_label(language, meal_key)
        lines.append(f"   {T['timing_label']}: {', '.join(times) if times else '-'} ({meal_text})")
        lines.append("")

    lines.append(T["disclaimer"])
    lines.append("")
    lines.append(T["generated_by_label"])
    return "\n".join(lines)


EMOJI_PATTERN = re.compile(
    "[\U0001F1E6-\U0001FAFF\U00002600-\U000027BF️‍]+", flags=re.UNICODE
)


def strip_emoji(text: str) -> str:
    """Remove emoji from text before writing it into the PDF.

    The bundled Noto Sans Bengali font covers Bangla and Latin text but has no
    emoji glyphs, so headers like "📋 প্রেসক্রিপশনের সারসংক্ষেপ" would otherwise
    silently drop the emoji mid-render (fpdf2 logs a "missing glyph" warning
    and skips it) - stripping it upfront keeps the PDF looking intentional.
    """
    return EMOJI_PATTERN.sub("", text).strip()


def build_pdf_report(result: dict, T: dict, valid_medicines: list, language: str) -> bytes:
    """Build a PDF version of the full analysis using a Bangla-capable font.

    fpdf2's built-in core fonts (Helvetica etc.) only cover Latin text, so
    Bangla content would render as empty boxes - a bundled Noto Sans Bengali
    TTF is embedded instead, which covers both Bangla and Latin glyphs.
    """
    patient_info = result.get("patient_info", {})
    if not isinstance(patient_info, dict):
        patient_info = {}
    doctor_info = result.get("doctor_info", {})
    if not isinstance(doctor_info, dict):
        doctor_info = {}

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Bengali", "", BENGALI_FONT_PATH)
    # Bangla is a complex script: vowel signs (matras) can visually precede
    # their consonant and conjuncts must render as fused ligatures. Without
    # HarfBuzz-based shaping, fpdf2 places glyphs in raw character order,
    # which reads as garbled/wrongly-ordered Bangla even though every
    # individual glyph exists in the font.
    pdf.set_text_shaping(use_shaping_engine=True)

    def heading(text: str, size: int = 13) -> None:
        pdf.ln(3)
        pdf.set_font("Bengali", size=size)
        pdf.multi_cell(0, 8, strip_emoji(text), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Bengali", size=11)

    def line(text: str) -> None:
        pdf.multi_cell(0, 7, strip_emoji(text), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Bengali", size=16)
    pdf.multi_cell(0, 10, strip_emoji(T["report_title"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Bengali", size=11)

    line(f"{T['patient_name_label']}: {patient_info.get('name') or T['not_identified']}")
    line(f"{T['patient_age_label']}: {patient_info.get('age') or T['not_identified']}")

    heading(T["summary_title"])
    line(result.get("overall_summary") or T["no_summary"])
    line(
        f"{T['probable_condition_label']}: "
        f"{result.get('probable_condition') or T['probable_condition_default']}"
    )
    line(
        f"{T['treatment_purpose_label']}: "
        f"{result.get('treatment_purpose') or T['treatment_purpose_default']}"
    )
    line(f"{T['total_medicines_label']}: {len(valid_medicines)} {T['total_medicines_suffix']}")

    heading(T["doctor_title"])
    line(f"{T['doctor_name_label']}: {doctor_info.get('name') or T['not_identified']}")
    line(f"{T['doctor_degree_label']}: {doctor_info.get('degree') or T['not_identified']}")
    line(f"{T['doctor_hospital_label']}: {doctor_info.get('hospital') or T['not_identified']}")
    line(f"{T['doctor_phone_label']}: {doctor_info.get('phone') or T['not_mentioned']}")

    heading(T["date_title"])
    line(f"{T['date_label']}: {result.get('prescription_date') or T['not_mentioned']}")
    line(f"{T['followup_label']}: {result.get('follow_up_date') or T['not_mentioned']}")

    heading(T["interaction_label"])
    line(result.get("drug_interactions") or T["no_interaction"])

    heading(T["medicines_section_title"])
    for idx, med in enumerate(valid_medicines, 1):
        name = med.get("medicine_name") or T["unknown_medicine"]
        pdf.set_font("Bengali", size=12)
        pdf.multi_cell(0, 7, strip_emoji(f"{idx}. {name}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Bengali", size=11)
        line(f"{T['purpose_label']}: {med.get('purpose') or T['no_info']}")
        line(f"{T['dosage_label']}: {med.get('dosage') or T['no_info']}")
        line(f"{T['duration_label']}: {med.get('duration') or T['not_mentioned']}")
        line(f"{T['side_effects_label']}: {med.get('side_effects') or T['no_info']}")
        line(f"{T['precautions_label']}: {med.get('precautions') or T['no_info']}")
        high_risk_text = T["yes_label"] if as_bool(med.get("high_risk")) else T["no_label"]
        line(f"{T['high_risk_col_label']}: {high_risk_text}")

        timing = med.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        times = [
            slot_label for flag_key, slot_label in (
                ("morning", T["morning_col"]),
                ("afternoon", T["afternoon_col"]),
                ("night", T["night_col"]),
            ) if as_bool(timing.get(flag_key))
        ]
        meal_key = str(timing.get("meal_relation", "unspecified")).strip().lower()
        meal_text = get_meal_label(language, meal_key)
        line(f"{T['timing_label']}: {', '.join(times) if times else '-'} ({meal_text})")
        pdf.ln(2)

    heading(T["disclaimer"], size=10)
    pdf.set_font("Bengali", size=8)
    line(T["generated_by_label"])

    return bytes(pdf.output())


st.set_page_config(
    page_title="MediClarity AI",
    page_icon="💊",
    layout="centered",
)

render_html(
    """
    <link href="https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    .stApp {
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
    """
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


def explain_prescription(client: genai.Client, image: Image.Image, content_language: str = "Bangla") -> dict:
    system_prompt = (
        "You are a medical assistant that reads prescription images and explains them to "
        f"patients in simple, easy-to-understand {content_language}. You must always respond "
        "ONLY with valid JSON, no extra text. The JSON must be a single object with exactly "
        "these top-level keys: \"overall_summary\", \"overall_confidence\", \"patient_info\", "
        "\"doctor_info\", \"prescription_date\", \"follow_up_date\", \"probable_condition\", "
        "\"detected_conditions\", \"treatment_purpose\", \"drug_interactions\", \"medicines\".\n"
        f"- \"overall_summary\": a 2-3 sentence plain-{content_language} summary of the whole "
        "prescription.\n"
        "- \"overall_confidence\": one of \"high\", \"medium\", or \"low\" (English, "
        "lowercase), reflecting your overall confidence in reading the entire image.\n"
        "- \"patient_info\": an object with exactly these keys: \"name\" and \"age\", each in "
        f"simple {content_language} if visible on the prescription, otherwise a short phrase "
        "meaning \"not identified\", written in the target language.\n"
        "- \"doctor_info\": an object with exactly these keys: \"name\", \"degree\", "
        "\"hospital\" (hospital or chamber/clinic name), and \"phone\", each in simple "
        f"{content_language} if visible on the prescription, otherwise a short phrase meaning "
        "\"not identified\", written in the target language.\n"
        "- \"prescription_date\": the date the prescription was written, if visible; "
        "otherwise a short phrase meaning \"not mentioned\", written in the target language.\n"
        "- \"follow_up_date\": the next follow-up/visit date, if mentioned; otherwise a short "
        "phrase meaning \"not mentioned\", written in the target language.\n"
        f"- \"probable_condition\": one plain-{content_language} sentence about the patient's "
        "likely health condition. If a diagnosis is written on the prescription, use that. If "
        "not, infer the most likely condition(s) from the combination of prescribed medicines, "
        "and clearly say this is an inference (not a written diagnosis).\n"
        "- \"detected_conditions\": a short list (0-4 items) of specific condition names in "
        f"simple {content_language} that this prescription appears to address, chosen from "
        "common categories such as fever, diabetes, hypertension, gastric issues, pain, "
        "infection, or other conditions you identify from the medicines; use an empty list if "
        "nothing specific can be identified.\n"
        f"- \"treatment_purpose\": one plain-{content_language} sentence describing the "
        "overall goal of this treatment as a whole (e.g. what combination of problems it's "
        "meant to address), distinct from each individual medicine's own purpose.\n"
        f"- \"drug_interactions\": a plain-{content_language} note about any potentially "
        "significant interaction between the identified medicines; if none are apparent, say "
        "so clearly in the target language. This is informational only, not a substitute for "
        "a pharmacist's or doctor's advice.\n"
        "- \"medicines\": a list of objects, one per medicine, each with these exact keys: "
        "\"medicine_name\", \"purpose\", \"dosage\", \"duration\", \"side_effects\", "
        "\"precautions\", \"confidence\", \"timing\", \"high_risk\".\n"
        "\"duration\" is how long to take the medicine (e.g. \"5 days\", \"7 days\", "
        f"\"1 month\"), in simple {content_language}; if not specified on the prescription, "
        "say so clearly in the target language.\n"
        f"All text values must be written in simple {content_language}, understandable by "
        "someone with no medical background. Each medicine's \"confidence\" value must be "
        "exactly one of "
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
        f"Analyze it fully and respond in simple {content_language} as instructed."
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

    patient_info_raw = data.get("patient_info")
    if not isinstance(patient_info_raw, dict):
        patient_info_raw = {}
    patient_info = {
        "name": str(patient_info_raw.get("name") or "").strip(),
        "age": str(patient_info_raw.get("age") or "").strip(),
    }

    doctor_info_raw = data.get("doctor_info")
    if not isinstance(doctor_info_raw, dict):
        doctor_info_raw = {}
    doctor_info = {
        "name": str(doctor_info_raw.get("name") or "").strip(),
        "degree": str(doctor_info_raw.get("degree") or "").strip(),
        "hospital": str(doctor_info_raw.get("hospital") or "").strip(),
        "phone": str(doctor_info_raw.get("phone") or "").strip(),
    }

    detected_conditions_raw = data.get("detected_conditions")
    if not isinstance(detected_conditions_raw, list):
        detected_conditions_raw = []
    detected_conditions = [
        str(condition).strip() for condition in detected_conditions_raw if str(condition).strip()
    ]

    return {
        "overall_summary": str(data.get("overall_summary") or "").strip(),
        "overall_confidence": str(data.get("overall_confidence", "low")).strip().lower(),
        "patient_info": patient_info,
        "doctor_info": doctor_info,
        "prescription_date": str(data.get("prescription_date") or "").strip(),
        "follow_up_date": str(data.get("follow_up_date") or "").strip(),
        "probable_condition": str(data.get("probable_condition") or "").strip(),
        "detected_conditions": detected_conditions,
        "treatment_purpose": str(data.get("treatment_purpose") or "").strip(),
        "drug_interactions": str(data.get("drug_interactions") or "").strip(),
        "medicines": medicines,
    }


def get_chat_session(
    client: genai.Client, report_context: str, content_language: str, session_key: str
):
    """Get or create a Gemini chat session grounded in this prescription's report.

    Reusing one Chat object (cached in session_state under `session_key`) lets
    Gemini remember earlier turns in the conversation instead of treating every
    question as a fresh, context-free request. `session_key` should change
    whenever the underlying prescription or language changes, so a new image
    (or a language switch) starts a clean conversation instead of mixing
    contexts.
    """
    if session_key not in st.session_state:
        system_instruction = (
            "You are a friendly medical assistant chatbot inside the MediClarity AI app. "
            "A patient has just had the following prescription analyzed:\n\n"
            f"{report_context}\n\n"
            f"Answer the patient's follow-up questions in simple, easy-to-understand "
            f"{content_language}, in 2-4 sentences unless they ask for more detail. You can: "
            "explain why a medicine was likely prescribed, clarify dosage timing and whether "
            "to take it with or without food, explain possible drug interactions, explain "
            "symptoms or medical terms in plain language, and answer general questions related "
            "to this prescription. If a question needs clinical judgment or is unrelated to "
            "this prescription, say so and remind the patient to consult their doctor or "
            "pharmacist. You are not a substitute for professional medical advice."
        )
        st.session_state[session_key] = client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            ),
        )
    return st.session_state[session_key]


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

    st.markdown("### ⚙️ Accessibility")
    language = st.radio(
        "🌐 ভাষা / Language", ["বাংলা", "English"], horizontal=True, key="ui_language"
    )
    elderly_mode = st.checkbox("🧓 বড় ফন্ট মোড (Elderly Mode)", key="elderly_mode")
    colorblind_mode = st.checkbox("🎨 Color Blind Friendly Mode", key="colorblind_mode")

    st.divider()
    st.caption("⚠️ এটি একটি সহায়ক টুল, চিকিৎসা পরামর্শের বিকল্প নয়।")

T = TRANSLATIONS[language]
CONFIDENCE_BADGES = get_confidence_badges(language, colorblind_mode)
DANGER_COLORS = DANGER_COLORS_COLORBLIND if colorblind_mode else DANGER_COLORS_DEFAULT

if elderly_mode:
    render_html(
        """
        <style>
        .stApp, .stApp p, .stApp li, .stApp label, .stApp .stMarkdown {
            font-size: 1.25rem !important;
            line-height: 1.7 !important;
        }
        .stApp h1 { font-size: 2.75rem !important; }
        .stApp h3 { font-size: 1.6rem !important; }
        .stButton > button {
            font-size: 1.35rem !important;
            padding: 1rem 2rem !important;
        }
        </style>
        """
    )

render_html(
    f"""
    <div class="hero-banner">
        <div class="hero-icon">💊</div>
        <h1>MediClarity AI</h1>
        <p>{T['hero_subtitle']}</p>
    </div>
    """
)

step_cols = st.columns(3)
STEPS = [
    ("📤", T["step1_title"], T["step1_desc"]),
    ("🤖", T["step2_title"], T["step2_desc"]),
    ("📋", T["step3_title"], T["step3_desc"]),
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
    st.error(T["api_key_missing"])

uploaded_file = st.file_uploader(
    T["upload_label"],
    type=["jpg", "jpeg", "png"],
    key="prescription_uploader",
)

if uploaded_file is not None:
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error(T["image_read_error"])
        st.stop()

    st.image(image, caption=T["upload_caption"], use_container_width=True)

    analyze_clicked = st.button(
        T["analyze_button"],
        disabled=gemini_client is None,
        use_container_width=True,
        key="analyze_button",
    )

    already_cached = (
        st.session_state.get("cached_file_hash") == file_hash
        and st.session_state.get("cached_language") == language
    )

    if analyze_clicked and not already_cached:
        with st.status(T["status_analyzing"], expanded=True) as status:
            st.write(T["status_preparing"])
            try:
                st.write(T["status_reading"])
                result = explain_prescription(
                    gemini_client, image, content_language=T["content_language"]
                )
                st.session_state["cached_file_hash"] = file_hash
                st.session_state["cached_language"] = language
                st.session_state["cached_result"] = result
                status.update(label=T["status_done"], state="complete", expanded=False)
            except genai_errors.APIError as exc:
                status.update(label=T["status_api_error"], state="error")
                st.error(f"{T['error_api_prefix']}{exc}")
                st.stop()
            except httpx.TransportError:
                status.update(label=T["status_network_error"], state="error")
                st.error(T["error_network"])
                st.stop()
            except json.JSONDecodeError:
                status.update(label=T["status_parse_error"], state="error")
                st.error(T["error_parse"])
                st.stop()
            except Exception as exc:
                status.update(label=T["status_unexpected_error"], state="error")
                st.error(f"{T['error_unexpected_prefix']}{exc}")
                st.stop()

    if st.session_state.get("cached_file_hash") == file_hash:
        if st.session_state.get("cached_language") != language:
            st.info(T["language_switch_notice"])

        result = st.session_state["cached_result"]
        medicines = result.get("medicines", [])
        valid_medicines = [med for med in medicines if isinstance(med, dict)]

        if not medicines:
            st.warning(T["no_medicines"])
        elif not valid_medicines:
            st.error(T["bad_format"])
        else:
            st.success(f"{T['success_prefix']}{len(valid_medicines)}{T['success_suffix']}")

            report_text = build_text_report(result, T, valid_medicines, language)
            action_cols = st.columns(4)
            with action_cols[0]:
                st.download_button(
                    T["download_pdf_button"],
                    data=build_pdf_report(result, T, valid_medicines, language),
                    file_name="prescription_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf",
                )
            with action_cols[1]:
                st.download_button(
                    T["download_txt_button"],
                    data=report_text,
                    file_name="prescription_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_txt",
                )
            with action_cols[2]:
                print_button(T["print_button"])
            with action_cols[3]:
                copy_button(
                    report_text, T["copy_button"], T["copy_success"], T["copy_failed"], "report"
                )

            overall_confidence_key = str(result.get("overall_confidence", "low")).strip().lower()
            oc_label, oc_fg, oc_bg = CONFIDENCE_BADGES.get(
                overall_confidence_key, CONFIDENCE_BADGES["low"]
            )
            overall_summary = html.escape(result.get("overall_summary") or T["no_summary"])
            probable_condition = html.escape(
                result.get("probable_condition") or T["probable_condition_default"]
            )
            treatment_purpose = html.escape(
                result.get("treatment_purpose") or T["treatment_purpose_default"]
            )
            detected_conditions = [
                html.escape(str(c)) for c in result.get("detected_conditions", []) if str(c).strip()
            ]
            condition_chips = "".join(
                f'<span style="background-color:#1e3a5f; color:#93c5fd; padding:2px 10px; '
                f'border-radius:9999px; font-size:0.75rem; font-weight:600; margin-right:6px; '
                f'display:inline-block; margin-top:4px;">🏷️ {c}</span>'
                for c in detected_conditions
            )
            patient_info = result.get("patient_info", {})
            if not isinstance(patient_info, dict):
                patient_info = {}
            patient_name = html.escape(str(patient_info.get("name") or T["not_identified"]))
            patient_age = html.escape(str(patient_info.get("age") or T["not_identified"]))
            render_html(
                f"""
                <div class="card">
                    <h3>{T['summary_title']}
                        <span style="background-color:{oc_bg}; color:{oc_fg};
                            padding:2px 10px; border-radius:9999px; font-size:0.75rem;
                            font-weight:600; vertical-align:middle;">
                            {T['summary_overall']} {oc_label}
                        </span>
                    </h3>
                    <p><strong>{T['patient_name_label']}:</strong> {patient_name}
                        &nbsp;|&nbsp; <strong>{T['patient_age_label']}:</strong> {patient_age}</p>
                    <p>{overall_summary}</p>
                    <p><strong>{T['probable_condition_label']}:</strong> {probable_condition}</p>
                    {condition_chips}
                    <p><strong>{T['treatment_purpose_label']}:</strong> {treatment_purpose}</p>
                    <p><strong>{T['total_medicines_label']}:</strong> {len(valid_medicines)} {T['total_medicines_suffix']}</p>
                </div>
                """
            )

            speech_text = " ".join(
                part for part in [
                    result.get("overall_summary"),
                    result.get("probable_condition"),
                    result.get("treatment_purpose"),
                ] if part
            )
            speak_button(
                speech_text,
                T["listen_button"],
                T["tts_lang"],
                T["tts_error"],
                key=f"speak_{file_hash}_{language}",
            )

            doctor_info = result.get("doctor_info", {})
            if not isinstance(doctor_info, dict):
                doctor_info = {}
            doctor_name = html.escape(str(doctor_info.get("name") or T["not_identified"]))
            doctor_degree = html.escape(str(doctor_info.get("degree") or T["not_identified"]))
            doctor_hospital = html.escape(str(doctor_info.get("hospital") or T["not_identified"]))
            doctor_phone = html.escape(str(doctor_info.get("phone") or T["not_mentioned"]))
            prescription_date = html.escape(result.get("prescription_date") or T["not_mentioned"])
            follow_up_date = html.escape(result.get("follow_up_date") or T["not_mentioned"])

            admin_cols = st.columns(2)
            with admin_cols[0]:
                render_html(
                    f"""
                    <div class="card">
                        <h3>{T['doctor_title']}</h3>
                        <p><strong>{T['doctor_name_label']}:</strong> {doctor_name}</p>
                        <p><strong>{T['doctor_degree_label']}:</strong> {doctor_degree}</p>
                        <p><strong>{T['doctor_hospital_label']}:</strong> {doctor_hospital}</p>
                        <p><strong>{T['doctor_phone_label']}:</strong> {doctor_phone}</p>
                    </div>
                    """
                )
            with admin_cols[1]:
                render_html(
                    f"""
                    <div class="card">
                        <h3>{T['date_title']}</h3>
                        <p><strong>{T['date_label']}:</strong> {prescription_date}</p>
                        <p><strong>{T['followup_label']}:</strong> {follow_up_date}</p>
                    </div>
                    """
                )

            drug_interactions = html.escape(result.get("drug_interactions") or T["no_interaction"])
            render_html(
                f"""
                <div style="background-color:{DANGER_COLORS['bg']}; border-left:4px solid {DANGER_COLORS['border']};
                    border-radius:8px; padding:1rem 1.25rem; margin-bottom:1rem; color:{DANGER_COLORS['text']};">
                    💊⚠️ <strong>{T['interaction_label']}:</strong> {drug_interactions}
                </div>
                """
            )

            high_risk_meds = [med for med in valid_medicines if as_bool(med.get("high_risk"))]
            if high_risk_meds:
                risk_names = ", ".join(
                    html.escape(str(m.get("medicine_name", T["unknown_medicine"]))) for m in high_risk_meds
                )
                render_html(
                    f"""
                    <div style="background-color:{DANGER_COLORS['bg']}; border-left:4px solid {DANGER_COLORS['border']};
                        border-radius:8px; padding:1rem 1.25rem; margin-bottom:1rem; color:{DANGER_COLORS['text']};">
                        🚨 <strong>{T['high_risk_label']}:</strong> {risk_names} — {T['high_risk_warning']}
                    </div>
                    """
                )

            st.divider()
            st.subheader(T["schedule_title"])
            timing_rows = []
            for med in valid_medicines:
                timing = med.get("timing")
                if not isinstance(timing, dict):
                    timing = {}
                meal_key = str(timing.get("meal_relation", "unspecified")).strip().lower()
                timing_rows.append(
                    {
                        T["medicine_col"]: str(med.get("medicine_name", T["unknown_medicine"])),
                        T["morning_col"]: "✔️" if as_bool(timing.get("morning")) else "—",
                        T["afternoon_col"]: "✔️" if as_bool(timing.get("afternoon")) else "—",
                        T["night_col"]: "✔️" if as_bool(timing.get("night")) else "—",
                        T["meal_col"]: get_meal_label(language, meal_key),
                    }
                )
            st.table(timing_rows)

            st.divider()
            st.subheader(T["details_title"])

            card_cols = st.columns(2)
            for idx, med in enumerate(valid_medicines):
                name = html.escape(str(med.get("medicine_name", T["unknown_medicine"])))
                purpose = html.escape(str(med.get("purpose", T["no_info"])))
                dosage = html.escape(str(med.get("dosage", T["no_info"])))
                duration = html.escape(str(med.get("duration", T["not_mentioned"])))
                side_effects = html.escape(str(med.get("side_effects", T["no_info"])))
                precautions = html.escape(str(med.get("precautions", T["no_info"])))
                confidence_key = str(med.get("confidence", "low")).strip().lower()
                confidence_label, confidence_fg, confidence_bg = CONFIDENCE_BADGES.get(
                    confidence_key, CONFIDENCE_BADGES["low"]
                )
                high_risk_badge = (
                    f'<span style="background-color:{DANGER_COLORS["bg"]}; color:{DANGER_COLORS["text"]}; '
                    f'padding:2px 10px; border-radius:9999px; font-size:0.75rem; font-weight:600; '
                    f'vertical-align:middle;">{T["high_risk_badge"]}</span>'
                    if as_bool(med.get("high_risk"))
                    else ""
                )
                with card_cols[idx % 2]:
                    render_html(
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
                            <p><strong>{T['purpose_label']}:</strong> {purpose}</p>
                            <p><strong>{T['dosage_label']}:</strong> {dosage}</p>
                            <p><strong>{T['duration_label']}:</strong> {duration}</p>
                            <p><strong>{T['side_effects_label']}:</strong> {side_effects}</p>
                            <p><strong>{T['precautions_label']}:</strong> {precautions}</p>
                        </div>
                        """
                    )

            st.divider()
            st.subheader(T["chat_title"])
            st.caption(T["chat_intro"])

            chat_scope_key = f"{file_hash}_{language}"
            chat_messages_key = f"chat_messages_{chat_scope_key}"
            chat_session_key = f"chat_session_{chat_scope_key}"
            if chat_messages_key not in st.session_state:
                st.session_state[chat_messages_key] = []

            suggestion_cols = st.columns(3)
            suggestions = [T["chat_suggestion_1"], T["chat_suggestion_2"], T["chat_suggestion_3"]]
            clicked_suggestion = None
            for col, suggestion in zip(suggestion_cols, suggestions):
                with col:
                    if st.button(
                        suggestion,
                        key=f"suggest_{suggestions.index(suggestion)}_{chat_scope_key}",
                        use_container_width=True,
                    ):
                        clicked_suggestion = suggestion

            for msg in st.session_state[chat_messages_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_question = st.chat_input(T["chat_placeholder"]) or clicked_suggestion

            if user_question:
                st.session_state[chat_messages_key].append(
                    {"role": "user", "content": user_question}
                )
                with st.chat_message("user"):
                    st.markdown(user_question)
                with st.chat_message("assistant"):
                    with st.spinner(T["chat_thinking"]):
                        try:
                            chat = get_chat_session(
                                gemini_client, report_text, T["content_language"], chat_session_key
                            )
                            answer = chat.send_message(user_question).text
                        except Exception:
                            answer = T["chat_error"]
                    st.markdown(answer)
                st.session_state[chat_messages_key].append(
                    {"role": "assistant", "content": answer}
                )

        render_html(
            f"""
            <div style="background-color:#451a03; border-left:4px solid #d97706;
                border-radius:8px; padding:1rem 1.25rem; margin-top:1.25rem; color:#fde68a;">
                {T['disclaimer']}
            </div>
            """
        )
else:
    st.info(T["empty_state"])

render_html(
    """
    <div class="app-footer" style="text-align:center; padding:1rem; color:#6b7280; font-size:0.85rem;">
        MediClarity AI — Hackathon Project
    </div>
    """
)
