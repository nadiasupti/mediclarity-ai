"""MediClarity AI - Streamlit app that explains prescriptions in simple Bangla."""

import base64
import hashlib
import html
import io
import os
import json
import re
from datetime import datetime

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

# ============================== AI SAFETY THRESHOLDS ==============================
# Gates that stop the app from hallucinating on non-prescription/non-report
# images, or comparing documents that belong to different patients. All three
# are single knobs - loosen them if genuine (but messy) documents get rejected
# too often during a demo.
CLASSIFY_CONFIDENCE_MIN = 0.85   # below this (or wrong category) -> reject, no analysis
EXTRACTION_CONFIDENCE_MIN = 0.70  # below this -> ask for a clearer image
PATIENT_MATCH_MIN = 70            # patient match score below this -> mismatch warning, no medical comparison

# image_category values the classifier may return (shared by both analyzers).
IMAGE_CATEGORIES = (
    "prescription", "test_report", "medicine_package", "medicine_strip",
    "pharmacy_label", "hospital_bill", "invoice", "non_medical", "unknown",
)


def get_category_label(T: dict, category: str) -> str:
    """Localized display name for a classifier image_category (falls back to unknown)."""
    key = category if category in IMAGE_CATEGORIES else "unknown"
    return T.get(f"cat_{key}", T.get("cat_unknown", key))


def as_confidence(value) -> float:
    """Coerce an AI-provided confidence to a float in [0, 1].

    Gemini may return the confidence as a number, a numeric string, or even a
    percentage (e.g. 92 meaning 0.92); anything unparseable is treated as 0.0
    so the caller fails safe (rejects) rather than trusting a bad reading.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if num > 1:
        num = num / 100.0
    return max(0.0, min(1.0, num))

# ============================== CHATBOT FAB CONFIG ==============================
# Single knobs for the floating chat launcher's position/size. Horizontal
# position is intentionally NOT configurable here - it's always pinned to the
# right edge of the viewport by design. Change CHATBOT_BOTTOM_PX to move the
# icon up/down; nothing else needs to change.
CHATBOT_RIGHT_PX = 24
CHATBOT_BOTTOM_PX = 40
CHATBOT_SIZE_DESKTOP_PX = 76
CHATBOT_SIZE_MOBILE_PX = 60

# Drop a file with one of these exact names into assets/chatbot/ to replace
# the fallback 🤖 emoji with a custom animated icon - no other code changes
# needed. First match wins. GIF/WebP are the recommended choice (rendered as
# a plain animated image, zero extra JS); Lottie (.json) is also supported
# but needs a small player script loaded from a CDN (see render_chat_panel).
CHATBOT_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "chatbot")
CHATBOT_ASSET_CANDIDATES = ("robot.webp", "robot.gif", "robot.json")


@st.cache_data
def load_chatbot_asset():
    """Look for a custom chatbot FAB icon in assets/chatbot/.

    Cached with st.cache_data so a multi-MB GIF/WebP isn't re-read and
    re-base64-encoded from disk on every Streamlit rerun. The cache is keyed
    off this function's own source, so it refreshes whenever app.py is
    edited/saved; if you only swap the asset FILE without touching app.py,
    restart the Streamlit server (or clear the cache) to pick it up.

    Returns (kind, payload) where kind is "image" (payload = a base64 data:
    URI, ready to drop straight into a CSS background-image or <img src>) or
    "lottie" (payload = the raw Lottie JSON text). Returns (None, None) if no
    asset file is present yet, so the caller can fall back to the plain emoji.
    """
    for filename in CHATBOT_ASSET_CANDIDATES:
        path = os.path.join(CHATBOT_ASSET_DIR, filename)
        if not os.path.isfile(path):
            continue
        if filename.endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return "lottie", f.read()
            except (OSError, UnicodeDecodeError):
                continue
        mime = "image/webp" if filename.endswith(".webp") else "image/gif"
        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
        except OSError:
            continue
        return "image", f"data:{mime};base64,{encoded}"
    return None, None


# Separate from the FAB icon above - this is the small robot shown in the
# chat panel's header and next to every assistant reply. Drop a file with
# one of these names into the same assets/chatbot/ folder to replace the
# fallback 🤖 emoji there too; no other code changes needed.
CHATBOT_AVATAR_CANDIDATES = ("avatar.png", "avatar.webp", "avatar.gif")
CHATBOT_AVATAR_MIME = {"png": "image/png", "webp": "image/webp", "gif": "image/gif"}


@st.cache_data
def load_chat_avatar_asset():
    """Look for a custom chat avatar icon in assets/chatbot/.

    Returns a base64 data: URI, or None if no avatar.png/.webp/.gif is
    present yet - callers should fall back to the plain 🤖 emoji in that case.
    """
    for filename in CHATBOT_AVATAR_CANDIDATES:
        path = os.path.join(CHATBOT_ASSET_DIR, filename)
        if not os.path.isfile(path):
            continue
        ext = filename.rsplit(".", 1)[-1]
        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
        except OSError:
            continue
        return f"data:{CHATBOT_AVATAR_MIME[ext]};base64,{encoded}"
    return None
#----------------------------CONFIDENCE --------------------------------------------------------------------------------
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

CONFIDENCE_PERCENT = {"high": 85, "medium": 60, "low": 35}
#-------------------------MEAL RELATIONSHIP --------------------------------------------------------------------------------    
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


def parse_duration_days(med: dict):
    """Read a medicine's AI-provided duration_days as a positive int, or None.

    The AI is asked to convert free-text duration ("1 week") into a plain
    integer, but LLM JSON output isn't always perfectly typed (e.g. it may
    return "7" as a string, or 0/negative for "not specified") - this coerces
    defensively instead of trusting the type.
    """
    raw = med.get("duration_days")
    try:
        days = int(float(raw))
    except (TypeError, ValueError):
        return None
    return days if days > 0 else None

def build_treatment_timeline(valid_medicines: list) -> list:
    # CHANGED: Timeline now uses fixed demo-friendly tabs like Day 1-5, 6-10, 11-15, 16-20, 21-30.
    # CHANGED: This matches the premium dashboard screenshot style instead of creating random duration-based tabs.

    fixed_ranges = [
        (1, 5),
        (6, 10),
        (11, 15),
        (16, 20),
        (21, 30),
    ]

    ranges = []

    for start_day, end_day in fixed_ranges:
        active_meds = []

        for med in valid_medicines:
            duration_days = parse_duration_days(med)

            # CHANGED: If duration is unknown, show medicine in all timeline ranges as ongoing.
            if duration_days is None:
                active_meds.append(med)

            # CHANGED: If medicine duration overlaps with this range, show it in this tab.
            elif duration_days >= start_day:
                active_meds.append(med)

        ranges.append(
            {
                "start_day": start_day,
                "end_day": end_day,
                "medicines": active_meds,
            }
        )

    return ranges

#-------------------------DANGER COLORS --------------------------------------------------------------------------------
# Danger/alert colors (drug interactions, high-risk medicine). The colorblind
# variant swaps red for dark-indigo/amber, since red-on-dark can be hard to
# distinguish from other saturated colors used elsewhere in the UI.
DANGER_COLORS_DEFAULT = {"bg": "#450a0a", "border": "#dc2626", "text": "#fecaca"}
DANGER_COLORS_COLORBLIND = {"bg": "#1e1b4b", "border": "#f59e0b", "text": "#fde68a"}

TEST_STATUS_LABELS = {
    "বাংলা": {"normal": "স্বাভাবিক", "borderline": "সীমারেখায়", "abnormal": "অস্বাভাবিক"},
    "English": {"normal": "Normal", "borderline": "Borderline", "abnormal": "Abnormal"},
}
TEST_STATUS_EMOJI_DEFAULT = {"normal": "🟢", "borderline": "🟡", "abnormal": "🔴"}
TEST_STATUS_EMOJI_COLORBLIND = {"normal": "🔵", "borderline": "🟠", "abnormal": "🟣"}
TEST_STATUS_COLORS_DEFAULT = {
    "normal": ("#166534", "#dcfce7"),
    "borderline": ("#92400e", "#fef3c7"),
    "abnormal": ("#991b1b", "#fee2e2"),
}
TEST_STATUS_COLORS_COLORBLIND = {
    "normal": ("#1e3a8a", "#dbeafe"),
    "borderline": ("#9a3412", "#fed7aa"),
    "abnormal": ("#6b21a8", "#f3e8ff"),
}


def get_test_status_badges(language: str, colorblind_mode: bool) -> dict:
    labels = TEST_STATUS_LABELS[language]
    emojis = TEST_STATUS_EMOJI_COLORBLIND if colorblind_mode else TEST_STATUS_EMOJI_DEFAULT
    colors = TEST_STATUS_COLORS_COLORBLIND if colorblind_mode else TEST_STATUS_COLORS_DEFAULT
    return {
        key: (f"{emojis[key]} {labels[key]}", colors[key][0], colors[key][1])
        for key in ("normal", "borderline", "abnormal")
    }

TRANSLATIONS = {
     #==========================bangla TRANSLATIONS ==========================
    "বাংলা": {
        "hero_subtitle": "প্রেসক্রিপশনের ছবি আপলোড করুন — সহজ বাংলায় বুঝে নিন আপনার ওষুধ সম্পর্কে",
        "step1_title": "ছবি আপলোড করুন", "step1_desc": "প্রেসক্রিপশনের স্পষ্ট ছবি দিন",
        "step2_title": "AI বিশ্লেষণ করবে", "step2_desc": "Gemini Vision ছবিটি পড়ে বুঝবে",
        "step3_title": "ফলাফল দেখুন", "step3_desc": "সহজ বাংলায় বিস্তারিত পাবেন",
        "upload_label": "প্রেসক্রিপশনের ছবি আপলোড করুন",
        "upload_caption": "আপলোড করা প্রেসক্রিপশন",
        "analyze_button": "🔍 প্রেসক্রিপশন বিশ্লেষণ করুন",
        "reupload_expander_label": "🔄 নতুন প্রেসক্রিপশন আপলোড করুন",
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
        "all_keys_busy": (
            "⏳ এই মুহূর্তে সার্ভার ব্যস্ত (সব API key-এর আজকের সীমা শেষ)। "
            "দয়া করে একটু পরে আবার চেষ্টা করুন।"
        ),
        "error_parse": "AI-এর উত্তর সঠিকভাবে বোঝা যায়নি। দয়া করে আবার চেষ্টা করুন।",
        "error_unexpected_prefix": "একটি অপ্রত্যাশিত সমস্যা হয়েছে: ",
        "pdf_generation_failed": "PDF রিপোর্ট তৈরি করা যায়নি। আপনি TXT বা প্রিন্ট বিকল্প ব্যবহার করতে পারেন।",
        "status_classifying": "🔎 ছবিটি যাচাই করা হচ্ছে...",
        "classify_not_prescription": (
            "❌ এই ছবিটি ডাক্তারের প্রেসক্রিপশন বলে মনে হচ্ছে না ({category})। "
            "দয়া করে একটি সঠিক হাতে লেখা বা প্রিন্ট করা প্রেসক্রিপশনের ছবি আপলোড করুন।"
        ),
        "classify_not_test_report": (
            "❌ এই ফাইলটি মেডিকেল টেস্ট রিপোর্ট বলে মনে হচ্ছে না ({category})। "
            "দয়া করে একটি সঠিক ল্যাব/টেস্ট রিপোর্ট (ছবি বা PDF) আপলোড করুন।"
        ),
        "extraction_unclear": (
            "⚠️ ছবিটি যথেষ্ট স্পষ্ট নয়, নির্ভরযোগ্যভাবে পড়া যাচ্ছে না। "
            "দয়া করে আরও স্পষ্ট, ভালো আলোয় তোলা একটি ছবি আপলোড করুন।"
        ),
        "cat_prescription": "প্রেসক্রিপশন",
        "cat_test_report": "টেস্ট রিপোর্ট",
        "cat_medicine_package": "ওষুধের প্যাকেট/বাক্স",
        "cat_medicine_strip": "ওষুধের স্ট্রিপ/পাতা",
        "cat_pharmacy_label": "ফার্মেসির লেবেল",
        "cat_hospital_bill": "হাসপাতালের বিল",
        "cat_invoice": "ইনভয়েস/রসিদ",
        "cat_non_medical": "চিকিৎসা-বহির্ভূত ছবি",
        "cat_unknown": "অজানা ধরন",
        "compare_status_match": "✅ মিল পাওয়া গেছে",
        "compare_status_partial": "⚠️ আংশিক মিল",
        "compare_status_mismatch": "❌ অমিল শনাক্ত হয়েছে",
        "compare_mismatch_warning": (
            "⚠️ এই টেস্ট রিপোর্টটি আপলোড করা প্রেসক্রিপশনের রোগীর নাও হতে পারে। "
            "তুলনার ওপর নির্ভর করার আগে নিশ্চিত করুন যে দুটো ফাইল একই রোগীর।"
        ),
        "compare_matched_label": "যা মিলেছে",
        "compare_mismatched_label": "যা মেলেনি / যাচাই করা যায়নি",
        "compare_score_label": "রোগীর মিল স্কোর",
        "compare_no_medical_on_mismatch": "রোগীর তথ্য না মেলায় কোনো মেডিকেল তুলনা দেখানো হয়নি।",
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
        "chat_panel_title": "MediClarity AI সহকারী",
        "chat_panel_subtitle": "প্রেসক্রিপশন নিয়ে আপনার প্রশ্ন করুন",
        "chat_beta_badge": "Beta",
        "chat_welcome_message": (
            "👋 হ্যালো! আমি আপনার MediClarity AI সহকারী। আপনি আপনার প্রেসক্রিপশন নিয়ে "
            "যেকোনো প্রশ্ন করতে পারেন।"
        ),
        "timeline_title": "🗓️ AI চিকিৎসার সময়রেখা",
        "timeline_intro": "প্রতিটা ওষুধের মেয়াদ অনুযায়ী দিন-ভিত্তিক সম্পূর্ণ চিকিৎসা পরিকল্পনা।",
        "timeline_day_single": "দিন {day}",
        "timeline_day_range": "দিন {start}–{end}",
        "timeline_full_course": "পূর্ণ কোর্স",
        "timeline_ongoing_badge": "চলমান",
        "timeline_no_meds": "এই সময়ে কোনো ওষুধ নেই",
        "test_report_title": "🧪 মেডিকেল টেস্ট রিপোর্ট ব্যাখ্যা",
        "test_report_intro": (
            "রক্ত পরীক্ষা, ইউরিন টেস্ট বা অন্য কোনো মেডিকেল রিপোর্টের ছবি বা PDF আপলোড করুন "
            "— সহজ ভাষায় বুঝে নিন প্রতিটা ফলাফলের মানে।"
        ),
        "test_upload_label": "টেস্ট রিপোর্ট আপলোড করুন (ছবি বা PDF)",
        "test_upload_caption": "আপলোড করা টেস্ট রিপোর্ট",
        "test_analyze_button": "🔬 টেস্ট রিপোর্ট বিশ্লেষণ করুন",
        "test_file_read_error": "ফাইলটি পড়া যায়নি। দয়া করে একটি সঠিক JPG/JPEG/PNG/PDF ফাইল আপলোড করুন।",
        "test_status_analyzing": "🔬 টেস্ট রিপোর্ট বিশ্লেষণ করা হচ্ছে...",
        "test_status_reading": "🤖 Gemini দিয়ে রিপোর্ট পড়া হচ্ছে... এটা কয়েক সেকেন্ড সময় নিতে পারে।",
        "no_tests": "কোনো টেস্ট ফলাফল শনাক্ত করা যায়নি।",
        "test_summary_title": "🧪 টেস্ট রিপোর্ট সারসংক্ষেপ",
        "lab_title": "🏥 ল্যাব/হাসপাতালের তথ্য",
        "lab_name_label": "নাম",
        "report_date_label": "রিপোর্টের তারিখ",
        "referring_doctor_label": "রেফারকারী ডাক্তার",
        "total_tests_label": "মোট টেস্ট",
        "tests_section_title": "🔬 টেস্ট ফলাফলের বিস্তারিত",
        "test_value_label": "মান",
        "reference_range_label": "স্বাভাবিক সীমা",
        "test_purpose_label": "কেন করা হয়েছে",
        "test_explanation_label": "ব্যাখ্যা",
        "unknown_test": "অজানা টেস্ট",
        "compare_button": "⚖️ প্রেসক্রিপশনের সাথে তুলনা করুন",
        "compare_title": "⚖️ প্রেসক্রিপশন ও টেস্ট রিপোর্ট তুলনা",
        "compare_need_both": (
            "তুলনা করতে প্রথমে একটা প্রেসক্রিপশন এবং একটা টেস্ট রিপোর্ট — দুটোই "
            "বিশ্লেষণ করুন।"
        ),
        "compare_thinking": "তুলনা করা হচ্ছে...",
        "test_disclaimer": (
            "⚠️ দ্রষ্টব্য: এই ব্যাখ্যা শুধুমাত্র শিক্ষামূলক উদ্দেশ্যে, এটি কোনো মেডিকেল রোগ "
            "নির্ণয় নয়। ফলাফল সম্পর্কে যেকোনো সিদ্ধান্তের জন্য সবসময় আপনার ডাক্তারের "
            "পরামর্শ নিন।"
        ),
        "nav_summary": "সারাংশ",
        "nav_medicines": "ওষুধসমূহ",
        "nav_schedule": "দৈনিক সময়সূচি",
        "nav_advice": "পরামর্শ ও সতর্কতা",
        "nav_test_report": "রিপোর্ট বিশ্লেষণ",
        "nav_about": "সম্পর্কে",
        "welcome_title": "👋 স্বাগতম!",
        "welcome_subtitle": "আপনার প্রেসক্রিপশনের AI বিশ্লেষণ করা হয়েছে। নিচে সারসংক্ষেপ দেখুন।",
        "stat_condition_label": "সম্ভাব্য রোগ নির্ণয়",
        "stat_purpose_label": "চিকিৎসার উদ্দেশ্য",
        "stat_confidence_label": "আস্থা স্তর",
        "stat_risk_label": "ঝুঁকি স্তর",
        "ai_inference_badge": "🤖 AI অনুমান",
        "ai_analysis_badge": "🤖 AI বিশ্লেষণ",
        "risk_high": "উচ্চ",
        "risk_low": "নিম্ন",
        "about_heading": "ℹ️ MediClarity AI সম্পর্কে",
        "about_body": (
            "প্রেসক্রিপশনের হাতের লেখা বোঝা প্রায়ই কঠিন। **MediClarity AI** একটি ছবি থেকেই "
            "Google Gemini Vision দিয়ে আপনার ওষুধের নাম, উদ্দেশ্য, সেবনবিধি, পার্শ্বপ্রতিক্রিয়া "
            "ও সতর্কতা সহজ বাংলায় বুঝিয়ে দেয়।"
        ),
        "about_tech_heading": "🛠️ ব্যবহৃত প্রযুক্তি",
        "about_disclaimer": "⚠️ এটি একটি সহায়ক টুল, চিকিৎসা পরামর্শের বিকল্প নয়।",
        "sidebar_tagline": "সহজ ভাষায় মেডিকেল সমাধান",
        "sidebar_description": (
            "প্রেসক্রিপশন সহজ ভাষায় বোঝাই আমাদের লক্ষ্য। MediClarity AI একটি স্মার্ট সহকারী, "
            "যা Google Gemini Vision দিয়ে আপনার প্রেসক্রিপশন পড়ে বুঝিয়ে দেয়।"
        ),
        "accessibility_heading": "⚙️ Accessibility",
    },
    #==========================ENGLISH TRANSLATIONS ==========================
    "English": {
        "hero_subtitle": "Upload a prescription image — understand your medicines in plain English",
        "step1_title": "Upload Image", "step1_desc": "Provide a clear prescription photo",
        "step2_title": "AI Analyzes It", "step2_desc": "Gemini Vision reads and understands it",
        "step3_title": "View Results", "step3_desc": "Get plain-English details",
        "upload_label": "Upload a prescription image",
        "upload_caption": "Uploaded prescription",
        "analyze_button": "🔍 Analyze Prescription",
        "reupload_expander_label": "🔄 Upload a new prescription",
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
        "all_keys_busy": (
            "⏳ The server is busy right now (all API keys have hit today's limit). "
            "Please try again in a little while."
        ),
        "error_parse": "Could not properly understand the AI's response. Please try again.",
        "error_unexpected_prefix": "An unexpected error occurred: ",
        "pdf_generation_failed": "Could not generate the PDF report. You can still use the TXT or Print options.",
        "status_classifying": "🔎 Checking the image...",
        "classify_not_prescription": (
            "❌ This image does not appear to be a doctor's prescription ({category}). "
            "Please upload a valid handwritten or printed prescription."
        ),
        "classify_not_test_report": (
            "❌ This file does not appear to be a medical test report ({category}). "
            "Please upload a valid lab/test report (image or PDF)."
        ),
        "extraction_unclear": (
            "⚠️ The image isn't clear enough to read reliably. "
            "Please upload a clearer, well-lit photo."
        ),
        "cat_prescription": "prescription",
        "cat_test_report": "test report",
        "cat_medicine_package": "medicine package/box",
        "cat_medicine_strip": "medicine strip/blister",
        "cat_pharmacy_label": "pharmacy label",
        "cat_hospital_bill": "hospital bill",
        "cat_invoice": "invoice/receipt",
        "cat_non_medical": "non-medical image",
        "cat_unknown": "unknown type",
        "compare_status_match": "✅ Match",
        "compare_status_partial": "⚠️ Partial match",
        "compare_status_mismatch": "❌ Mismatch detected",
        "compare_mismatch_warning": (
            "⚠️ This medical report may not belong to the uploaded prescription. "
            "Please verify that both files belong to the same patient before relying on this comparison."
        ),
        "compare_matched_label": "What matched",
        "compare_mismatched_label": "What didn't match / couldn't verify",
        "compare_score_label": "Patient match score",
        "compare_no_medical_on_mismatch": "No medical comparison is shown because the patient details do not match.",
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
        "chat_panel_title": "MediClarity AI Assistant",
        "chat_panel_subtitle": "Ask your questions about the prescription",
        "chat_beta_badge": "Beta",
        "chat_welcome_message": (
            "👋 Hello! I'm your MediClarity AI assistant. You can ask me anything about your "
            "prescription."
        ),
        "timeline_title": "🗓️ AI Treatment Timeline",
        "timeline_intro": "Your full day-by-day treatment plan based on each medicine's duration.",
        "timeline_day_single": "Day {day}",
        "timeline_day_range": "Day {start}–{end}",
        "timeline_full_course": "Full course",
        "timeline_ongoing_badge": "Ongoing",
        "timeline_no_meds": "No medicine at this time",
        "test_report_title": "🧪 Medical Test Report Interpreter",
        "test_report_intro": (
            "Upload a photo or PDF of a blood test, urine test, or any other medical "
            "report — understand what each result means in plain language."
        ),
        "test_upload_label": "Upload a test report (image or PDF)",
        "test_upload_caption": "Uploaded test report",
        "test_analyze_button": "🔬 Analyze Test Report",
        "test_file_read_error": "Could not read the file. Please upload a valid JPG/JPEG/PNG/PDF file.",
        "test_status_analyzing": "🔬 Analyzing test report...",
        "test_status_reading": "🤖 Reading the report with Gemini... this may take a few seconds.",
        "no_tests": "No test results were identified.",
        "test_summary_title": "🧪 Test Report Summary",
        "lab_title": "🏥 Lab/Hospital Information",
        "lab_name_label": "Name",
        "report_date_label": "Report Date",
        "referring_doctor_label": "Referring Doctor",
        "total_tests_label": "Total Tests",
        "tests_section_title": "🔬 Test Result Details",
        "test_value_label": "Value",
        "reference_range_label": "Reference Range",
        "test_purpose_label": "Why It Was Done",
        "test_explanation_label": "Explanation",
        "unknown_test": "Unknown test",
        "compare_button": "⚖️ Compare with Prescription",
        "compare_title": "⚖️ Prescription vs Test Report Comparison",
        "compare_need_both": (
            "To compare, first analyze both a prescription and a test report above."
        ),
        "compare_thinking": "Comparing...",
        "test_disclaimer": (
            "⚠️ Note: This interpretation is for educational purposes only and is not a "
            "medical diagnosis. Always consult your doctor about any decisions based on "
            "these results."
        ),
        "nav_summary": "Summary",
        "nav_medicines": "Medicines",
        "nav_schedule": "Daily Schedule",
        "nav_advice": "Advice & Warnings",
        "nav_test_report": "Report Analysis",
        "nav_about": "About",
        "welcome_title": "👋 Welcome!",
        "welcome_subtitle": "Your prescription has been analyzed by AI. See the summary below.",
        "stat_condition_label": "Probable Diagnosis",
        "stat_purpose_label": "Treatment Purpose",
        "stat_confidence_label": "Confidence Level",
        "stat_risk_label": "Risk Level",
        "ai_inference_badge": "🤖 AI Inference",
        "ai_analysis_badge": "🤖 AI Analysis",
        "risk_high": "High",
        "risk_low": "Low",
        "about_heading": "ℹ️ About MediClarity AI",
        "about_body": (
            "Prescription handwriting is often hard to read. **MediClarity AI** uses Google "
            "Gemini Vision to explain your medicine names, purpose, dosage, side effects, and "
            "precautions from just a photo, in plain language."
        ),
        "about_tech_heading": "🛠️ Tech Used",
        "about_disclaimer": "⚠️ This is a supportive tool, not a substitute for medical advice.",
        "sidebar_tagline": "Simple Language Medical Solution",
        "sidebar_description": (
            "Making prescriptions easy to understand is our goal. MediClarity AI is a smart "
            "assistant that uses Google Gemini Vision to read and explain your prescription."
        ),
        "accessibility_heading": "⚙️ Accessibility",
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
    # json.dumps() does not escape "</", so a "</script>" substring inside `text`
    # (e.g. AI-transcribed text from an adversarial prescription image) would
    # otherwise close this script tag early and let the rest run as live HTML/JS.
    safe_text = json.dumps(text).replace("</", "<\\/")
    safe_success = json.dumps(success_text).replace("</", "<\\/")
    safe_failed = json.dumps(failed_text).replace("</", "<\\/")
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

    lines.append(T["timeline_title"])
    lines.append("-" * len(T["timeline_title"]))
    for day_range in build_treatment_timeline(valid_medicines):
        if day_range["start_day"] == day_range["end_day"]:
            day_label = T["timeline_day_single"].format(day=day_range["start_day"])
        else:
            day_label = T["timeline_day_range"].format(
                start=day_range["start_day"], end=day_range["end_day"]
            )
        lines.append(day_label)
        for flag_key, slot_label in (
            ("morning", T["morning_col"]),
            ("afternoon", T["afternoon_col"]),
            ("night", T["night_col"]),
        ):
            slot_names = []
            for med in day_range["medicines"]:
                timing = med.get("timing")
                if not isinstance(timing, dict):
                    timing = {}
                if not as_bool(timing.get(flag_key)):
                    continue
                meal_key = str(timing.get("meal_relation", "unspecified")).strip().lower()
                meal_text = get_meal_label(language, meal_key)
                ongoing_suffix = (
                    f" ({T['timeline_ongoing_badge']})" if parse_duration_days(med) is None else ""
                )
                slot_names.append(
                    f"{med.get('medicine_name', T['unknown_medicine'])} - {meal_text}{ongoing_suffix}"
                )
            names_text = ", ".join(slot_names) if slot_names else T["timeline_no_meds"]
            lines.append(f"   {slot_label}: {names_text}")
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

    heading(T["timeline_title"])
    for day_range in build_treatment_timeline(valid_medicines):
        if day_range["start_day"] == day_range["end_day"]:
            day_label = T["timeline_day_single"].format(day=day_range["start_day"])
        else:
            day_label = T["timeline_day_range"].format(
                start=day_range["start_day"], end=day_range["end_day"]
            )
        pdf.set_font("Bengali", size=12)
        pdf.multi_cell(0, 7, strip_emoji(day_label), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Bengali", size=11)
        for flag_key, slot_label in (
            ("morning", T["morning_col"]),
            ("afternoon", T["afternoon_col"]),
            ("night", T["night_col"]),
        ):
            slot_names = []
            for med in day_range["medicines"]:
                timing = med.get("timing")
                if not isinstance(timing, dict):
                    timing = {}
                if not as_bool(timing.get(flag_key)):
                    continue
                meal_key = str(timing.get("meal_relation", "unspecified")).strip().lower()
                meal_text = get_meal_label(language, meal_key)
                ongoing_suffix = (
                    f" ({T['timeline_ongoing_badge']})" if parse_duration_days(med) is None else ""
                )
                slot_names.append(
                    f"{med.get('medicine_name', T['unknown_medicine'])} - {meal_text}{ongoing_suffix}"
                )
            names_text = ", ".join(slot_names) if slot_names else T["timeline_no_meds"]
            line(f"{slot_label}: {names_text}")
        pdf.ln(2)

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
    layout="wide",
)

render_html(
    """
    <!-- CHANGED: Imported stronger Bangla font weights for a premium dashboard look -->
    <link href="https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
    /* CHANGED: Added global theme variables so every UI element uses one consistent color system */
    :root {
        --bg-main: #07111f;
        --bg-deep: #050914;
        --card-bg: rgba(15, 30, 52, 0.78);
        --card-bg-strong: rgba(22, 39, 70, 0.92);
        --sidebar-gradient: linear-gradient(180deg, #35208f 0%, #1b2450 45%, #0a1020 100%);
        --primary: #6d4aff;
        --primary-soft: #8b6cff;
        --cyan: #14b8d4;
        --text-main: #f8fafc;
        --text-muted: #b6c2d6;
        --border-soft: rgba(148, 163, 184, 0.18);
        --shadow-card: 0 18px 45px rgba(0, 0, 0, 0.32);
    }

    /* CHANGED: Applied Bangla-friendly font and default readable text color */
    html, body, .stApp {
        font-family: 'Hind Siliguri', 'Segoe UI', sans-serif !important;
        color: var(--text-main);
    }

    /* CHANGED: Replaced flat dark background with premium navy radial gradient */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(109, 74, 255, 0.24), transparent 34%),
            radial-gradient(circle at top right, rgba(20, 184, 212, 0.16), transparent 30%),
            linear-gradient(135deg, #07111f 0%, #081629 45%, #050914 100%);
    }

    /* CHANGED: Streamlit top header made transparent and glassy */
    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.55);
        backdrop-filter: blur(14px);
    }

    /* CHANGED: Main content spacing adjusted to match dashboard layout */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1500px;
    }

    /* CHANGED: Heading colors fixed for dark premium UI */
    h1, h2, h3, h4 {
        color: var(--text-main) !important;
        letter-spacing: -0.02em;
    }

    /* CHANGED: Body text contrast improved */
    p, li, label, span {
        color: #dbe4f0;
    }

    /* CHANGED: Sidebar turned into purple-blue gradient like the target design */
    [data-testid="stSidebar"] > div:first-child {
        background: var(--sidebar-gradient);
        border-right: 1px solid rgba(255,255,255,0.08);
        box-shadow: 10px 0 35px rgba(0,0,0,0.28);
    }

    /* CHANGED: Sidebar text forced to light color for readability */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #f8fafc !important;
    }

    /* CHANGED: Sidebar header card upgraded to glassmorphism style */
    .sidebar-header-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 22px;
        padding: 1.15rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 18px 35px rgba(0,0,0,0.25);
        backdrop-filter: blur(18px);
        /* Forces the same edge-to-edge width as the nav buttons below it -
           without this, the card's own padding could sit "inside" its width
           differently than the buttons' padding, making the two look misaligned. */
        width: 100%;
        box-sizing: border-box;
    }

    /* CHANGED: Sidebar logo circle redesigned with glowing gradient */
    .sidebar-logo-circle {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        background: linear-gradient(135deg, #36d1dc, #6d4aff, #a855f7);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.55rem;
        flex-shrink: 0;
        box-shadow: 0 0 24px rgba(109, 74, 255, 0.55);
    }

    /* CHANGED: Sidebar title made bolder */
    .sidebar-header-title {
        font-weight: 800;
        font-size: 1.12rem;
        color: #ffffff;
    }

    /* CHANGED: Sidebar tagline softened */
    .sidebar-header-tagline {
        font-size: 0.76rem;
        color: #d8d5ff;
        margin-top: 2px;
    }

    /* CHANGED: Sidebar description spacing improved */
    .sidebar-header-desc {
        font-size: 0.84rem;
        color: #e2e8f0;
        line-height: 1.65;
        margin-top: 0.85rem;
    }

    /* CHANGED: Sidebar navigation buttons converted into dashboard pill buttons */
    .st-key-sidebar_nav {
        width: 100%;
    }
    /* Streamlit wraps a container's contents in its own nested block/element
       divs, which can carry their own left/right padding - reset those to 0
       so the button group's edges match the header card's full-width span. */
    .st-key-sidebar_nav,
    .st-key-sidebar_nav > div,
    .st-key-sidebar_nav [data-testid="stVerticalBlock"],
    .st-key-sidebar_nav [data-testid="element-container"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    .st-key-sidebar_nav .stButton {
        width: 100%;
    }
    .st-key-sidebar_nav .stButton > button {
        width: 100%;
        box-sizing: border-box;
        text-align: left;
        justify-content: flex-start;
        font-weight: 700;
        border-radius: 14px;
        margin-bottom: 5px;
        padding: 0.75rem 0.9rem;
        transition: all 0.2s ease;
    }

    /* CHANGED: Inactive sidebar buttons made glassy */
    .st-key-sidebar_nav .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.05);
        color: #dbe4f0;
    }

    /* CHANGED: Sidebar hover gets premium purple highlight */
    .st-key-sidebar_nav .stButton > button[kind="secondary"]:hover {
        background: rgba(109, 74, 255, 0.22);
        border-color: rgba(139, 108, 255, 0.45);
        color: #ffffff;
        transform: translateX(3px);
    }

    /* CHANGED: Active sidebar tab uses purple gradient */
    .st-key-sidebar_nav .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6d4aff, #4f46e5);
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 12px 25px rgba(109, 74, 255, 0.35);
        color: #ffffff;
    }

    /* CHANGED: Hero banner upgraded from flat blue to premium gradient card */
    .hero-banner {
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.18), transparent 28%),
            linear-gradient(135deg, rgba(109, 74, 255, 0.96), rgba(20, 184, 212, 0.72));
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 26px;
        padding: 2.4rem 1.8rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 24px 55px rgba(0, 0, 0, 0.32);
        overflow: hidden;
    }

    /* CHANGED: Hero icon enlarged with glow */
    .hero-banner .hero-icon {
        font-size: 3rem;
        filter: drop-shadow(0 0 16px rgba(255,255,255,0.35));
    }

    /* CHANGED: Hero title made stronger */
    .hero-banner h1 {
        color: white !important;
        margin: 0.25rem 0 0.4rem 0;
        font-size: 2.25rem;
        font-weight: 800;
    }

    /* CHANGED: Hero subtitle contrast improved */
    .hero-banner p {
        color: #eaf2ff !important;
        margin: 0;
        font-size: 1.05rem;
    }

    /* CHANGED: All app cards converted into glassmorphism dashboard cards */
    .card {
        background: linear-gradient(145deg, rgba(22, 39, 70, 0.90), rgba(10, 21, 39, 0.86));
        border: 1px solid var(--border-soft);
        border-radius: 22px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-card);
        color: #e2e8f0;
        backdrop-filter: blur(18px);
    }

    /* CHANGED: Card hover effect added for premium feel */
    .card:hover {
        border-color: rgba(139, 108, 255, 0.38);
        box-shadow: 0 22px 55px rgba(0, 0, 0, 0.40);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }

    /* CHANGED: Card heading typography improved */
    .card h3 {
        color: #f8fafc !important;
        font-weight: 800;
        margin-top: 0;
    }

    /* CHANGED: Card text forced to readable color */
    .card p,
    .card strong,
    .card li {
        color: #dbe4f0 !important;
    }

    /* CHANGED: Schedule and timeline outer containers made premium glass boxes */
    .st-key-schedule_outer_card,
    .st-key-timeline_outer_card {
        background: rgba(10, 21, 39, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        padding: 1.35rem 1.5rem 1.6rem 1.5rem;
        box-shadow: var(--shadow-card);
        backdrop-filter: blur(18px);
    }

    /* CHANGED: Welcome heading styled like a dashboard header */
    .welcome-header h2 {
        margin-bottom: 0.1rem;
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff !important;
    }

    /* CHANGED: Welcome subtitle softened */
    .welcome-header p {
        color: var(--text-muted) !important;
        margin-top: 0;
    }

    /* CHANGED: Stat cards redesigned as premium summary tiles */
    .stat-card {
        background: linear-gradient(145deg, rgba(22, 39, 70, 0.92), rgba(13, 25, 45, 0.88));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        padding: 1.15rem 1.2rem;
        height: 100%;
        box-shadow: 0 15px 35px rgba(0,0,0,0.25);
        backdrop-filter: blur(14px);
    }

    /* CHANGED: Stat label color improved */
    .stat-card .stat-label {
        font-size: 0.82rem;
        color: #aebbd0;
        margin-bottom: 6px;
    }

    /* CHANGED: Stat value enlarged */
    .stat-card .stat-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
    }

    /* CHANGED: AI badge redesigned */
    .stat-card .stat-badge {
        display: inline-block;
        margin-top: 8px;
        font-size: 0.72rem;
        background: rgba(20, 184, 212, 0.16);
        color: #67e8f9;
        padding: 2px 9px;
        border-radius: 9999px;
        border: 1px solid rgba(103, 232, 249, 0.18);
    }

    /* CHANGED: Confidence progress bar track improved */
    .stat-progress-track {
        background-color: rgba(51, 65, 85, 0.7);
        border-radius: 9999px;
        height: 9px;
        margin-top: 10px;
        overflow: hidden;
    }

    /* CHANGED: Confidence progress fill uses purple-cyan gradient */
    .stat-progress-fill {
        background: linear-gradient(90deg, #6d4aff, #14b8d4);
        height: 100%;
        border-radius: 9999px;
    }

    /* CHANGED: All Streamlit buttons upgraded with rounded gradient style */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: linear-gradient(135deg, #6d4aff, #4f46e5) !important;
        color: white !important;
        font-weight: 800 !important;
        box-shadow: 0 12px 26px rgba(109, 74, 255, 0.25);
        transition: all 0.2s ease;
    }

    /* CHANGED: Button hover animation added */
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 34px rgba(109, 74, 255, 0.36);
    }

    /* CHANGED: File uploader converted into dark glass upload panel */
    [data-testid="stFileUploader"] {
        background: rgba(15, 30, 52, 0.75) !important;
        border: 1px dashed rgba(139, 108, 255, 0.45) !important;
        border-radius: 22px !important;
        padding: 1rem !important;
        box-shadow: var(--shadow-card);
    }

    /* CHANGED: Streamlit table layout improved for medicine schedule */
    table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
        overflow: hidden;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* CHANGED: Table header made purple-tinted */
    thead tr th {
        background: rgba(109, 74, 255, 0.18) !important;
        color: #f8fafc !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
        font-weight: 800 !important;
    }

    /* CHANGED: Table body rows made dark and readable */
    tbody tr td {
        background: rgba(15, 30, 52, 0.62) !important;
        color: #e5e7eb !important;
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    }

    /* CHANGED: Streamlit tabs redesigned as pill buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 30, 52, 0.64);
        padding: 0.45rem;
        border-radius: 18px;
    }

    /* CHANGED: Tab text made bolder */
    .stTabs [data-baseweb="tab"] {
        border-radius: 14px;
        color: #cbd5e1;
        font-weight: 800;
    }

    /* CHANGED: Active tab uses purple gradient */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6d4aff, #4f46e5);
        color: white !important;
    }

    /* CHANGED: Input fields made dark and rounded */
    input,
    textarea,
    select {
        background: rgba(15, 30, 52, 0.95) !important;
        color: #f8fafc !important;
        border-radius: 14px !important;
    }

    /* CHANGED: Selectbox/text-input inner text color fixed */
    [data-testid="stSelectbox"] div,
    [data-testid="stTextInput"] div,
    [data-testid="stTextArea"] div {
        color: #f8fafc !important;
    }

    /* CHANGED: Alerts made consistent with dark glass UI */
    .stAlert {
        background: rgba(15, 30, 52, 0.80) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        color: #f8fafc !important;
    }

    /* CHANGED: Chat panel redesigned to match the right-side assistant in target image */
    .st-key-chat_panel {
        position: sticky;
        top: 1rem;
        align-self: flex-start;
        background: linear-gradient(180deg, rgba(37, 32, 102, 0.98), rgba(10, 20, 38, 0.96));
        border: 1px solid rgba(139, 92, 246, 0.60);
        border-radius: 24px;
        padding: 1rem;
        max-height: 650px;
        overflow-y: auto;
        box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.25), 0 22px 55px rgba(0, 0, 0, 0.42);
        backdrop-filter: blur(18px);
    }

    /* CHANGED: Chat close button made minimal */
    .st-key-chat_close_wrap .stButton > button,
    .st-key-chat_clear_wrap .stButton > button {
        background: transparent !important;
        border: none !important;
        color: #cbd5e1 !important;
        font-size: 1rem;
        padding: 0.1rem 0.4rem;
        min-height: auto;
        float: right;
        box-shadow: none;
    }

    /* CHANGED: Chat close hover kept clean */
    .st-key-chat_close_wrap .stButton > button:hover,
    .st-key-chat_clear_wrap .stButton > button:hover {
        color: #ffffff !important;
        background: transparent !important;
        transform: none;
    }

    /* CHANGED: Chat header upgraded */
    .chat-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 0.9rem;
    }

    /* CHANGED: Chat avatar redesigned with cyan-purple gradient */
    .chat-avatar,
    .chat-avatar-sm {
        background: linear-gradient(135deg, #14b8d4, #6d4aff);
        box-shadow: 0 0 18px rgba(20, 184, 212, 0.35);
    }

    /* CHANGED: Main chat avatar sizing restored after redesign */
    .chat-avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }

    /* CHANGED: Small assistant bubble avatar sizing restored after redesign */
    .chat-avatar-sm {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    /* Custom avatar image (assets/chatbot/avatar.*) fills the circle
       regardless of the source file's own aspect ratio/background. */
    .chat-avatar-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
    }

    /* CHANGED: Chat title made bolder */
    .chat-header-title {
        font-weight: 800;
        color: #ffffff;
    }

    /* CHANGED: Chat subtitle color softened */
    .chat-header-subtitle {
        font-size: 0.76rem;
        color: #cbd5e1;
    }

    /* CHANGED: Beta badge made brighter */
    .beta-badge {
        background-color: #16a34a;
        color: #dcfce7;
        font-size: 0.66rem;
        padding: 1px 7px;
        border-radius: 9999px;
        margin-left: 5px;
        vertical-align: middle;
        font-weight: 800;
    }

    /* CHANGED: Chat suggestion buttons redesigned as gold outline pills */
    .st-key-chat_suggestions .stButton > button {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(251, 191, 36, 0.55) !important;
        color: #fde68a !important;
        border-radius: 9999px !important;
        font-size: 0.78rem;
        padding: 0.35rem 0.55rem;
        box-shadow: none;
    }

    /* CHANGED: Chat suggestion hover improved */
    .st-key-chat_suggestions .stButton > button:hover {
        background: rgba(251, 191, 36, 0.12) !important;
        color: #fff7cc !important;
        transform: translateY(-1px);
    }

    /* CHANGED: Chat bubble row spacing improved */
    .bubble-row {
        display: flex;
        gap: 8px;
        margin-bottom: 14px;
        align-items: flex-end;
    }

    /* CHANGED: User bubble aligned right */
    .bubble-row-user {
        justify-content: flex-end;
    }

    /* CHANGED: Bubble width improved */
    .bubble-col {
        display: flex;
        flex-direction: column;
        max-width: 80%;
    }

    /* CHANGED: User bubble column aligned right */
    .bubble-row-user .bubble-col {
        align-items: flex-end;
    }

    /* CHANGED: Bubble typography and shape improved */
    .bubble {
        padding: 0.65rem 0.9rem;
        border-radius: 16px;
        font-size: 0.88rem;
        line-height: 1.55;
        text-align: left;
    }

    /* CHANGED: Assistant bubble made glassy */
    .bubble-assistant {
        background: rgba(22, 39, 70, 0.88);
        color: #e2e8f0;
        border-bottom-left-radius: 5px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* CHANGED: User bubble made purple gradient */
    .bubble-user {
        background: linear-gradient(135deg, #6d4aff, #4f46e5);
        color: white;
        border-bottom-right-radius: 5px;
    }

    /* CHANGED: Bubble timestamp softened */
    .bubble-time {
        font-size: 0.66rem;
        color: #94a3b8;
        margin-top: 3px;
        padding: 0 4px;
    }

    /* CHANGED: User timestamp color improved */
    .bubble-time-user {
        color: #c4b5fd;
    }

    /* CHANGED: Footer made cleaner */
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.86rem;
        padding: 1.2rem 0 0.7rem 0;
    }

    /* CHANGED: Dividers made softer */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* CHANGED: Scrollbar styled with purple thumb */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    /* CHANGED: Scrollbar track darkened */
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }

    /* CHANGED: Scrollbar thumb colored */
    ::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.75);
        border-radius: 999px;
    }
    /* CHANGED: Premium routine timeline cards for AI চিকিৎসার সময়রেখা section */
.routine-slot-card {
    position: relative;
    min-height: 245px;
    background: linear-gradient(145deg, rgba(22, 39, 70, 0.95), rgba(10, 21, 39, 0.92));
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 22px;
    padding: 1.25rem 1.25rem 4.5rem 1.25rem;
    overflow: hidden;
    box-shadow: 0 18px 40px rgba(0,0,0,0.30);
}

.routine-slot-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.25rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}

.routine-slot-icon {
    font-size: 1.45rem;
}

.routine-slot-subtitle {
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-bottom: 0.8rem;
}

.routine-slot-list {
    margin: 0;
    padding-left: 1.1rem;
    position: relative;
    z-index: 2;
}

.routine-slot-list li {
    margin-bottom: 0.65rem;
    color: #dbe4f0;
    line-height: 1.45;
}

.routine-slot-list li strong {
    color: #ffffff;
}

.routine-slot-list li span {
    color: #b6c2d6;
    font-size: 0.88rem;
}

.empty-med {
    color: #94a3b8 !important;
}

.ongoing-badge {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 7px;
    border-radius: 999px;
    background: rgba(20, 184, 212, 0.16);
    color: #67e8f9 !important;
    border: 1px solid rgba(103, 232, 249, 0.22);
    font-size: 0.72rem !important;
}

/* CHANGED: Small scenic bottom area like the reference picture */
.routine-landscape {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 70px;
    opacity: 0.85;
    z-index: 1;
}

.morning-card .routine-landscape {
    background:
        radial-gradient(circle at 22% 42%, #fbbf24 0 14px, transparent 15px),
        radial-gradient(circle at 20% 100%, #22c55e 0 42px, transparent 43px),
        radial-gradient(circle at 45% 105%, #16a34a 0 46px, transparent 47px),
        linear-gradient(180deg, transparent 0%, rgba(34,197,94,0.22) 100%);
}

.noon-card .routine-landscape {
    background:
        radial-gradient(circle at 25% 100%, #22c55e 0 42px, transparent 43px),
        radial-gradient(circle at 55% 105%, #16a34a 0 46px, transparent 47px),
        radial-gradient(circle at 75% 100%, #15803d 0 40px, transparent 41px),
        linear-gradient(180deg, transparent 0%, rgba(34,197,94,0.18) 100%);
}

.night-card .routine-landscape {
    background:
        radial-gradient(circle at 72% 35%, #e0e7ff 0 11px, transparent 12px),
        radial-gradient(circle at 25% 105%, #312e81 0 45px, transparent 46px),
        radial-gradient(circle at 55% 110%, #4338ca 0 50px, transparent 51px),
        linear-gradient(180deg, transparent 0%, rgba(79,70,229,0.22) 100%);
}
/* CHANGED: Light mode chat panel text visibility fix only */
/* Keeps dark mode unchanged and does NOT change main button colors */

.theme-light .st-key-chat_panel,
body[data-theme="light"] .st-key-chat_panel,
.stApp.theme-light .st-key-chat_panel {
    background: #ffffff !important;
    border: 1px solid #dbe3ef !important;
}

/* Header text */
.theme-light .chat-header-title,
.theme-light .chat-header-subtitle,
body[data-theme="light"] .chat-header-title,
body[data-theme="light"] .chat-header-subtitle,
.stApp.theme-light .chat-header-title,
.stApp.theme-light .chat-header-subtitle {
    color: #0f172a !important;
}

/* Assistant/user bubble text */
.theme-light .bubble,
.theme-light .bubble p,
.theme-light .bubble span,
.theme-light .bubble div,
body[data-theme="light"] .bubble,
body[data-theme="light"] .bubble p,
body[data-theme="light"] .bubble span,
body[data-theme="light"] .bubble div,
.stApp.theme-light .bubble,
.stApp.theme-light .bubble p,
.stApp.theme-light .bubble span,
.stApp.theme-light .bubble div {
    color: #0f172a !important;
}

/* Assistant bubble background + text */
.theme-light .bubble-assistant,
body[data-theme="light"] .bubble-assistant,
.stApp.theme-light .bubble-assistant {
    background: #eef4fb !important;
    border: 1px solid #d5e2f1 !important;
    color: #0f172a !important;
}

/* User bubble stays blue, text must stay white */
.theme-light .bubble-user,
body[data-theme="light"] .bubble-user,
.stApp.theme-light .bubble-user {
    color: #ffffff !important;
}

/* Time text */
.theme-light .bubble-time,
body[data-theme="light"] .bubble-time,
.stApp.theme-light .bubble-time {
    color: #64748b !important;
}

.theme-light .bubble-time-user,
body[data-theme="light"] .bubble-time-user,
.stApp.theme-light .bubble-time-user {
    color: #94a3b8 !important;
}

/* Suggested question buttons */
.theme-light .st-key-chat_suggestions .stButton > button,
body[data-theme="light"] .st-key-chat_suggestions .stButton > button,
.stApp.theme-light .st-key-chat_suggestions .stButton > button {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e7c76a !important;
}

/* Chat input box */
.theme-light .st-key-chat_panel input,
.theme-light .st-key-chat_panel textarea,
body[data-theme="light"] .st-key-chat_panel input,
body[data-theme="light"] .st-key-chat_panel textarea,
.stApp.theme-light .st-key-chat_panel input,
.stApp.theme-light .st-key-chat_panel textarea {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cfd9e6 !important;
}

/* Placeholder text */
.theme-light .st-key-chat_panel input::placeholder,
.theme-light .st-key-chat_panel textarea::placeholder,
body[data-theme="light"] .st-key-chat_panel input::placeholder,
body[data-theme="light"] .st-key-chat_panel textarea::placeholder,
.stApp.theme-light .st-key-chat_panel input::placeholder,
.stApp.theme-light .st-key-chat_panel textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* Small labels inside chat area */
.theme-light .st-key-chat_panel label,
.theme-light .st-key-chat_panel small,
.theme-light .st-key-chat_panel p,
.theme-light .st-key-chat_panel span,
body[data-theme="light"] .st-key-chat_panel label,
body[data-theme="light"] .st-key-chat_panel small,
body[data-theme="light"] .st-key-chat_panel p,
body[data-theme="light"] .st-key-chat_panel span,
.stApp.theme-light .st-key-chat_panel label,
.stApp.theme-light .st-key-chat_panel small,
.stApp.theme-light .st-key-chat_panel p,
.stApp.theme-light .st-key-chat_panel span {
    color: #0f172a !important;
}
    </style>
    """
)


# ============================== CHATBOT FAB: POSITION + ANIMATION ==============================
# Fixed to the viewport (not the page/column flow), so it stays put on scroll
# and only ever moves vertically - CHATBOT_BOTTOM_PX (top of file) is the one
# knob to change. Horizontal position is not exposed on purpose: --chatbot-right
# always pins the icon to the right edge.
render_html(
    f"""
    <style>
    :root {{
        --chatbot-right: {CHATBOT_RIGHT_PX}px;
        --chatbot-bottom: {CHATBOT_BOTTOM_PX}px;
        --chatbot-size: {CHATBOT_SIZE_DESKTOP_PX}px;
    }}
    @media (max-width: 640px) {{
        :root {{ --chatbot-size: {CHATBOT_SIZE_MOBILE_PX}px; }}
    }}

    .st-key-chat_fab {{
        position: fixed !important;
        right: var(--chatbot-right) !important;
        bottom: var(--chatbot-bottom) !important;
        top: auto !important;
        left: auto !important;
        z-index: 10000;
        width: auto !important;
        display: block !important;
    }}

    .st-key-chat_fab .stButton > button {{
        width: var(--chatbot-size) !important;
        height: var(--chatbot-size) !important;
        border-radius: 50% !important;
        padding: 0 !important;
        font-size: calc(var(--chatbot-size) * 0.42);
        line-height: 1 !important;
        background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
        border: none !important;
        box-shadow: 0 10px 26px rgba(37, 99, 235, 0.45);
        animation: chatbotFloat 3.2s ease-in-out infinite;
        transition: box-shadow 300ms ease, transform 300ms ease;
    }}

    /* Idle float + gentle "breathing" scale, subtle on purpose (not distracting) */
    @keyframes chatbotFloat {{
        0%, 100% {{ transform: translateY(0) scale(1); }}
        50% {{ transform: translateY(-8px) scale(1.03); }}
    }}

    .st-key-chat_fab .stButton > button:hover {{
        animation-play-state: paused;
        transform: scale(1.08);
        box-shadow: 0 14px 34px rgba(37, 99, 235, 0.65);
    }}

    .st-key-chat_fab .stButton > button:active {{
        animation: chatbotBounce 300ms ease !important;
    }}

    @keyframes chatbotBounce {{
        0% {{ transform: scale(1); }}
        35% {{ transform: scale(0.86); }}
        65% {{ transform: scale(1.1); }}
        100% {{ transform: scale(1); }}
    }}

    @media (prefers-reduced-motion: reduce) {{
        .st-key-chat_fab .stButton > button {{
            animation: none !important;
        }}
    }}
    </style>
    """
)


class AllKeysExhaustedError(Exception):
    """Raised when every configured Gemini API key hit a quota/permission wall."""


def _load_gemini_keys() -> list:
    """Collect Gemini API keys in fallback priority order.

    Reads GEMINI_API_KEY_1..9 (tried in that order) followed by a plain
    GEMINI_API_KEY, so a multi-key setup (HF Secrets / .env) and the old
    single-key setup both keep working. Duplicates removed, order preserved.
    """
    keys = []
    for i in range(1, 10):
        k = os.getenv(f"GEMINI_API_KEY_{i}")
        if k and k.strip():
            keys.append(k.strip())
    plain = os.getenv("GEMINI_API_KEY")
    if plain and plain.strip():
        keys.append(plain.strip())
    seen, ordered = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


def _is_quota_error(exc) -> bool:
    """True for API errors worth failing over to the next key for: a rate limit,
    an exhausted daily quota, or a project whose access was denied."""
    code = getattr(exc, "code", None)
    if code in (429, 403):
        return True
    msg = str(exc).lower()
    return any(
        s in msg for s in
        ("resource_exhausted", "quota", "rate limit", "permission_denied", "429", "403")
    )


class _FallbackModels:
    """Mimics client.models, but generate_content fails over across keys."""

    def __init__(self, clients):
        self._clients = clients

    def generate_content(self, **kwargs):
        last_exc = None
        for client in self._clients:
            try:
                return client.models.generate_content(**kwargs)
            except genai_errors.APIError as exc:
                if _is_quota_error(exc):
                    last_exc = exc
                    continue  # this key is rate-limited/denied - try the next one
                raise  # a different API error (bad request, etc.) - surface as-is
        raise AllKeysExhaustedError(str(last_exc) if last_exc else "no keys configured")


class FallbackGeminiClient:
    """Drop-in for genai.Client that tries each API key in order on quota errors.

    Exposes only the `.models.generate_content(...)` surface the app uses, so
    every existing call site keeps working unchanged. Fallback (not round-robin):
    always start at key #1, move to #2, #3 only when one fails - predictable and
    easy to debug.
    """

    def __init__(self, keys):
        self._clients = [genai.Client(api_key=k) for k in keys]
        self.models = _FallbackModels(self._clients)


def get_gemini_client():
    keys = _load_gemini_keys()
    if not keys:
        return None
    return FallbackGeminiClient(keys)


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


def prepare_test_report_part(uploaded_file) -> types.Part:
    """Build a Gemini Part from an uploaded test report file (image or PDF).

    Gemini 2.5 models natively understand PDF documents (including multi-page
    ones), so PDFs are sent as-is; images go through the same resize/compress
    step as prescription photos to keep the request small.
    """
    file_bytes = uploaded_file.getvalue()
    mime_type = (uploaded_file.type or "").lower()
    is_pdf = mime_type == "application/pdf" or (uploaded_file.name or "").lower().endswith(".pdf")
    if is_pdf:
        return types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return prepare_image_part(image)


def explain_prescription(client: genai.Client, image: Image.Image, content_language: str = "Bangla") -> dict:
    system_prompt = (
        "You are a strict medical-document classifier AND a medical assistant that reads "
        "doctor's prescriptions and explains them to patients in simple, easy-to-understand "
        f"{content_language}. You must always respond ONLY with valid JSON, no extra text.\n"
        "STEP 1 - CLASSIFY the image first. Decide what the image actually is:\n"
        "  - \"prescription\": a doctor's medical order (handwritten or printed) - typically "
        "has an Rx symbol, a list of medicines with dosage/timing instructions, and usually a "
        "doctor/chamber/hospital letterhead, a patient name, and a date.\n"
        "  - \"test_report\": a lab/diagnostic report (blood, urine, imaging, etc.).\n"
        "  - \"medicine_package\" / \"medicine_strip\" / \"pharmacy_label\": a MEDICINE PRODUCT "
        "photo (box, blister strip, bottle, or a pharmacy sticker) showing brand/generic name, "
        "manufacturer, batch, or expiry - this is NOT a prescription.\n"
        "  - \"hospital_bill\" / \"invoice\": a bill or money receipt.\n"
        "  - \"non_medical\": anything unrelated to healthcare.  - \"unknown\": cannot tell.\n"
        "A photo of a medicine box/strip/label is NEVER a prescription, even though it shows a "
        "medicine name - do not treat it as one.\n"
        "The JSON must be a single object with exactly these top-level keys: \"image_category\", "
        "\"is_prescription\", \"classification_confidence\", \"extraction_confidence\", "
        "\"classification_reason\", \"overall_summary\", \"overall_confidence\", "
        "\"patient_info\", \"doctor_info\", \"prescription_date\", \"follow_up_date\", "
        "\"probable_condition\", \"detected_conditions\", \"treatment_purpose\", "
        "\"drug_interactions\", \"medicines\".\n"
        "- \"image_category\": exactly one of \"prescription\", \"test_report\", "
        "\"medicine_package\", \"medicine_strip\", \"pharmacy_label\", \"hospital_bill\", "
        "\"invoice\", \"non_medical\", \"unknown\" (English lowercase).\n"
        "- \"is_prescription\": boolean - true ONLY if image_category is \"prescription\".\n"
        "- \"classification_confidence\": a number 0 to 1 - how sure you are of image_category.\n"
        "- \"extraction_confidence\": a number 0 to 1 - how legible/readable the prescription is "
        "(use 0 when it is not a prescription).\n"
        f"- \"classification_reason\": one short {content_language} phrase naming what the image "
        "looks like (e.g. a medicine box, a bill), shown to the user when it is not a "
        "prescription.\n"
        "STEP 2 - Only if image_category is \"prescription\", extract the details below. If it "
        "is NOT a prescription, set is_prescription=false and extraction_confidence=0 and leave "
        "ALL medical fields empty (empty strings, empty lists, empty objects) - do NOT invent or "
        "guess any doctor, patient, date, or medicine data.\n"
        f"- \"overall_summary\": a 2-3 sentence plain-{content_language} summary of the whole "
        "prescription.\n"
        "- \"overall_confidence\": one of \"high\", \"medium\", or \"low\" (English, "
        "lowercase), reflecting your overall confidence in reading the entire image.\n"
        "- \"patient_info\": an object with exactly these keys: \"name\", \"age\", and "
        "\"gender\". \"name\" and \"age\" in simple "
        f"{content_language} if visible on the prescription, otherwise a short phrase meaning "
        "\"not identified\". \"gender\" must be exactly \"male\", \"female\", or \"unknown\" "
        "(English lowercase).\n"
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
        "\"medicine_name\", \"purpose\", \"dosage\", \"duration\", \"duration_days\", "
        "\"side_effects\", \"precautions\", \"confidence\", \"timing\", \"high_risk\".\n"
        "\"duration\" is how long to take the medicine (e.g. \"5 days\", \"7 days\", "
        f"\"1 month\"), in simple {content_language}; if not specified on the prescription, "
        "say so clearly in the target language.\n"
        "\"duration_days\" is the same duration converted to a plain integer number of days "
        "(e.g. \"5 days\" -> 5, \"1 week\" -> 7, \"1 month\" -> 30); use JSON null if the "
        "duration is not specified or is open-ended (e.g. \"continue\", \"as needed\").\n"
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
        "Below is an uploaded image. First classify what it is, then - only if it is a "
        f"doctor's prescription - analyze it fully and respond in simple {content_language} as "
        "instructed. If it is not a prescription, return the classification fields with empty "
        "medical fields."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prepare_image_part(image), user_prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
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

    image_category = str(data.get("image_category") or "unknown").strip().lower()
    if image_category not in IMAGE_CATEGORIES:
        image_category = "unknown"
    is_prescription = as_bool(data.get("is_prescription")) and image_category == "prescription"

    medicines = data.get("medicines")
    if not isinstance(medicines, list):
        medicines = []
    medicines = [med for med in medicines if isinstance(med, dict)]

    patient_info_raw = data.get("patient_info")
    if not isinstance(patient_info_raw, dict):
        patient_info_raw = {}
    age_raw = patient_info_raw.get("age")
    gender_raw = str(patient_info_raw.get("gender") or "unknown").strip().lower()
    patient_info = {
        "name": str(patient_info_raw.get("name") or "").strip(),
        "age": ("" if age_raw is None else str(age_raw)).strip(),
        "gender": gender_raw if gender_raw in ("male", "female", "unknown") else "unknown",
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
        "image_category": image_category,
        "is_prescription": is_prescription,
        "classification_confidence": as_confidence(data.get("classification_confidence")),
        "extraction_confidence": as_confidence(data.get("extraction_confidence")),
        "classification_reason": str(data.get("classification_reason") or "").strip(),
        "overall_summary": str(data.get("overall_summary") or "").strip(),
        "overall_confidence": str(data.get("overall_confidence") or "low").strip().lower(),
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


def explain_test_report(
    client: genai.Client, file_part: types.Part, content_language: str = "Bangla"
) -> dict:
    system_prompt = (
        "You are a strict medical-document classifier AND a medical assistant that reads "
        "medical test/lab reports (blood tests, urine tests, imaging reports, etc.) and "
        f"explains them to patients in simple, easy-to-understand {content_language}. You must "
        "always respond ONLY with valid JSON, no extra text.\n"
        "STEP 1 - CLASSIFY the document first. \"test_report\" = a lab/diagnostic report with "
        "test names, values, and reference ranges. A doctor's prescription, a medicine "
        "box/strip/label, a bill/invoice, or a non-medical image is NOT a test report.\n"
        "The JSON must be a single object with exactly these top-level keys: \"image_category\", "
        "\"is_test_report\", \"classification_confidence\", \"extraction_confidence\", "
        "\"classification_reason\", \"overall_summary\", \"overall_confidence\", "
        "\"patient_info\", \"lab_info\", \"report_date\", \"tests\".\n"
        "- \"image_category\": exactly one of \"prescription\", \"test_report\", "
        "\"medicine_package\", \"medicine_strip\", \"pharmacy_label\", \"hospital_bill\", "
        "\"invoice\", \"non_medical\", \"unknown\" (English lowercase).\n"
        "- \"is_test_report\": boolean - true ONLY if image_category is \"test_report\".\n"
        "- \"classification_confidence\": a number 0 to 1 - how sure you are of image_category.\n"
        "- \"extraction_confidence\": a number 0 to 1 - how legible/readable the report is "
        "(use 0 when it is not a test report).\n"
        f"- \"classification_reason\": one short {content_language} phrase naming what the "
        "document looks like, shown to the user when it is not a test report.\n"
        "STEP 2 - Only if image_category is \"test_report\", extract the details below. If it is "
        "NOT a test report, set is_test_report=false and extraction_confidence=0 and leave ALL "
        "medical fields empty (empty strings, empty lists, empty objects) - do NOT invent data.\n"
        f"- \"overall_summary\": a 2-3 sentence plain-{content_language} summary of the whole "
        "report, mentioning whether results are mostly normal or if anything stands out.\n"
        "- \"overall_confidence\": one of \"high\", \"medium\", or \"low\" (English, "
        "lowercase), reflecting your overall confidence in reading the entire document.\n"
        "- \"patient_info\": an object with exactly these keys: \"name\", \"age\", and "
        "\"gender\". \"name\" and \"age\" in simple "
        f"{content_language} if visible, otherwise a short phrase meaning \"not identified\". "
        "\"gender\" must be exactly \"male\", \"female\", or \"unknown\" (English lowercase).\n"
        "- \"lab_info\": an object with exactly these keys: \"name\" (lab or hospital name) "
        f"and \"referring_doctor\", each in simple {content_language} if visible, otherwise a "
        "short phrase meaning \"not identified\", written in the target language.\n"
        "- \"report_date\": the date of the report, if visible; otherwise a short phrase "
        "meaning \"not mentioned\", written in the target language.\n"
        "- \"tests\": a list of objects, one per test/measurement found in the report, each "
        "with these exact keys: \"test_name\", \"value\", \"unit\", \"reference_range\", "
        "\"status\", \"purpose\", \"explanation\". \"value\", \"unit\", and "
        "\"reference_range\" should be written as they appear on the report (numbers/units "
        "can stay as-is). \"status\" must be exactly one of \"normal\", \"borderline\", or "
        "\"abnormal\" (English, lowercase), based on comparing the value to the reference "
        f"range. \"purpose\" is one short {content_language} sentence explaining why this "
        f"test is commonly done. \"explanation\" is one or two {content_language} sentences "
        "explaining in plain language what this specific result means for the patient - if "
        "the value is borderline or abnormal, explain what that can possibly indicate in "
        "general terms, but do NOT state or imply a specific diagnosis; recommend discussing "
        "the result with a doctor instead.\n"
        f"All text values must be written in simple {content_language}, understandable by "
        "someone with no medical background."
    )

    user_prompt = (
        "Below is an uploaded document (image or PDF). First classify what it is, then - only "
        f"if it is a medical test/lab report - analyze it fully and respond in simple "
        f"{content_language} as instructed. If it is not a test report, return the "
        "classification fields with empty medical fields."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[file_part, user_prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )

    content = response.text.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    data = json.loads(content)
    if isinstance(data, list):
        data = {"tests": data}
    elif not isinstance(data, dict):
        data = {}

    image_category = str(data.get("image_category") or "unknown").strip().lower()
    if image_category not in IMAGE_CATEGORIES:
        image_category = "unknown"
    is_test_report = as_bool(data.get("is_test_report")) and image_category == "test_report"

    tests = data.get("tests")
    if not isinstance(tests, list):
        tests = []
    tests = [test for test in tests if isinstance(test, dict)]

    patient_info_raw = data.get("patient_info")
    if not isinstance(patient_info_raw, dict):
        patient_info_raw = {}
    age_raw = patient_info_raw.get("age")
    gender_raw = str(patient_info_raw.get("gender") or "unknown").strip().lower()
    patient_info = {
        "name": str(patient_info_raw.get("name") or "").strip(),
        "age": ("" if age_raw is None else str(age_raw)).strip(),
        "gender": gender_raw if gender_raw in ("male", "female", "unknown") else "unknown",
    }

    lab_info_raw = data.get("lab_info")
    if not isinstance(lab_info_raw, dict):
        lab_info_raw = {}
    lab_info = {
        "name": str(lab_info_raw.get("name") or "").strip(),
        "referring_doctor": str(lab_info_raw.get("referring_doctor") or "").strip(),
    }

    return {
        "image_category": image_category,
        "is_test_report": is_test_report,
        "classification_confidence": as_confidence(data.get("classification_confidence")),
        "extraction_confidence": as_confidence(data.get("extraction_confidence")),
        "classification_reason": str(data.get("classification_reason") or "").strip(),
        "overall_summary": str(data.get("overall_summary") or "").strip(),
        "overall_confidence": str(data.get("overall_confidence") or "low").strip().lower(),
        "patient_info": patient_info,
        "lab_info": lab_info,
        "report_date": str(data.get("report_date") or "").strip(),
        "tests": tests,
    }


def compare_reports(
    client: genai.Client, prescription_result: dict, test_result: dict, content_language: str
) -> dict:
    """Check that a prescription and a test report belong to the same patient, and
    only then whether the medicines are consistent with the results.

    Returns a structured dict so the UI can show MATCH / PARTIAL_MATCH / MISMATCH
    and refuse to draw medical conclusions when the two documents look like they
    belong to different patients.
    """
    p_info = prescription_result.get("patient_info") or {}
    r_info = test_result.get("patient_info") or {}
    presc_meds = ", ".join(
        str(m.get("medicine_name", "")).strip()
        for m in prescription_result.get("medicines", []) if isinstance(m, dict)
    ) or "none"
    report_tests = ", ".join(
        str(t.get("test_name", "")).strip()
        for t in test_result.get("tests", []) if isinstance(t, dict)
    ) or "none"

    presc_context = (
        f"Patient name: {p_info.get('name') or 'not stated'}\n"
        f"Age: {p_info.get('age') or 'not stated'}\n"
        f"Gender: {p_info.get('gender') or 'unknown'}\n"
        f"Prescription date: {prescription_result.get('prescription_date') or 'not stated'}\n"
        f"Probable condition: {prescription_result.get('probable_condition') or 'not stated'}\n"
        f"Medicines: {presc_meds}"
    )
    report_context = (
        f"Patient name: {r_info.get('name') or 'not stated'}\n"
        f"Age: {r_info.get('age') or 'not stated'}\n"
        f"Gender: {r_info.get('gender') or 'unknown'}\n"
        f"Report date: {test_result.get('report_date') or 'not stated'}\n"
        f"Report summary: {test_result.get('overall_summary') or 'not stated'}\n"
        f"Tests: {report_tests}"
    )

    system_prompt = (
        "You are a careful medical safety checker. You are given identity + medical details "
        "extracted from a patient's PRESCRIPTION and, separately, from a TEST REPORT. Decide "
        "whether the two documents plausibly belong to the SAME patient, and only then whether "
        "the prescribed medicines are consistent with the test results. Respond ONLY with valid "
        "JSON: a single object with exactly these keys: \"match_status\", "
        "\"patient_match_score\", \"matched\", \"mismatched\", \"comparison\".\n"
        "STEP 1 - Verify identity: compare patient name similarity, age consistency, gender "
        "consistency, and date proximity. Set \"patient_match_score\" to an integer 0-100 for "
        "how confident you are they are the same patient. A clear age gap (e.g. a young child "
        "vs an elderly adult), opposite gender, or a clearly different name is a strong "
        "MISMATCH signal and must lower the score well below 50.\n"
        "- \"match_status\": \"MATCH\" if identity is clearly consistent; \"PARTIAL_MATCH\" if "
        "some fields match but others are missing/unverifiable; \"MISMATCH\" if there is a "
        f"clear identity conflict OR patient_match_score is below {PATIENT_MATCH_MIN}.\n"
        f"- \"matched\": a list of short {content_language} phrases for what is consistent "
        "(empty list if none).\n"
        f"- \"mismatched\": a list of short {content_language} phrases for each inconsistency or "
        "unverifiable point (empty list if none).\n"
        f"- \"comparison\": if match_status is MATCH or PARTIAL_MATCH, a 3-6 sentence plain-"
        f"{content_language} explanation of whether the medicines are consistent with the test "
        "results (no diagnosis; advise confirming with a doctor). If match_status is MISMATCH, "
        "set \"comparison\" to an empty string and do NOT draw any medical conclusion.\n"
        "Never invent a medical conclusion when identity does not match."
    )
    user_prompt = f"PRESCRIPTION:\n{presc_context}\n\nTEST REPORT:\n{report_context}"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
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
    if not isinstance(data, dict):
        data = {}

    status = str(data.get("match_status") or "").strip().upper().replace(" ", "_")
    if status not in ("MATCH", "PARTIAL_MATCH", "MISMATCH"):
        status = "PARTIAL_MATCH"
    try:
        score = int(float(data.get("patient_match_score")))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    matched_raw = data.get("matched")
    matched = [str(x).strip() for x in matched_raw if str(x).strip()] if isinstance(matched_raw, list) else []
    mismatched_raw = data.get("mismatched")
    mismatched = [str(x).strip() for x in mismatched_raw if str(x).strip()] if isinstance(mismatched_raw, list) else []
    comparison = str(data.get("comparison") or "").strip()

    # Enforce the threshold on our side too - never trust the model alone to gate
    # a medical conclusion. Below the score floor it is always a MISMATCH with no
    # comparison text, regardless of what the model labelled it.
    if score < PATIENT_MATCH_MIN:
        status = "MISMATCH"
    if status == "MISMATCH":
        comparison = ""

    return {
        "match_status": status,
        "patient_match_score": score,
        "matched": matched,
        "mismatched": mismatched,
        "comparison": comparison,
    }


def build_comparison_card_html(comp: dict, T: dict) -> str:
    """Render the prescription-vs-report consistency result INSIDE the existing
    .card component (no new UI components). Shows the MATCH/PARTIAL/MISMATCH
    verdict, the patient match score, what matched/didn't, and - only when the
    documents plausibly belong to the same patient - the medical comparison.
    """
    status = comp.get("match_status", "PARTIAL_MATCH")
    score = int(comp.get("patient_match_score", 0) or 0)
    matched = comp.get("matched") or []
    mismatched = comp.get("mismatched") or []
    comparison = comp.get("comparison") or ""

    if status == "MATCH":
        status_label = T["compare_status_match"]
    elif status == "MISMATCH":
        status_label = T["compare_status_mismatch"]
    else:
        status_label = T["compare_status_partial"]

    def _list_block(label: str, items: list) -> str:
        if not items:
            return ""
        lis = "".join(f"<li>{html.escape(str(i))}</li>" for i in items)
        return (
            f"<p style='margin:0.5rem 0 0.2rem 0;'><strong>{html.escape(label)}:</strong></p>"
            f"<ul style='margin:0 0 0.4rem 1.1rem;'>{lis}</ul>"
        )

    if status == "MISMATCH":
        warning_html = (
            f"<div style='background:{DANGER_COLORS['bg']}; border:1px solid {DANGER_COLORS['border']}; "
            f"border-left:6px solid {DANGER_COLORS['border']}; border-radius:12px; padding:0.85rem 1rem; "
            f"margin:0.6rem 0; color:{DANGER_COLORS['text']} !important;'>"
            f"<span style='color:{DANGER_COLORS['text']} !important;'>{html.escape(T['compare_mismatch_warning'])}</span></div>"
        )
        body_html = warning_html + _list_block(T["compare_mismatched_label"], mismatched)
        body_html += f"<p style='opacity:0.85;'>{html.escape(T['compare_no_medical_on_mismatch'])}</p>"
    else:
        body_html = _list_block(T["compare_matched_label"], matched)
        body_html += _list_block(T["compare_mismatched_label"], mismatched)
        if comparison:
            body_html += f"<p style='margin-top:0.5rem; line-height:1.65;'>{html.escape(comparison)}</p>"

    return (
        f"<div class='card'>"
        f"<h3 style='margin-top:0;'>{html.escape(status_label)} "
        f"<span style='font-size:0.8rem; font-weight:700; opacity:0.85;'>&middot; "
        f"{html.escape(T['compare_score_label'])}: {score}%</span></h3>"
        f"{body_html}"
        f"</div>"
    )


def build_test_report_text(result: dict, T: dict) -> str:
    """Build a compact plain-text summary of a test report analysis.

    Used as Gemini's context for the prescription-vs-test comparison prompt.
    """
    lines = [T["test_report_title"], "=" * len(T["test_report_title"]), ""]
    lines.append(result.get("overall_summary") or T["no_summary"])
    lines.append("")
    for test in result.get("tests", []):
        if not isinstance(test, dict):
            continue
        name = test.get("test_name") or T["unknown_test"]
        status = str(test.get("status", "")).strip().lower()
        lines.append(
            f"- {name}: {test.get('value', '')} {test.get('unit', '')} "
            f"({T['reference_range_label']}: {test.get('reference_range', '')}) - {status}"
        )
        if test.get("explanation"):
            lines.append(f"  {test['explanation']}")
    return "\n".join(lines)


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


def render_stat_card(label: str, value: str, badge: str = "", progress_pct=None) -> None:
    """Render one small dashboard-style stat card (used on the Summary page)."""
    progress_html = (
        f'<div class="stat-progress-track"><div class="stat-progress-fill" '
        f'style="width:{progress_pct}%;"></div></div>'
        if progress_pct is not None
        else ""
    )
    badge_html = f'<div class="stat-badge">{badge}</div>' if badge else ""
    render_html(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {progress_html}
            {badge_html}
        </div>
        """
    )


def _chat_now() -> str:
    """Current local time formatted like '11:45 AM', for chat bubble timestamps."""
    return datetime.now().strftime("%I:%M %p").lstrip("0")

#============================== CHAT BUBBLE RENDERING ==============================
def _chat_avatar_inner_html() -> str:
    """Custom robot <img> if assets/chatbot/avatar.* exists, else the 🤖 emoji.

    Shared by the chat panel header avatar and every assistant bubble avatar
    so both stay in sync automatically.
    """
    avatar_uri = load_chat_avatar_asset()
    if avatar_uri:
        return f'<img src="{avatar_uri}" alt="🤖" class="chat-avatar-img">'
    return "🤖"


def _render_chat_bubble(role: str, content: str, time_str: str) -> str:
    """Build one WhatsApp-style chat bubble row as an HTML fragment.

    FIXED: Inline critical bubble colors based on the current theme. This makes
    chat text readable even if Streamlit's internal CSS changes.
    """
    safe_content = html.escape(content).replace("\n", "<br>")
    is_light = st.session_state.get("app_theme", "Dark") == "Light"

    assistant_style = (
        "background:#f1f5f9; color:#0f172a; border:1px solid #dbe3ef;"
        if is_light
        else "background:rgba(30,41,59,0.96); color:#f8fafc; border:1px solid rgba(148,163,184,0.24);"
    )
    assistant_time_style = "color:#64748b;" if is_light else "color:#94a3b8;"
    user_style = (
        "background:linear-gradient(135deg,#2563eb,#0891b2); color:#ffffff; border:none;"
    )
    user_time_style = "color:rgba(255,255,255,0.82);"

    if role == "user":
        return f"""
        <div class="bubble-row bubble-row-user">
            <div class="bubble-col">
                <div class="bubble bubble-user" style="{user_style}">{safe_content}</div>
                <div class="bubble-time bubble-time-user" style="{user_time_style}">{time_str} ✓✓</div>
            </div>
        </div>
        """

    return f"""
    <div class="bubble-row bubble-row-assistant">
        <div class="chat-avatar-sm">{_chat_avatar_inner_html()}</div>
        <div class="bubble-col">
            <div class="bubble bubble-assistant" style="{assistant_style}">{safe_content}</div>
            <div class="bubble-time" style="{assistant_time_style}">{time_str}</div>
        </div>
    </div>
    """


def render_chat_panel(gemini_client, file_hash: str, language: str, T: dict) -> None:
    """Render the floating chat assistant widget as a fixed bottom-right overlay.

    Uses `chat_open` in session_state to toggle between the full panel and a
    small round launcher button, similar to a typical website live-chat widget.
    """
    if "chat_open" not in st.session_state:
        st.session_state["chat_open"] = True

    if not st.session_state["chat_open"]:
        with st.container(key="chat_fab"):
            asset_kind, asset_payload = load_chatbot_asset()
            if asset_kind == "image":
                # Swap the emoji for the custom animated icon: the button's own
                # background shows the GIF/WebP (browsers animate that natively,
                # no JS needed), and the emoji text is hidden rather than removed
                # so the button still has non-empty accessible label text.
                render_html(
                    f"""
                    <style>
                    .st-key-chat_fab .stButton > button {{
                        background-image: url('{asset_payload}') !important;
                        background-size: cover !important;
                        background-position: center !important;
                        background-repeat: no-repeat !important;
                        color: transparent !important;
                    }}
                    </style>
                    """
                )
            elif asset_kind == "lottie":
                # NOTE: Lottie needs a JS player (<lottie-player>/lottie-web),
                # and Streamlit's st.markdown (which render_html uses) strips
                # <script> tags for security, so it can't be wired up the same
                # simple way as image assets. A .json file here currently just
                # falls back to the emoji below. If you specifically need
                # Lottie, say so and it can be added via
                # st.components.v1.html (a sandboxed iframe that does allow
                # scripts) with extra positioning work to keep it locked to
                # this same fixed bottom-right spot.
                pass

            if st.button(" ", key="chat_fab_button"):
                st.session_state["chat_open"] = True
                st.rerun()
        return

    chat_scope_key = f"{file_hash}_{language}"
    chat_messages_key = f"chat_messages_{chat_scope_key}"
    chat_session_key = f"chat_session_{chat_scope_key}"
    if chat_messages_key not in st.session_state:
        st.session_state[chat_messages_key] = []

    # FIXED: Critical chat text colors are also set inline so Light/Dark mode
    # remains readable even when Streamlit changes internal class names.
    is_light_theme = st.session_state.get("app_theme", "Dark") == "Light"
    chat_title_style = "color:#0f172a;" if is_light_theme else "color:#ffffff;"
    chat_subtitle_style = "color:#475569;" if is_light_theme else "color:#cbd5e1;"

    with st.container(key="chat_panel"):
        header_col, clear_col, close_col = st.columns([6, 1, 1])
        with header_col:
            render_html(
                f"""
                <div class="chat-header">
                    <div class="chat-avatar">{_chat_avatar_inner_html()}</div>
                    <div>
                        <div class="chat-header-title" style="{chat_title_style}">{T['chat_panel_title']}
                            <span class="beta-badge">{T['chat_beta_badge']}</span></div>
                        <div class="chat-header-subtitle" style="{chat_subtitle_style}">{T['chat_panel_subtitle']}</div>
                    </div>
                </div>
                """
            )
        with clear_col:
            with st.container(key="chat_clear_wrap"):
                if st.button("↻", key="chat_clear_button", help="Clear chat"):
                    st.session_state[chat_messages_key] = []
                    if chat_session_key in st.session_state:
                        del st.session_state[chat_session_key]
                    st.rerun()
        with close_col:
            with st.container(key="chat_close_wrap"):
                if st.button("✕", key="chat_close_button"):
                    st.session_state["chat_open"] = False
                    st.rerun()

        bubbles_html = []
        if not st.session_state[chat_messages_key]:
            bubbles_html.append(
                _render_chat_bubble("assistant", T["chat_welcome_message"], _chat_now())
            )
        for msg in st.session_state[chat_messages_key]:
            bubbles_html.append(
                _render_chat_bubble(msg["role"], msg["content"], msg.get("time", ""))
            )
        with st.container(key="chat_messages"):
            render_html(f"<div>{''.join(bubbles_html)}</div>")

        suggestions = [T["chat_suggestion_1"], T["chat_suggestion_2"], T["chat_suggestion_3"]]
        clicked_suggestion = None
        with st.container(key="chat_suggestions"):
            suggestion_cols = st.columns(3)
            for col, suggestion in zip(suggestion_cols, suggestions):
                with col:
                    if st.button(
                        suggestion,
                        key=f"suggest_{suggestions.index(suggestion)}_{chat_scope_key}",
                        use_container_width=True,
                    ):
                        clicked_suggestion = suggestion

        user_question = st.chat_input(T["chat_placeholder"]) or clicked_suggestion

        if user_question:
            # CHAT RUNTIME FIX: Store only plain user/assistant messages in session_state.
            # Do NOT store/reuse the Google GenAI Chat object across Streamlit reruns.
            # Reusing that SDK chat object was the cause of repeated RuntimeError after
            # asking several questions one after another.
            st.session_state[chat_messages_key].append(
                {"role": "user", "content": user_question, "time": _chat_now()}
            )

            with st.spinner(T["chat_thinking"]):
                try:
                    if gemini_client is None:
                        answer = T["api_key_missing"]
                    else:
                        report_cache_key = f"report_{file_hash}_{language}"
                        if report_cache_key not in st.session_state:
                            # CHAT RUNTIME FIX: Rebuild missing report context from cached result.
                            cached_result = st.session_state.get("cached_result", {})
                            cached_medicines = (
                                cached_result.get("medicines", [])
                                if isinstance(cached_result, dict)
                                else []
                            )
                            cached_valid_medicines = [m for m in cached_medicines if isinstance(m, dict)]
                            st.session_state[report_cache_key] = {
                                "text": build_text_report(
                                    cached_result, T, cached_valid_medicines, language
                                )
                            }

                        report_text = st.session_state[report_cache_key]["text"]

                        # CHAT RUNTIME FIX: Use a stateless Gemini call with the latest chat
                        # history. This is safer in Streamlit than keeping a Chat object in
                        # session_state and prevents RuntimeError on repeated questions.
                        recent_messages = st.session_state[chat_messages_key][-10:]
                        history_lines = []
                        for msg in recent_messages[:-1]:
                            role_label = "Patient" if msg.get("role") == "user" else "Assistant"
                            content = str(msg.get("content", "")).strip()
                            if content:
                                history_lines.append(f"{role_label}: {content}")
                        history_text = "\n".join(history_lines) if history_lines else "No previous chat."

                        chat_prompt = (
                            "You are a friendly medical assistant chatbot inside the MediClarity AI app.\n"
                            "You must answer only using the prescription analysis context below.\n"
                            "Do not diagnose. Do not replace a doctor or pharmacist.\n"
                            f"Answer in simple, easy-to-understand {T['content_language']}.\n"
                            "Keep the answer short: 2-4 sentences unless the patient asks for detail.\n\n"
                            f"PRESCRIPTION ANALYSIS CONTEXT:\n{report_text}\n\n"
                            f"RECENT CHAT HISTORY:\n{history_text}\n\n"
                            f"PATIENT QUESTION:\n{user_question}"
                        )

                        response = gemini_client.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=chat_prompt,
                            config=types.GenerateContentConfig(temperature=0.4),
                        )
                        answer = (response.text or "").strip() or T["chat_error"]

                except AllKeysExhaustedError:
                    answer = T["all_keys_busy"]
                except genai_errors.APIError as exc:
                    answer = f"{T['error_api_prefix']}{exc}"
                except httpx.TransportError:
                    answer = T["error_network"]
                except Exception as exc:
                    # CHAT RUNTIME FIX: Show the real error type only once for debugging.
                    answer = f"{T['chat_error']} ({type(exc).__name__})"

            st.session_state[chat_messages_key].append(
                {"role": "assistant", "content": answer, "time": _chat_now()}
            )
            st.rerun()


# ============================== SIDEBAR ==============================
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "summary"

with st.sidebar:
    _lang_for_nav = st.session_state.get("ui_language", "বাংলা")
    _T_nav = TRANSLATIONS[_lang_for_nav]

    render_html(
        f"""
        <div class="sidebar-header-card">
            <div style="display:flex; align-items:center; gap:10px;">
                <div class="sidebar-logo-circle">💊</div>
                <div>
                    <div class="sidebar-header-title">MediClarity AI</div>
                    <div class="sidebar-header-tagline">{_T_nav['sidebar_tagline']}</div>
                </div>
            </div>
            <div class="sidebar-header-desc">{_T_nav['sidebar_description']}</div>
        </div>
        """
    )

    have_prescription_sidebar = "cached_result" in st.session_state
    if have_prescription_sidebar:
        NAV_ITEMS = [
            ("summary", "📋", _T_nav["nav_summary"]),
            ("medicines", "💊", _T_nav["nav_medicines"]),
            ("schedule", "📅", _T_nav["nav_schedule"]),
            ("advice", "🩺", _T_nav["nav_advice"]),
            ("test_report", "📄", _T_nav["nav_test_report"]),
            ("about", "ℹ️", _T_nav["nav_about"]),
        ]
        with st.container(key="sidebar_nav"):
            for item_key, icon, label in NAV_ITEMS:
                is_active = st.session_state["nav_page"] == item_key
                if st.button(
                    f"{icon}  {label}",
                    key=f"nav_btn_{item_key}",
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["nav_page"] = item_key
                    st.rerun()
        st.divider()

    st.markdown(f"### {_T_nav['accessibility_heading']}")
    language = st.selectbox(
        "🌐 ভাষা / Language", ["বাংলা", "English"], key="ui_language"
    )
    _T_nav = TRANSLATIONS[language]

    # LIGHT THEME DISABLED (commented out on request):
    # appearance_mode = st.selectbox(
    #     "🌓 Theme / থিম", ["Dark", "Light"], key="app_theme"
    # )
    appearance_mode = "Dark"

    elderly_mode = st.toggle("🧓 বড় ফন্ট মোড (Elderly Mode)", key="elderly_mode")

    # COLORBLIND FEATURE DISABLED (commented out on request):
    # colorblind_mode = st.toggle("🎨 Color Blind Friendly Mode", key="colorblind_mode")
    colorblind_mode = False

    st.divider()
    st.caption("⚠️ এটি একটি সহায়ক টুল, চিকিৎসা পরামর্শের বিকল্প নয়।")

T = TRANSLATIONS[language]
CONFIDENCE_BADGES = get_confidence_badges(language, colorblind_mode)

# CHANGED: Replaced the harsh red danger cards with calmer, theme-aware safety colors.
# In Dark mode: safety warnings use deep blue/amber cards.
# In Light mode: safety warnings use soft blue/amber cards with dark readable text.
if appearance_mode == "Light":
    DANGER_COLORS = (
        {"bg": "#eff6ff", "border": "#2563eb", "text": "#1e3a8a"}
        if not colorblind_mode
        else {"bg": "#eef2ff", "border": "#4f46e5", "text": "#312e81"}
    )
    WARNING_COLORS = {"bg": "#fffbeb", "border": "#f59e0b", "text": "#78350f"}
else:
    DANGER_COLORS = (
        {"bg": "#0f2a44", "border": "#38bdf8", "text": "#dbeafe"}
        if not colorblind_mode
        else {"bg": "#1e1b4b", "border": "#f59e0b", "text": "#fde68a"}
    )
    WARNING_COLORS = {"bg": "#33260b", "border": "#fbbf24", "text": "#fef3c7"}

have_prescription = "cached_result" in st.session_state
nav_page = st.session_state.get("nav_page", "summary") if have_prescription else "summary"

# CHANGED: Summary page polish CSS. This makes the Summary page less empty and more like a dashboard.
render_html(
    """
    <style>
    /* CHANGED: Compact success/result bar for the analyzed prescription page */
    .result-status-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(20, 184, 212, 0.10));
        border: 1px solid rgba(52, 211, 153, 0.28);
        border-radius: 18px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 14px 30px rgba(0,0,0,0.18);
    }

    /* CHANGED: Main summary card redesigned for the first page after analysis */
    .summary-hero-card {
        background:
            radial-gradient(circle at top left, rgba(109, 74, 255, 0.20), transparent 30%),
            linear-gradient(145deg, rgba(22, 39, 70, 0.92), rgba(10, 21, 39, 0.88));
        border: 1px solid rgba(139, 108, 255, 0.25);
        border-radius: 24px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.05rem;
        box-shadow: 0 22px 50px rgba(0,0,0,0.30);
    }

    /* CHANGED: Patient info line now looks like chips instead of plain paragraph text */
    .patient-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.65rem 0 0.8rem 0;
    }

    .patient-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        font-size: 0.84rem;
        font-weight: 700;
    }

    /* CHANGED: New responsive dashboard grid for diagnosis, purpose, confidence, and risk */
    .overview-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 1.2rem 0;
    }

    .overview-card {
        background: linear-gradient(145deg, rgba(22, 39, 70, 0.92), rgba(13, 25, 45, 0.88));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        padding: 1rem;
        min-height: 145px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.24);
        backdrop-filter: blur(14px);
    }

    /* Taller variant for the condition/purpose cards, so the longer AI text
       has room to breathe and the stat-badge chip doesn't get cramped. */
    .overview-card-tall {
        min-height: 230px;
        display: flex;
        flex-direction: column;
    }

    .overview-label {
        font-size: 1.05rem;
        color: #aebbd0;
        margin-bottom: 0.5rem;
        font-weight: 900;
    }

    .overview-value {
        font-size: 0.98rem;
        line-height: 1.48;
        font-weight: 400;
        color: #f8fafc;
        max-height: 105px;
        overflow-y: auto;
        padding-right: 4px;
    }

    .overview-card-tall .overview-value {
        max-height: 160px;
        flex: 1 1 auto;
    }

    /* The stat-badge chip only had styling scoped under .stat-card - inside
       .overview-card it was rendering as an unrounded, unpadded plain block. */
    .overview-card .stat-badge {
        display: inline-block;
        margin-top: 10px;
        font-size: 0.72rem;
        background: rgba(20, 184, 212, 0.16);
        color: #67e8f9;
        padding: 2px 9px;
        border-radius: 9999px;
        border: 1px solid rgba(103, 232, 249, 0.18);
    }

    .overview-big-number {
        font-size: 2.05rem;
        font-weight: 900;
        color: #f8fafc;
        line-height: 1;
    }

    .overview-progress-track {
        width: 100%;
        height: 9px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(51, 65, 85, 0.72);
        margin-top: 0.85rem;
    }

    .overview-progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #6d4aff, #14b8d4);
    }

    .risk-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        font-weight: 900;
        margin-top: 0.8rem;
        background: rgba(16, 185, 129, 0.14);
        color: #86efac;
        border: 1px solid rgba(74, 222, 128, 0.24);
    }

    .risk-pill.high-risk {
        background: rgba(239, 68, 68, 0.14);
        color: #fecaca;
        border-color: rgba(248, 113, 113, 0.28);
    }

    @media (max-width: 1100px) {
        .overview-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 720px) {
        .overview-grid { grid-template-columns: 1fr; }
    }
    </style>
    """
)

# LIGHT THEME DISABLED (commented out on request):
# # CHANGED: Light mode override. This is intentionally placed AFTER sidebar controls,
# # so changing the Theme selectbox reruns the app and updates the full UI immediately.
# if appearance_mode == "Light":
#     render_html(
#         """
#         <style>
# 
#         .stApp {
#             background:
#                 radial-gradient(circle at top left, rgba(99, 102, 241, 0.14), transparent 34%),
#                 radial-gradient(circle at top right, rgba(20, 184, 212, 0.12), transparent 30%),
#                 linear-gradient(135deg, #f8fafc 0%, #eef6ff 48%, #f8fafc 100%) !important;
#             color: #0f172a !important;
#         }
#         [data-testid="stHeader"] {
#             background: rgba(248, 250, 252, 0.75) !important;
#             backdrop-filter: blur(14px);
#         }
#         [data-testid="stSidebar"] > div:first-child {
#             background: linear-gradient(180deg, #eef2ff 0%, #e0f2fe 48%, #f8fafc 100%) !important;
#             border-right: 1px solid rgba(15, 23, 42, 0.10) !important;
#             box-shadow: 10px 0 35px rgba(15,23,42,0.10) !important;
#         }
#         [data-testid="stSidebar"] h1,
#         [data-testid="stSidebar"] h2,
#         [data-testid="stSidebar"] h3,
#         [data-testid="stSidebar"] p,
#         [data-testid="stSidebar"] label,
#         [data-testid="stSidebar"] span,
#         .stApp p, .stApp li, .stApp label, .stApp span {
#             color: #0f172a !important;
#         }
#         h1, h2, h3, h4 { color: #0f172a !important; }
#         .sidebar-header-card, .card, .summary-hero-card, .overview-card, .stat-card,
#         .st-key-schedule_outer_card, .st-key-timeline_outer_card {
#             background: rgba(255, 255, 255, 0.82) !important;
#             border-color: rgba(15, 23, 42, 0.10) !important;
#             color: #0f172a !important;
#             box-shadow: 0 18px 45px rgba(15,23,42,0.10) !important;
#         }
#         .sidebar-header-title, .sidebar-header-tagline, .sidebar-header-desc,
#         .card h3, .card p, .card strong, .card li,
#         .welcome-header h2, .overview-value, .overview-big-number, .stat-card .stat-value {
#             color: #0f172a !important;
#         }
#         .welcome-header p, .overview-label, .stat-card .stat-label, .app-footer {
#             color: #475569 !important;
#         }
#         .patient-chip {
#             background: rgba(15, 23, 42, 0.05) !important;
#             border-color: rgba(15, 23, 42, 0.10) !important;
#             color: #0f172a !important;
#         }
#         .st-key-sidebar_nav .stButton > button[kind="secondary"] {
#             background: rgba(15, 23, 42, 0.04) !important;
#             border-color: rgba(15, 23, 42, 0.08) !important;
#             color: #1e293b !important;
#         }
#         .st-key-sidebar_nav .stButton > button[kind="secondary"]:hover {
#             background: rgba(99, 102, 241, 0.12) !important;
#             color: #111827 !important;
#         }
#         table { border-color: rgba(15,23,42,0.10) !important; }
#         thead tr th {
#             background: rgba(99, 102, 241, 0.12) !important;
#             color: #0f172a !important;
#         }
#         tbody tr td {
#             background: rgba(255, 255, 255, 0.72) !important;
#             color: #0f172a !important;
#         }
#         [data-testid="stFileUploader"], .stAlert {
#             background: rgba(255,255,255,0.82) !important;
#             border-color: rgba(99,102,241,0.26) !important;
#             color: #0f172a !important;
#         }
#         input, textarea, select {
#             background: rgba(255, 255, 255, 0.95) !important;
#             color: #0f172a !important;
#             border: 1px solid rgba(15,23,42,0.12) !important;
#         }
#         .st-key-chat_panel {
#             background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(239,246,255,0.96)) !important;
#             border-color: rgba(99,102,241,0.32) !important;
#             box-shadow: 0 20px 45px rgba(15,23,42,0.12) !important;
#         }
#         .chat-header-title, .chat-header-subtitle { color: #0f172a !important; }
#         .bubble-assistant {
#             background: rgba(226, 232, 240, 0.78) !important;
#             color: #0f172a !important;
#             border-color: rgba(15,23,42,0.08) !important;
#         }
#         .bubble-user, .bubble-user * { color: white !important; }
#         .risk-pill {
#             background: rgba(22, 163, 74, 0.10) !important;
#             color: #166534 !important;
#             border-color: rgba(22, 163, 74, 0.20) !important;
#         }
#         .risk-pill.high-risk {
#             background: rgba(220, 38, 38, 0.10) !important;
#             color: #991b1b !important;
#             border-color: rgba(220, 38, 38, 0.22) !important;
#         }
#         /* FINAL FIX: Light mode chat panel readability */
# /* This does not depend on .theme-light class */
# 
# /* Chat panel container */
# .st-key-chat_panel {
#     background: #ffffff !important;
#     border: 1px solid #dbe3ef !important;
# }
# 
# /* Chat header */
# .st-key-chat_panel .chat-header-title,
# .st-key-chat_panel .chat-header-subtitle {
#     color: #0f172a !important;
# }
# 
# /* Assistant message bubble */
# .st-key-chat_panel .bubble-assistant,
# .st-key-chat_panel .bubble-assistant *,
# .st-key-chat_panel .bubble-assistant p,
# .st-key-chat_panel .bubble-assistant span,
# .st-key-chat_panel .bubble-assistant div {
#     background: #eef4fb !important;
#     color: #0f172a !important;
# }
# 
# /* User message bubble */
# .st-key-chat_panel .bubble-user,
# .st-key-chat_panel .bubble-user *,
# .st-key-chat_panel .bubble-user p,
# .st-key-chat_panel .bubble-user span,
# .st-key-chat_panel .bubble-user div {
#     color: #ffffff !important;
# }
# 
# /* Chat timestamps */
# .st-key-chat_panel .bubble-time {
#     color: #64748b !important;
# }
# 
# /* Suggested question buttons */
# .st-key-chat_suggestions .stButton > button,
# .st-key-chat_suggestions .stButton > button *,
# .st-key-chat_suggestions button,
# .st-key-chat_suggestions button * {
#     background: #ffffff !important;
#     color: #0f172a !important;
#     border-color: #e7c76a !important;
# }
# 
# /* Chat input area */
# [data-testid="stChatInput"],
# [data-testid="stChatInput"] *,
# [data-testid="stChatInput"] textarea,
# [data-testid="stChatInput"] input {
#     background: #ffffff !important;
#     color: #0f172a !important;
# }
# 
# /* Chat input placeholder */
# [data-testid="stChatInput"] textarea::placeholder,
# [data-testid="stChatInput"] input::placeholder {
#     color: #64748b !important;
#     opacity: 1 !important;
# }
# 
# /* Dark input wrapper inside Streamlit chat */
# [data-testid="stChatInput"] div {
#     background: #ffffff !important;
#     color: #0f172a !important;
#     border-color: #cbd5e1 !important;
# }
# 
# /* Send button area */
# [data-testid="stChatInput"] button,
# [data-testid="stChatInput"] button * {
#     color: #ffffff !important;
# }
# 
#         /* FINAL PATCH: Light mode clear/close chat action buttons */
#         .st-key-chat_clear_wrap .stButton > button,
#         .st-key-chat_close_wrap .stButton > button {
#             background: #f8fafc !important;
#             color: #334155 !important;
#             border: 1px solid #cbd5e1 !important;
#             border-radius: 999px !important;
#             width: 34px !important;
#             height: 34px !important;
#             min-height: 34px !important;
#             padding: 0 !important;
#             display: inline-flex !important;
#             align-items: center !important;
#             justify-content: center !important;
#             box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08) !important;
#         }
# 
#         .st-key-chat_clear_wrap .stButton > button:hover,
#         .st-key-chat_close_wrap .stButton > button:hover {
#             background: #e0f2fe !important;
#             color: #0f172a !important;
#             border-color: #38bdf8 !important;
#             transform: none !important;
#             box-shadow: 0 8px 18px rgba(14, 165, 233, 0.18) !important;
#         }
# 
#         .st-key-chat_clear_wrap .stButton > button *,
#         .st-key-chat_close_wrap .stButton > button * {
#             color: #334155 !important;
#         }
# 
#         .st-key-chat_clear_wrap .stButton > button:hover *,
#         .st-key-chat_close_wrap .stButton > button:hover * {
#             color: #0f172a !important;
#         }
#         </style>
#         """
#     )

# COLORBLIND FEATURE DISABLED (commented out on request):
# # CHANGED: Color Blind global override. It avoids relying on red/green as the main app accent.
# if colorblind_mode:
#     render_html(
#         """
#         <style>
#         .hero-banner, .stButton > button, .stDownloadButton > button,
#         .st-key-sidebar_nav .stButton > button[kind="primary"], .bubble-user {
#             background: linear-gradient(135deg, #2563eb, #0891b2) !important;
#         }
#         .sidebar-logo-circle, .chat-avatar, .chat-avatar-sm, .st-key-chat_fab .stButton > button {
#             background: linear-gradient(135deg, #2563eb, #0891b2, #f97316) !important;
#         }
#         .overview-progress-fill, .stat-progress-fill {
#             background: linear-gradient(90deg, #2563eb, #f97316) !important;
#         }
#         .summary-hero-card, .card:hover {
#             border-color: rgba(249, 115, 22, 0.35) !important;
#         }
#         .risk-pill.high-risk {
#             background: rgba(249, 115, 22, 0.16) !important;
#             color: #fed7aa !important;
#             border-color: rgba(249, 115, 22, 0.35) !important;
#         }
#         </style>
#         """
#     )


# ============================== FINAL THEME CONTRAST SYSTEM ==============================
# FIXED: This block is the final source of truth for colors.
# It is intentionally rendered AFTER the older CSS + Light Mode + Color Blind patches
# so it overrides all conflicting styles and keeps text readable in both modes.

_theme_is_light = appearance_mode == "Light"

if _theme_is_light:
    _theme = {
        "app_bg": "radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 32%), radial-gradient(circle at top right, rgba(20, 184, 166, 0.12), transparent 28%), linear-gradient(135deg, #f8fafc 0%, #eef6ff 48%, #f8fafc 100%)",
        "sidebar_bg": "linear-gradient(180deg, #f8fafc 0%, #eef6ff 50%, #f8fafc 100%)",
        "card_bg": "#ffffff",
        "card_bg_soft": "#f8fafc",
        "surface_bg": "#ffffff",
        "elevated_bg": "#f1f5f9",
        "input_bg": "#ffffff",
        "text": "#0f172a",
        "heading": "#0b1220",
        "muted": "#475569",
        "border": "#dbe3ef",
        "border_strong": "#cbd5e1",
        "primary": "#2563eb",
        "primary_2": "#0891b2",
        "primary_hover": "#1d4ed8",
        "button_text": "#ffffff",
        "chip_bg": "#eef6ff",
        "chip_text": "#0f172a",
        "info_bg": "#eaf4ff",
        "info_text": "#0f172a",
        "success_bg": "#e9fbf3",
        "success_text": "#065f46",
        "warning_bg": "#fff7ed",
        "warning_text": "#7c2d12",
        "danger_bg": "#fef2f2",
        "danger_text": "#991b1b",
        "chat_bg": "#ffffff",
        "chat_assistant_bg": "#f1f5f9",
        "chat_input_bg": "#ffffff",
        "shadow": "0 18px 45px rgba(15,23,42,0.10)",
    }
else:
    _theme = {
        "app_bg": "radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 34%), radial-gradient(circle at top right, rgba(20, 184, 166, 0.14), transparent 28%), linear-gradient(135deg, #07111f 0%, #081629 45%, #050914 100%)",
        "sidebar_bg": "linear-gradient(180deg, #0f172a 0%, #111827 52%, #07111f 100%)",
        "card_bg": "rgba(15, 30, 52, 0.92)",
        "card_bg_soft": "rgba(22, 39, 70, 0.88)",
        "surface_bg": "rgba(10, 21, 39, 0.88)",
        "elevated_bg": "rgba(22, 39, 70, 0.94)",
        "input_bg": "rgba(15, 30, 52, 0.96)",
        "text": "#f8fafc",
        "heading": "#ffffff",
        "muted": "#cbd5e1",
        "border": "rgba(148, 163, 184, 0.22)",
        "border_strong": "rgba(148, 163, 184, 0.35)",
        "primary": "#3b82f6",
        "primary_2": "#06b6d4",
        "primary_hover": "#2563eb",
        "button_text": "#ffffff",
        "chip_bg": "rgba(59, 130, 246, 0.16)",
        "chip_text": "#dbeafe",
        "info_bg": "rgba(59, 130, 246, 0.16)",
        "info_text": "#dbeafe",
        "success_bg": "rgba(16, 185, 129, 0.16)",
        "success_text": "#a7f3d0",
        "warning_bg": "rgba(245, 158, 11, 0.16)",
        "warning_text": "#fde68a",
        "danger_bg": "rgba(239, 68, 68, 0.16)",
        "danger_text": "#fecaca",
        "chat_bg": "rgba(15, 30, 52, 0.96)",
        "chat_assistant_bg": "rgba(30, 41, 59, 0.96)",
        "chat_input_bg": "rgba(15, 23, 42, 0.96)",
        "shadow": "0 18px 45px rgba(0,0,0,0.32)",
    }

# COLORBLIND FEATURE DISABLED (commented out on request):
# # Color blind mode changes semantic accents but keeps main buttons blue for consistency.
# if colorblind_mode:
#     _theme["primary"] = "#2563eb"
#     _theme["primary_2"] = "#0891b2"

render_html(
    f"""
    <style>
    /* FINAL FIX: universal theme variables */
    :root {{
        --mc-app-bg: {_theme["app_bg"]};
        --mc-sidebar-bg: {_theme["sidebar_bg"]};
        --mc-card-bg: {_theme["card_bg"]};
        --mc-card-bg-soft: {_theme["card_bg_soft"]};
        --mc-surface-bg: {_theme["surface_bg"]};
        --mc-elevated-bg: {_theme["elevated_bg"]};
        --mc-input-bg: {_theme["input_bg"]};
        --mc-text: {_theme["text"]};
        --mc-heading: {_theme["heading"]};
        --mc-muted: {_theme["muted"]};
        --mc-border: {_theme["border"]};
        --mc-border-strong: {_theme["border_strong"]};
        --mc-primary: {_theme["primary"]};
        --mc-primary-2: {_theme["primary_2"]};
        --mc-primary-hover: {_theme["primary_hover"]};
        --mc-button-text: {_theme["button_text"]};
        --mc-chip-bg: {_theme["chip_bg"]};
        --mc-chip-text: {_theme["chip_text"]};
        --mc-info-bg: {_theme["info_bg"]};
        --mc-info-text: {_theme["info_text"]};
        --mc-success-bg: {_theme["success_bg"]};
        --mc-success-text: {_theme["success_text"]};
        --mc-warning-bg: {_theme["warning_bg"]};
        --mc-warning-text: {_theme["warning_text"]};
        --mc-danger-bg: {_theme["danger_bg"]};
        --mc-danger-text: {_theme["danger_text"]};
        --mc-chat-bg: {_theme["chat_bg"]};
        --mc-chat-assistant-bg: {_theme["chat_assistant_bg"]};
        --mc-chat-input-bg: {_theme["chat_input_bg"]};
        --mc-shadow: {_theme["shadow"]};
        --mc-button-bg: linear-gradient(135deg, var(--mc-primary), var(--mc-primary-2));
    }}

    /* FINAL FIX: App-level text and background */
    .stApp {{
        background: var(--mc-app-bg) !important;
        color: var(--mc-text) !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
    .stApp div, .stApp small, .stApp section, .stApp article {{
        color: var(--mc-text) !important;
    }}

    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
        color: var(--mc-heading) !important;
    }}

    .stCaptionContainer, [data-testid="stCaptionContainer"], .stCaptionContainer *,
    .welcome-header p, .overview-label, .stat-card .stat-label,
    .bubble-time, .app-footer {{
        color: var(--mc-muted) !important;
    }}

    /* FINAL FIX: Sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background: var(--mc-sidebar-bg) !important;
        border-right: 1px solid var(--mc-border) !important;
        box-shadow: var(--mc-shadow) !important;
    }}

    [data-testid="stSidebar"] *, 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {{
        color: var(--mc-text) !important;
    }}

    .sidebar-header-card {{
        background: var(--mc-card-bg) !important;
        border: 1px solid var(--mc-border) !important;
        box-shadow: var(--mc-shadow) !important;
    }}

    .sidebar-header-title,
    .sidebar-header-tagline,
    .sidebar-header-desc {{
        color: var(--mc-text) !important;
    }}

    /* FINAL FIX: Cards and custom HTML sections */
    .card, .summary-hero-card, .overview-card, .stat-card,
    .st-key-schedule_outer_card, .st-key-timeline_outer_card,
    .routine-slot-card {{
        background: var(--mc-card-bg) !important;
        border: 1px solid var(--mc-border) !important;
        color: var(--mc-text) !important;
        box-shadow: var(--mc-shadow) !important;
    }}

    .card *, .summary-hero-card *, .overview-card *, .stat-card *,
    .st-key-schedule_outer_card *, .st-key-timeline_outer_card *,
    .routine-slot-card * {{
        color: var(--mc-text) !important;
    }}

    .card h3, .summary-hero-card h3, .overview-big-number,
    .stat-card .stat-value, .routine-slot-title {{
        color: var(--mc-heading) !important;
    }}

    .overview-label, .stat-card .stat-label,
    .routine-slot-subtitle, .routine-slot-list li span {{
        color: var(--mc-muted) !important;
    }}

    .patient-chip, .stat-badge, .ongoing-badge {{
        background: var(--mc-chip-bg) !important;
        color: var(--mc-chip-text) !important;
        border: 1px solid var(--mc-border) !important;
    }}

    .result-status-card {{
        background: var(--mc-success-bg) !important;
        border: 1px solid var(--mc-border-strong) !important;
        color: var(--mc-success-text) !important;
    }}
    .result-status-card * {{
        color: var(--mc-success-text) !important;
    }}

    /* FINAL FIX: Keep main buttons blue and readable in both modes */
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button,
    button[kind="primary"] {{
        background: var(--mc-button-bg) !important;
        color: var(--mc-button-text) !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24) !important;
    }}

    .stButton > button *,
    .stDownloadButton > button *,
    .stFormSubmitButton > button *,
    button[kind="primary"] * {{
        color: var(--mc-button-text) !important;
    }}

    .stButton > button[kind="secondary"],
    .st-key-sidebar_nav .stButton > button[kind="secondary"] {{
        background: var(--mc-card-bg-soft) !important;
        color: var(--mc-text) !important;
        border: 1px solid var(--mc-border) !important;
        box-shadow: none !important;
    }}

    .stButton > button[kind="secondary"] *,
    .st-key-sidebar_nav .stButton > button[kind="secondary"] * {{
        color: var(--mc-text) !important;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        filter: brightness(0.98);
        transform: translateY(-1px);
    }}

    /* FINAL FIX: Inputs, selectboxes, dropdown menus, textareas */
    input, textarea, select,
    [data-baseweb="input"],
    [data-baseweb="textarea"],
    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {{
        background: var(--mc-input-bg) !important;
        color: var(--mc-text) !important;
        border-color: var(--mc-border-strong) !important;
    }}

    input::placeholder, textarea::placeholder {{
        color: var(--mc-muted) !important;
        opacity: 1 !important;
    }}

    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {{
        background: var(--mc-surface-bg) !important;
        color: var(--mc-text) !important;
        border: 1px solid var(--mc-border) !important;
    }}

    [role="option"], [role="option"] *,
    [data-baseweb="select"] *, [data-baseweb="menu"] * {{
        color: var(--mc-text) !important;
    }}

    /* FINAL FIX: File uploader */
    [data-testid="stFileUploader"] {{
        background: var(--mc-card-bg) !important;
        border: 1px dashed var(--mc-border-strong) !important;
        color: var(--mc-text) !important;
    }}
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {{
        color: var(--mc-text) !important;
    }}
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {{
        background: var(--mc-button-bg) !important;
        color: #ffffff !important;
    }}

    /* FINAL FIX: Alerts/info/success/warning/error boxes */
    [data-testid="stAlert"] {{
        background: var(--mc-info-bg) !important;
        border: 1px solid var(--mc-border-strong) !important;
        color: var(--mc-info-text) !important;
    }}
    [data-testid="stAlert"] *,
    .stAlert, .stAlert * {{
        color: var(--mc-info-text) !important;
    }}

    /* FINAL FIX: Tables */
    table, [data-testid="stTable"] {{
        background: var(--mc-card-bg) !important;
        border-color: var(--mc-border) !important;
        color: var(--mc-text) !important;
    }}
    thead tr th {{
        background: var(--mc-elevated-bg) !important;
        color: var(--mc-heading) !important;
        border-color: var(--mc-border) !important;
    }}
    tbody tr td {{
        background: var(--mc-card-bg) !important;
        color: var(--mc-text) !important;
        border-color: var(--mc-border) !important;
    }}
    table * {{
        color: var(--mc-text) !important;
    }}

    /* FINAL FIX: Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--mc-card-bg-soft) !important;
        border: 1px solid var(--mc-border) !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: var(--mc-muted) !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab"] * {{
        color: var(--mc-muted) !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: var(--mc-button-bg) !important;
        color: #ffffff !important;
    }}
    .stTabs [aria-selected="true"] * {{
        color: #ffffff !important;
    }}

    /* FINAL FIX: Chat panel, bubbles, suggestions and input */
    .st-key-chat_panel {{
        background: var(--mc-chat-bg) !important;
        border: 1px solid var(--mc-border) !important;
        color: var(--mc-text) !important;
        box-shadow: var(--mc-shadow) !important;
    }}
    .st-key-chat_panel * {{
        color: var(--mc-text) !important;
    }}

    .chat-header-title {{
        color: var(--mc-heading) !important;
    }}
    .chat-header-subtitle {{
        color: var(--mc-muted) !important;
    }}
    .beta-badge {{
        background: #16a34a !important;
        color: #ffffff !important;
    }}

    .bubble-assistant {{
        background: var(--mc-chat-assistant-bg) !important;
        color: var(--mc-text) !important;
        border: 1px solid var(--mc-border) !important;
    }}
    .bubble-assistant * {{
        color: var(--mc-text) !important;
    }}

    .bubble-user {{
        background: var(--mc-button-bg) !important;
        color: #ffffff !important;
        border: none !important;
    }}
    .bubble-user * {{
        color: #ffffff !important;
    }}

    .bubble-time {{
        color: var(--mc-muted) !important;
    }}
    .bubble-time-user {{
        color: rgba(255,255,255,0.78) !important;
    }}

    .st-key-chat_suggestions .stButton > button {{
        background: var(--mc-chip-bg) !important;
        color: var(--mc-chip-text) !important;
        border: 1px solid var(--mc-border-strong) !important;
        box-shadow: none !important;
    }}
    .st-key-chat_suggestions .stButton > button * {{
        color: var(--mc-chip-text) !important;
    }}

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {{
        background: var(--mc-chat-input-bg) !important;
        color: var(--mc-text) !important;
        border-color: var(--mc-border-strong) !important;
    }}
    [data-testid="stChatInput"] * {{
        color: var(--mc-text) !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {{
        color: var(--mc-muted) !important;
        opacity: 1 !important;
    }}
    [data-testid="stChatInput"] button,
    [data-testid="stChatInput"] button * {{
        background: var(--mc-button-bg) !important;
        color: #ffffff !important;
    }}

    /* FINAL FIX: Markdown/component text areas that used hardcoded dark/light values */
    .stMarkdown, .stMarkdown *,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {{
        color: var(--mc-text) !important;
    }}

    .hero-banner h1, .hero-banner p,
    .stButton > button *, .stDownloadButton > button * {{
        color: #ffffff !important;
    }}

    /* FINAL FIX: Preserve medical warning readability */
    div[style*="background-color:#451a03"],
    div[style*="background-color:#451a03"] * {{
        color: #fde68a !important;
    }}
    </style>
    """
)


# ============================== TIMELINE LAYOUT POLISH ==============================
# FIXED: Equal-height routine cards and more organized timeline tabs.
# This only changes layout/spacing. It does not touch Gemini/API logic.
render_html(
    """
    <style>
    /* FINAL UI FIX: More breathing room around the timeline section */
    .st-key-timeline_outer_card {
        padding: 1.55rem 1.65rem 1.75rem 1.65rem !important;
    }

    /* FINAL UI FIX: Give Day 1-5 / Day 6-10 / etc. tabs proper spacing and pill shape */
    .st-key-timeline_outer_card .stTabs [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 0.75rem !important;
        padding: 0.65rem !important;
        margin: 0.75rem 0 1.05rem 0 !important;
        border-radius: 18px !important;
    }

    .st-key-timeline_outer_card .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] {
        min-width: 104px !important;
        min-height: 42px !important;
        padding: 0.55rem 0.95rem !important;
        border-radius: 999px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: 800 !important;
        white-space: nowrap !important;
    }

    .st-key-timeline_outer_card .stTabs [data-baseweb="tab-panel"],
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.65rem !important;
    }

    /* FINAL UI FIX: Increase gap between Morning / Noon / Night cards */
    .st-key-timeline_outer_card [data-testid="stHorizontalBlock"] {
        gap: 1.1rem !important;
        align-items: stretch !important;
    }

    .st-key-timeline_outer_card [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: 100% !important;
    }

    /* FINAL UI FIX: Routine cards now always have the same box size regardless of text amount */
    .routine-slot-card {
        height: 360px !important;
        min-height: 360px !important;
        max-height: 360px !important;
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        padding: 1.25rem 1.25rem 4.6rem 1.25rem !important;
        box-sizing: border-box !important;
    }

    /* FINAL UI FIX: Long medicine lists scroll inside the card instead of stretching card height */
    .routine-slot-list {
        flex: 1 1 auto !important;
        max-height: 215px !important;
        overflow-y: auto !important;
        padding-right: 0.45rem !important;
        padding-bottom: 0.75rem !important;
        margin-top: 0.5rem !important;
    }

    .routine-slot-list li {
        margin-bottom: 0.8rem !important;
        line-height: 1.5 !important;
    }

    .routine-slot-title {
        min-height: 34px !important;
        margin-bottom: 0.2rem !important;
    }

    .routine-slot-subtitle {
        min-height: 22px !important;
        margin-bottom: 0.55rem !important;
    }

    /* FINAL UI FIX: Keep the scenic decoration fixed at bottom so it does not resize the card */
    .routine-landscape {
        height: 76px !important;
        pointer-events: none !important;
    }

    /* FINAL UI FIX: Better scrollbar inside routine cards */
    .routine-slot-list::-webkit-scrollbar {
        width: 6px !important;
    }
    .routine-slot-list::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.45) !important;
        border-radius: 999px !important;
    }
    .routine-slot-list::-webkit-scrollbar-track {
        background: transparent !important;
    }

    /* FINAL UI FIX: On smaller screens keep the cards readable instead of squeezed */
    @media (max-width: 1100px) {
        .routine-slot-card {
            height: 340px !important;
            min-height: 340px !important;
            max-height: 340px !important;
        }
        .routine-slot-list {
            max-height: 195px !important;
        }
    }
    </style>
    """
)

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


# FINAL CHAT FIX: Keep same width, give better usable height, and prevent layout stretching.
# The message history scrolls inside the panel instead of making the whole page awkwardly tall.
render_html(
    """
    <style>
    /* CHAT PANEL FIX: same width, controlled height.
       Uses a FIXED pixel height (not vh/calc(100vh)) on purpose: on Hugging
       Face Spaces the app runs inside an auto-sizing iframe, and a vh-based
       panel height creates a resize feedback loop that makes the whole page
       jitter/shake. Fixed px breaks that loop. */
    .st-key-chat_panel {
        height: 650px !important;
        min-height: 560px !important;
        max-height: 650px !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* CHAT PANEL FIX: message history gets scroll, not the full page */
    .st-key-chat_messages {
        flex: 1 1 auto !important;
        min-height: 280px !important;
        max-height: 380px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 0.85rem !important;
        margin: 0.55rem 0 0.70rem 0 !important;
        border-radius: 18px !important;
        background: var(--mc-chat-input-bg) !important;
        border: 1px solid var(--mc-border) !important;
        scroll-behavior: smooth !important;
    }

    /* CHAT PANEL FIX: compact suggestions so the message area has more room */
    .st-key-chat_suggestions {
        flex: 0 0 auto !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.45rem !important;
    }

    .st-key-chat_suggestions .stButton > button {
        min-height: 40px !important;
        padding: 0.42rem 0.60rem !important;
        white-space: normal !important;
        line-height: 1.25 !important;
    }

    /* CHAT PANEL FIX: message bubbles wrap cleanly inside same width */
    .st-key-chat_panel .bubble-col {
        max-width: 84% !important;
    }

    .st-key-chat_panel .bubble {
        line-height: 1.55 !important;
        word-break: break-word !important;
    }

    @media (max-height: 760px) {
        .st-key-chat_panel {
            min-height: 480px !important;
            height: 480px !important;
            max-height: 480px !important;
        }
        .st-key-chat_messages {
            min-height: 230px !important;
            max-height: 310px !important;
        }
    }

    /* RESPONSIVE FIX: Streamlit's own columns never wrap on narrow windows - they
       just keep shrinking, which used to squeeze the chat panel (and its nested
       header/suggestion columns) down to a sliver, breaking words onto one
       character per line. Give it a safe floor width on medium windows, and
       fully stack the chat panel below the main content on narrow ones. */
    .st-key-main_chat_row [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    .st-key-chat_panel {
        min-width: 300px;
    }

    @media (max-width: 1100px) {
        .st-key-main_chat_row [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .st-key-chat_panel {
            position: static !important;
            margin-top: 1.25rem !important;
            min-width: 0;
        }
    }
    </style>
    """
)


# ============================== SCROLLBAR STABILITY ==============================
# Reserve the vertical scrollbar's space permanently so it never toggles on/off
# as the page height changes (during upload/analysis, the spinner/status box
# makes the content height hover right around the viewport height). Each toggle
# changed the page width by the scrollbar's width, shifting everything
# left-right = the horizontal "shake" seen on Hugging Face's iframe. With the
# gutter always reserved (and the main area forced to always show a scrollbar),
# the width stays constant and the jitter stops.
render_html(
    """
    <style>
    html, body, .stApp,
    [data-testid="stAppViewContainer"] {
        scrollbar-gutter: stable !important;
    }
    [data-testid="stMain"], section[data-testid="stMain"],
    section.main, .main {
        overflow-y: scroll !important;
        scrollbar-gutter: stable !important;
    }
    </style>
    """
)

gemini_client = get_gemini_client()
if gemini_client is None:
    st.error(T["api_key_missing"])


def render_upload_flow() -> None:
    """The pre-analysis flow: hero banner, steps, uploader, analyze button."""
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
    steps = [
        ("📤", T["step1_title"], T["step1_desc"]),
        ("🤖", T["step2_title"], T["step2_desc"]),
        ("📋", T["step3_title"], T["step3_desc"]),
    ]
    for col, (icon, step_title, step_desc) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"<div style='text-align:center; font-size:1.75rem;'>{icon}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align:center;'><strong>{step_title}</strong></p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align:center; color:var(--mc-muted, #94a3b8); "
                f"font-size:0.85rem;'>{html.escape(step_desc)}</p>",
                unsafe_allow_html=True,
            )

    st.divider()

    uploaded_file = st.file_uploader(
        T["upload_label"],
        type=["jpg", "jpeg", "png"],
        key="prescription_uploader",
    )

    if uploaded_file is None:
        st.info(T["empty_state"])
        return

    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error(T["image_read_error"])
        st.stop()

    preview_col, _ = st.columns([2, 3])
    with preview_col:
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
                st.write(T["status_classifying"])
                result = explain_prescription(
                    gemini_client, image, content_language=T["content_language"]
                )
                # Safety gate: only accept real, legible prescriptions. A medicine
                # box/strip/label/bill or an unreadable image is rejected here so
                # the app never hallucinates a prescription around it.
                if (
                    not result.get("is_prescription")
                    or result.get("classification_confidence", 0) < CLASSIFY_CONFIDENCE_MIN
                ):
                    status.update(label=T["status_classifying"], state="error")
                    cat_label = get_category_label(T, result.get("image_category", "unknown"))
                    st.error(T["classify_not_prescription"].format(category=cat_label))
                    st.stop()
                if result.get("extraction_confidence", 0) < EXTRACTION_CONFIDENCE_MIN:
                    status.update(label=T["status_classifying"], state="error")
                    st.error(T["extraction_unclear"])
                    st.stop()
                st.write(T["status_reading"])
                st.session_state["cached_file_hash"] = file_hash
                st.session_state["cached_language"] = language
                st.session_state["cached_result"] = result
                st.session_state.pop("cached_comparison", None)
                status.update(label=T["status_done"], state="complete", expanded=False)
            except AllKeysExhaustedError:
                status.update(label=T["status_api_error"], state="error")
                st.error(T["all_keys_busy"])
                st.stop()
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
        st.rerun()

    if st.session_state.get("cached_file_hash") == file_hash:
        if st.session_state.get("cached_language") != language:
            st.info(T["language_switch_notice"])


if not have_prescription:
    render_upload_flow()
else:
    file_hash = st.session_state["cached_file_hash"]
    result = st.session_state["cached_result"]
    medicines = result.get("medicines", [])
    valid_medicines = [med for med in medicines if isinstance(med, dict)]

    if not medicines:
        st.warning(T["no_medicines"])
    elif not valid_medicines:
        st.error(T["bad_format"])
    else:
        report_cache_key = f"report_{file_hash}_{language}"
        if report_cache_key not in st.session_state:
            try:
                report_pdf_bytes = build_pdf_report(result, T, valid_medicines, language)
            except Exception:
                report_pdf_bytes = None
            st.session_state[report_cache_key] = {
                "text": build_text_report(result, T, valid_medicines, language),
                "pdf": report_pdf_bytes,
            }
        report_text = st.session_state[report_cache_key]["text"]
        report_pdf = st.session_state[report_cache_key]["pdf"]

        with st.container(key="main_chat_row"):
            main_col, chat_col = st.columns([7, 3])

        with main_col:
            header_col, pdf_col, print_col = st.columns([5, 2, 2])
            with header_col:
                render_html(
                    f"""
                    <div class="welcome-header">
                        <h2>{T['welcome_title']}</h2>
                        <p>{T['welcome_subtitle']}</p>
                    </div>
                    """
                )
            with pdf_col:
                if report_pdf is not None:
                    st.download_button(
                        T["download_pdf_button"],
                        data=report_pdf,
                        file_name="prescription_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf",
                    )
                else:
                    st.button(
                        T["download_pdf_button"],
                        disabled=True,
                        use_container_width=True,
                        key="download_pdf",
                        help=T["pdf_generation_failed"],
                    )
            with print_col:
                print_button(T["print_button"])

            with st.expander(T["reupload_expander_label"], expanded=False):
                render_upload_flow()

            st.divider()

            if nav_page == "summary":
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
                    f'<span style="background-color:#1e3a5f !important; color:#93c5fd !important; padding:2px 10px; '
                    f'border-radius:9999px; font-size:0.75rem; font-weight:600; margin-right:6px; '
                    f'display:inline-block; margin-top:4px;">🏷️ {c}</span>'
                    for c in detected_conditions
                )
                patient_info = result.get("patient_info", {})
                if not isinstance(patient_info, dict):
                    patient_info = {}
                patient_name = html.escape(str(patient_info.get("name") or T["not_identified"]))
                patient_age = html.escape(str(patient_info.get("age") or T["not_identified"]))

                # CHANGED: Replaced the default Streamlit success bar with a custom dashboard-style result bar.
                render_html(
                    f"""
                    <div class="result-status-card">
                        <div><strong>{T['status_done']}</strong> — {len(valid_medicines)} {T['success_suffix'].strip()}</div>
                        <div style="font-size:0.82rem; opacity:0.85;">MediClarity AI</div>
                    </div>
                    """
                )

                # CHANGED: Summary card redesigned using chips and a cleaner hero-style card.
                render_html(
                    f"""
                    <div class="summary-hero-card">
                        <h3>{T['summary_title']}
                            <span style="background-color:{oc_bg} !important; color:{oc_fg} !important;
                                padding:3px 11px; border-radius:9999px; font-size:0.75rem;
                                font-weight:800; vertical-align:middle; margin-left:6px;">
                                {T['summary_overall']} {oc_label}
                            </span>
                        </h3>
                        <div class="patient-chip-row">
                            <span class="patient-chip">👤 {T['patient_name_label']}: {patient_name}</span>
                            <span class="patient-chip">🎂 {T['patient_age_label']}: {patient_age}</span>
                            <span class="patient-chip">💊 {T['total_medicines_label']}: {len(valid_medicines)} {T['total_medicines_suffix']}</span>
                        </div>
                        <p style="margin-bottom:0.7rem; line-height:1.65;">{overall_summary}</p>
                        {condition_chips}
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

                # CHANGED: Kept voice/TXT/copy features, but placed them before dashboard cards for faster demo workflow.
                speak_button(
                    speech_text, T["listen_button"], T["tts_lang"], T["tts_error"],
                    key=f"speak_{file_hash}_{language}",
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        T["download_txt_button"], data=report_text,
                        file_name="prescription_report.txt", mime="text/plain",
                        use_container_width=True, key="download_txt",
                    )
                with col2:
                    copy_button(
                        report_text, T["copy_button"], T["copy_success"], T["copy_failed"], "report"
                    )

                # CHANGED: Replaced four Streamlit stat cards with custom dashboard cards; long Bangla text no longer stretches the whole page badly.
                confidence_pct = CONFIDENCE_PERCENT.get(overall_confidence_key, 50)
                has_high_risk = any(as_bool(m.get("high_risk")) for m in valid_medicines)
                risk_class = "high-risk" if has_high_risk else ""
                risk_value = T["risk_high"] if has_high_risk else T["risk_low"]
                render_html(
                    f"""
                    <div class="overview-grid">
                        <div class="overview-card overview-card-tall">
                            <div class="overview-label">{T['stat_condition_label']}</div>
                            <div class="overview-value">{probable_condition}</div>
                            <div class="stat-badge">{T['ai_inference_badge']}</div>
                        </div>
                        <div class="overview-card overview-card-tall">
                            <div class="overview-label">{T['stat_purpose_label']}</div>
                            <div class="overview-value">{treatment_purpose}</div>
                            <div class="stat-badge">{T['ai_analysis_badge']}</div>
                        </div>
                        <div class="overview-card">
                            <div class="overview-label">{T['stat_confidence_label']}</div>
                            <div class="overview-big-number">{confidence_pct}%</div>
                            <div class="overview-progress-track">
                                <div class="overview-progress-fill" style="width:{confidence_pct}%;"></div>
                            </div>
                        </div>
                        <div class="overview-card">
                            <div class="overview-label">{T['stat_risk_label']}</div>
                            <div class="risk-pill {risk_class}">{'⚠️' if has_high_risk else '✅'} {risk_value}</div>
                        </div>
                    </div>
                    """
                )

                doctor_info = result.get("doctor_info", {})
                if not isinstance(doctor_info, dict):
                    doctor_info = {}
                admin_cols = st.columns(2)
                with admin_cols[0]:
                    render_html(
                        f"""
                        <div class="card">
                            <h3>{T['doctor_title']}</h3>
                            <p><strong>{T['doctor_name_label']}:</strong> {html.escape(str(doctor_info.get('name') or T['not_identified']))}</p>
                            <p><strong>{T['doctor_degree_label']}:</strong> {html.escape(str(doctor_info.get('degree') or T['not_identified']))}</p>
                            <p><strong>{T['doctor_hospital_label']}:</strong> {html.escape(str(doctor_info.get('hospital') or T['not_identified']))}</p>
                            <p><strong>{T['doctor_phone_label']}:</strong> {html.escape(str(doctor_info.get('phone') or T['not_mentioned']))}</p>
                        </div>
                        """
                    )
                with admin_cols[1]:
                    render_html(
                        f"""
                        <div class="card">
                            <h3>{T['date_title']}</h3>
                            <p><strong>{T['date_label']}:</strong> {html.escape(result.get('prescription_date') or T['not_mentioned'])}</p>
                            <p><strong>{T['followup_label']}:</strong> {html.escape(result.get('follow_up_date') or T['not_mentioned'])}</p>
                        </div>
                        """
                    )
#==================================== medicines page ========================================
            elif nav_page == "medicines":
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
                        f'<span style="background-color:{DANGER_COLORS["bg"]} !important; color:{DANGER_COLORS["text"]} !important; '
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
                                    <span style="background-color:{confidence_bg} !important; color:{confidence_fg} !important;
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
#==================================== schedule page ========================================

            elif nav_page == "schedule":
                with st.container(key="schedule_outer_card"):
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

                st.write("")

                with st.container(key="timeline_outer_card"):
                    # CHANGED: Timeline section redesigned like the reference dashboard picture.
                    # CHANGED: Uses fixed tabs: Day 1-5, 6-10, 11-15, 16-20, 21-30, Full Course.

                    st.subheader(T["timeline_title"])
                    st.caption(T["timeline_intro"])

                    day_ranges = build_treatment_timeline(valid_medicines)

                    # CHANGED: Fixed timeline tab labels.
                    if language == "বাংলা":
                        tab_labels = [
                            "দিন ১-৫",
                            "দিন ৬-১০",
                            "দিন ১১-১৫",
                            "দিন ১৬-২০",
                            "দিন ২১-৩০",
                            "পূর্ণ কোর্স",
                        ]
                    else:
                        tab_labels = [
                            "Day 1-5",
                            "Day 6-10",
                            "Day 11-15",
                            "Day 16-20",
                            "Day 21-30",
                            "Full Course",
                        ]

                    tabs = st.tabs(tab_labels)

                    def render_timeline_slot_card(slot_icon, slot_title, slot_bg_class, medicines_html):
                        # CHANGED: New premium slot card that looks like the picture: morning/noon/night cards.
                        render_html(
                            f"""
                            <div class="routine-slot-card {slot_bg_class}">
                                <div class="routine-slot-title">
                                    <span class="routine-slot-icon">{slot_icon}</span>
                                    <span>{slot_title}</span>
                                </div>
                                <div class="routine-slot-subtitle">খাবার পর</div>
                                <ul class="routine-slot-list">
                                    {medicines_html}
                                </ul>
                                <div class="routine-landscape"></div>
                            </div>
                            """
                        )

                    def render_day_range(day_range) -> None:
                        # CHANGED: Three-column visual medicine routine: Morning, Noon, Night.
                        slot_cols = st.columns(3)

                        slot_config = [
                            ("morning", "☀️", T["morning_col"], "morning-card"),
                            ("afternoon", "🌤️", T["afternoon_col"], "noon-card"),
                            ("night", "🌙", T["night_col"], "night-card"),
                        ]

                        for slot_col, (flag_key, slot_icon, slot_label, slot_bg_class) in zip(slot_cols, slot_config):
                            slot_items = []

                            for med in day_range["medicines"]:
                                timing = med.get("timing")
                                if not isinstance(timing, dict):
                                    timing = {}

                                if not as_bool(timing.get(flag_key)):
                                    continue

                                med_name = html.escape(str(med.get("medicine_name", T["unknown_medicine"])))
                                meal_key = str(timing.get("meal_relation", "unspecified")).strip().lower()
                                meal_text = get_meal_label(language, meal_key)

                                ongoing_note = (
                                    f' <span class="ongoing-badge">{T["timeline_ongoing_badge"]}</span>'
                                    if parse_duration_days(med) is None
                                    else ""
                                )

                                slot_items.append(
                                    f"<li><strong>{med_name}</strong><br><span>{meal_text}</span>{ongoing_note}</li>"
                                )

                            items_html = "".join(slot_items) or (
                                f'<li class="empty-med">{T["timeline_no_meds"]}</li>'
                            )

                            with slot_col:
                                render_timeline_slot_card(
                                    slot_icon, strip_emoji(slot_label), slot_bg_class, items_html
                                )

                    # CHANGED: Render Day 1-5, 6-10, 11-15, 16-20, 21-30.
                    for tab, day_range in zip(tabs[:5], day_ranges):
                        with tab:
                            render_day_range(day_range)

                    # CHANGED: Full Course tab shows all medicines together.
                    with tabs[-1]:
                        all_meds_range = {
                            "start_day": 1,
                            "end_day": 30,
                            "medicines": valid_medicines,
                        }
                        render_day_range(all_meds_range)

            elif nav_page == "advice":
                st.subheader(T["nav_advice"])
                drug_interactions = html.escape(result.get("drug_interactions") or T["no_interaction"])
                render_html(
                    f"""
                    <div style="background:{DANGER_COLORS['bg']}; border:1px solid {DANGER_COLORS['border']};
                        border-left:6px solid {DANGER_COLORS['border']}; border-radius:14px;
                        padding:1rem 1.25rem; margin-bottom:1rem; color:{DANGER_COLORS['text']} !important;
                        box-shadow:0 12px 28px rgba(15,23,42,0.12); line-height:1.65;">
                        <strong style="color:{DANGER_COLORS['text']} !important;">💊⚠️ {T['interaction_label']}:</strong>
                        <span style="color:{DANGER_COLORS['text']} !important;"> {drug_interactions}</span>
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
                        <div style="background:{DANGER_COLORS['bg']}; border:1px solid {DANGER_COLORS['border']};
                            border-left:6px solid {DANGER_COLORS['border']}; border-radius:14px;
                            padding:1rem 1.25rem; margin-bottom:1rem; color:{DANGER_COLORS['text']} !important;
                            box-shadow:0 12px 28px rgba(15,23,42,0.12); line-height:1.65;">
                            <strong style="color:{DANGER_COLORS['text']} !important;">⚠️ {T['high_risk_label']}:</strong>
                            <span style="color:{DANGER_COLORS['text']} !important;"> {risk_names} — {T['high_risk_warning']}</span>
                        </div>
                        """
                    )

                render_html(
                    f"""
                    <div style="background:{WARNING_COLORS['bg']}; border:1px solid {WARNING_COLORS['border']};
                        border-left:6px solid {WARNING_COLORS['border']}; border-radius:14px;
                        padding:1rem 1.25rem; margin-top:1.25rem; color:{WARNING_COLORS['text']} !important;
                        box-shadow:0 12px 28px rgba(15,23,42,0.10); line-height:1.65;">
                        <span style="color:{WARNING_COLORS['text']} !important;">{T['disclaimer']}</span>
                    </div>
                    """
                )

            elif nav_page == "test_report":
                st.subheader(T["test_report_title"])
                st.caption(T["test_report_intro"])

                test_uploaded_file = st.file_uploader(
                    T["test_upload_label"],
                    type=["jpg", "jpeg", "png", "pdf"],
                    key="test_report_uploader",
                )

                if test_uploaded_file is not None:
                    test_file_hash = hashlib.md5(test_uploaded_file.getvalue()).hexdigest()
                    test_mime = (test_uploaded_file.type or "").lower()
                    test_is_pdf = test_mime == "application/pdf" or (
                        test_uploaded_file.name or ""
                    ).lower().endswith(".pdf")

                    if test_is_pdf:
                        st.info(f"📄 {test_uploaded_file.name}")
                    else:
                        try:
                            test_preview_image = Image.open(test_uploaded_file).convert("RGB")
                        except Exception:
                            st.error(T["test_file_read_error"])
                            st.stop()
                        test_preview_col, _ = st.columns([2, 3])
                        with test_preview_col:
                            st.image(
                                test_preview_image, caption=T["test_upload_caption"],
                                use_container_width=True,
                            )

                    test_analyze_clicked = st.button(
                        T["test_analyze_button"],
                        disabled=gemini_client is None,
                        use_container_width=True,
                        key="test_analyze_button",
                    )

                    test_already_cached = (
                        st.session_state.get("cached_test_file_hash") == test_file_hash
                        and st.session_state.get("cached_test_language") == language
                    )

                    if test_analyze_clicked and not test_already_cached:
                        with st.status(T["test_status_analyzing"], expanded=True) as status:
                            st.write(T["test_status_reading"])
                            try:
                                test_file_part = prepare_test_report_part(test_uploaded_file)
                                test_result = explain_test_report(
                                    gemini_client, test_file_part, content_language=T["content_language"]
                                )
                                # Safety gate: only accept real, legible test reports.
                                if (
                                    not test_result.get("is_test_report")
                                    or test_result.get("classification_confidence", 0) < CLASSIFY_CONFIDENCE_MIN
                                ):
                                    status.update(label=T["status_classifying"], state="error")
                                    cat_label = get_category_label(
                                        T, test_result.get("image_category", "unknown")
                                    )
                                    st.error(T["classify_not_test_report"].format(category=cat_label))
                                    st.stop()
                                if test_result.get("extraction_confidence", 0) < EXTRACTION_CONFIDENCE_MIN:
                                    status.update(label=T["status_classifying"], state="error")
                                    st.error(T["extraction_unclear"])
                                    st.stop()
                                st.session_state["cached_test_file_hash"] = test_file_hash
                                st.session_state["cached_test_language"] = language
                                st.session_state["cached_test_result"] = test_result
                                st.session_state.pop("cached_comparison", None)
                                status.update(label=T["status_done"], state="complete", expanded=False)
                            except AllKeysExhaustedError:
                                status.update(label=T["status_api_error"], state="error")
                                st.error(T["all_keys_busy"])
                                st.stop()
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

                    if st.session_state.get("cached_test_file_hash") == test_file_hash:
                        if st.session_state.get("cached_test_language") != language:
                            st.info(T["language_switch_notice"])

                        test_result = st.session_state["cached_test_result"]
                        test_valid_tests = [
                            t for t in test_result.get("tests", []) if isinstance(t, dict)
                        ]

                        if not test_valid_tests:
                            st.warning(T["no_tests"])
                        else:
                            TEST_STATUS_BADGES = get_test_status_badges(language, colorblind_mode)

                            test_confidence_key = str(
                                test_result.get("overall_confidence", "low")
                            ).strip().lower()
                            test_oc_label, test_oc_fg, test_oc_bg = CONFIDENCE_BADGES.get(
                                test_confidence_key, CONFIDENCE_BADGES["low"]
                            )
                            test_overall_summary = html.escape(
                                test_result.get("overall_summary") or T["no_summary"]
                            )

                            test_patient_info = test_result.get("patient_info", {})
                            if not isinstance(test_patient_info, dict):
                                test_patient_info = {}
                            test_patient_name = html.escape(
                                str(test_patient_info.get("name") or T["not_identified"])
                            )
                            test_patient_age = html.escape(
                                str(test_patient_info.get("age") or T["not_identified"])
                            )

                            render_html(
                                f"""
                                <div class="card">
                                    <h3>{T['test_summary_title']}
                                        <span style="background-color:{test_oc_bg} !important; color:{test_oc_fg} !important;
                                            padding:2px 10px; border-radius:9999px; font-size:0.75rem;
                                            font-weight:600; vertical-align:middle;">
                                            {T['summary_overall']} {test_oc_label}
                                        </span>
                                    </h3>
                                    <p><strong>{T['patient_name_label']}:</strong> {test_patient_name}
                                        &nbsp;|&nbsp; <strong>{T['patient_age_label']}:</strong> {test_patient_age}</p>
                                    <p>{test_overall_summary}</p>
                                    <p><strong>{T['total_tests_label']}:</strong> {len(test_valid_tests)}</p>
                                </div>
                                """
                            )

                            test_lab_info = test_result.get("lab_info", {})
                            if not isinstance(test_lab_info, dict):
                                test_lab_info = {}
                            test_lab_name = html.escape(str(test_lab_info.get("name") or T["not_identified"]))
                            test_referring_doctor = html.escape(
                                str(test_lab_info.get("referring_doctor") or T["not_identified"])
                            )
                            test_report_date = html.escape(
                                test_result.get("report_date") or T["not_mentioned"]
                            )

                            render_html(
                                f"""
                                <div class="card">
                                    <h3>{T['lab_title']}</h3>
                                    <p><strong>{T['lab_name_label']}:</strong> {test_lab_name}</p>
                                    <p><strong>{T['referring_doctor_label']}:</strong> {test_referring_doctor}</p>
                                    <p><strong>{T['report_date_label']}:</strong> {test_report_date}</p>
                                </div>
                                """
                            )

                            st.divider()
                            st.subheader(T["tests_section_title"])
                            test_cols = st.columns(2)
                            for test_idx, test in enumerate(test_valid_tests):
                                test_name = html.escape(str(test.get("test_name") or T["unknown_test"]))
                                test_value = html.escape(str(test.get("value") or "-"))
                                test_unit = html.escape(str(test.get("unit") or ""))
                                test_ref_range = html.escape(str(test.get("reference_range") or "-"))
                                test_purpose = html.escape(str(test.get("purpose") or T["no_info"]))
                                test_explanation = html.escape(str(test.get("explanation") or T["no_info"]))
                                test_status_key = str(test.get("status", "normal")).strip().lower()
                                test_status_label, test_status_fg, test_status_bg = TEST_STATUS_BADGES.get(
                                    test_status_key, TEST_STATUS_BADGES["normal"]
                                )
                                with test_cols[test_idx % 2]:
                                    render_html(
                                        f"""
                                        <div class="card">
                                            <h3>🔬 {test_name}
                                                <span style="background-color:{test_status_bg} !important; color:{test_status_fg} !important;
                                                    padding:2px 10px; border-radius:9999px; font-size:0.75rem;
                                                    font-weight:600; vertical-align:middle;">
                                                    {test_status_label}
                                                </span>
                                            </h3>
                                            <p><strong>{T['test_value_label']}:</strong> {test_value} {test_unit}</p>
                                            <p><strong>{T['reference_range_label']}:</strong> {test_ref_range}</p>
                                            <p><strong>{T['test_purpose_label']}:</strong> {test_purpose}</p>
                                            <p><strong>{T['test_explanation_label']}:</strong> {test_explanation}</p>
                                        </div>
                                        """
                                    )

                            render_html(
                                f"""
                                <div style="background-color:#451a03; border-left:4px solid #d97706;
                                    border-radius:8px; padding:1rem 1.25rem; margin-top:1.25rem; color:#fde68a;">
                                    {T['test_disclaimer']}
                                </div>
                                """
                            )

                st.divider()
                st.subheader(T["compare_title"])
                if "cached_test_result" not in st.session_state:
                    st.info(T["compare_need_both"])
                else:
                    if st.button(
                        T["compare_button"], key="compare_button", use_container_width=True
                    ):
                        with st.spinner(T["compare_thinking"]):
                            try:
                                st.session_state["cached_comparison"] = compare_reports(
                                    gemini_client,
                                    result,
                                    st.session_state["cached_test_result"],
                                    T["content_language"],
                                )
                            except AllKeysExhaustedError:
                                st.error(T["all_keys_busy"])
                            except genai_errors.APIError as exc:
                                st.error(f"{T['error_api_prefix']}{exc}")
                            except httpx.TransportError:
                                st.error(T["error_network"])
                            except json.JSONDecodeError:
                                st.error(T["error_parse"])
                            except Exception as exc:
                                st.error(f"{T['error_unexpected_prefix']}{exc}")

                    if isinstance(st.session_state.get("cached_comparison"), dict):
                        render_html(
                            build_comparison_card_html(st.session_state["cached_comparison"], T)
                        )

            elif nav_page == "about":
                render_html(
                    f"""
                    <div class="card">
                        <h3>{T['about_heading']}</h3>
                        <p>{T['about_body']}</p>
                    </div>
                    """
                )
                render_html(
                    f"""
                    <div class="card">
                        <h3>{T['about_tech_heading']}</h3>
                        <p>Streamlit &middot; Google Gemini 2.5 Flash (Vision) &middot; Python</p>
                    </div>
                    """
                )
                st.caption(T["about_disclaimer"])

        with chat_col:
            render_chat_panel(gemini_client, file_hash, language, T)

render_html(
    """
    <div class="app-footer" style="text-align:center; padding:1rem; color:#6b7280; font-size:0.85rem;">
        MediClarity AI — Hackathon Project
    </div>
    """
)
