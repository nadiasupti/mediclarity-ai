"""MediClarity AI - Streamlit app that explains prescriptions in simple Bangla."""

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
    """Group each medicine's active days into day-ranges with a stable set of
    medicines, so a 28-day medicine doesn't produce 28 near-identical blocks.

    A medicine with no determinable duration is treated as active for the
    whole visible timeline (marked "ongoing"), since assuming it stops at an
    arbitrary day would be misleading; if not a single medicine has a known
    duration, the timeline is just 1 day (there's no basis to extend further).
    """
    known_days = [
        parse_duration_days(med) for med in valid_medicines if parse_duration_days(med) is not None
    ]
    max_days = max(known_days) if known_days else 1

    day_snapshots = []
    for day in range(1, max_days + 1):
        active = []
        for med in valid_medicines:
            duration_days = parse_duration_days(med)
            if duration_days is None or day <= duration_days:
                active.append(med)
        signature = tuple(sorted(str(med.get("medicine_name", "")) for med in active))
        day_snapshots.append((day, signature, active))

    ranges = []
    i = 0
    while i < len(day_snapshots):
        start_day, signature, active = day_snapshots[i]
        j = i
        while j + 1 < len(day_snapshots) and day_snapshots[j + 1][1] == signature:
            j += 1
        end_day = day_snapshots[j][0]
        ranges.append({"start_day": start_day, "end_day": end_day, "medicines": active})
        i = j + 1
    return ranges


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
    /* Outer section wrapper: a slightly darker box that groups a whole section
       (table + tabs + inner .card boxes) so it reads as one nested box-in-box
       card, instead of loose widgets floating directly on the page background. */
    .st-key-schedule_outer_card, .st-key-timeline_outer_card {
        background-color: #161b2e;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.25rem 1.5rem 1.5rem 1.5rem;
    }
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        padding: 1rem 0 0.5rem 0;
    }
    .welcome-header h2 { margin-bottom: 0.1rem; }
    .welcome-header p { color: #94a3b8; margin-top: 0; }
    .stat-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .stat-card .stat-label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; }
    .stat-card .stat-value { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; }
    .stat-card .stat-badge {
        display: inline-block; margin-top: 6px; font-size: 0.7rem;
        background-color: #1e3a5f; color: #93c5fd; padding: 1px 8px; border-radius: 9999px;
    }
    .stat-progress-track {
        background-color: #334155; border-radius: 9999px; height: 8px; margin-top: 8px; overflow: hidden;
    }
    .stat-progress-fill {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6); height: 100%;
    }
    /* Sidebar nav pill buttons */
    .sidebar-header-card {
        background: linear-gradient(160deg, #1e1b4b, #1e293b);
        border: 1px solid #312e81;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-logo-circle {
        width: 44px; height: 44px; border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.35rem; flex-shrink: 0;
    }
    .sidebar-header-title { font-weight: 700; font-size: 1.05rem; color: #f1f5f9; }
    .sidebar-header-tagline { font-size: 0.72rem; color: #a5b4fc; }
    .sidebar-header-desc { font-size: 0.78rem; color: #cbd5e1; line-height: 1.5; margin-top: 0.6rem; }
    .st-key-sidebar_nav .stButton > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        font-weight: 500;
        border-radius: 10px;
        margin-bottom: 2px;
    }
    .st-key-sidebar_nav .stButton > button[kind="secondary"] {
        background-color: transparent;
        border: none;
        color: #cbd5e1;
    }
    .st-key-sidebar_nav .stButton > button[kind="secondary"]:hover {
        background-color: #1e293b;
        color: #f1f5f9;
    }
    .st-key-sidebar_nav .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.35);
    }
    /* Chat widget lives in its own reserved right-hand column (not a page-wide
       fixed overlay) so it can never cover the main content underneath it -
       `position: fixed` was sitting on top of whatever content happened to be
       scrolled into that corner. `sticky` instead keeps it pinned near the top
       of its own column while the page scrolls, with zero overlap risk. */
    .st-key-chat_panel {
        position: sticky;
        top: 1rem;
        align-self: flex-start;
        background-color: #12122b;
        border: 1px solid #4c1d95;
        border-radius: 20px;
        padding: 1rem;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
        box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.25), 0 12px 30px rgba(0, 0, 0, 0.45);
    }
    .st-key-chat_fab {
        position: sticky;
        top: 1rem;
        align-self: flex-start;
        display: flex;
        justify-content: flex-end;
    }
    .st-key-chat_fab .stButton > button {
        width: 60px; height: 60px; border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none; font-size: 1.6rem; padding: 0;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.5);
    }
    .st-key-chat_close_wrap .stButton > button {
        background: transparent; border: none; color: #94a3b8;
        font-size: 1rem; padding: 0.1rem 0.4rem; min-height: auto; float: right;
    }
    .st-key-chat_close_wrap .stButton > button:hover { color: #f1f5f9; background: transparent; }
    .st-key-chat_suggestions .stButton > button {
        background: transparent;
        border: 1px solid #4c1d95;
        color: #c4b5fd;
        border-radius: 9999px;
        font-size: 0.7rem;
        padding: 0.3rem 0.5rem;
    }
    .st-key-chat_suggestions .stButton > button:hover {
        background-color: #312e81;
        color: #f1f5f9;
    }
    .chat-header {
        display: flex; align-items: center; gap: 10px;
        padding-bottom: 0.75rem; border-bottom: 1px solid #334155; margin-bottom: 0.75rem;
    }
    .chat-avatar {
        width: 40px; height: 40px; border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6, #1e3a8a);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem; flex-shrink: 0;
    }
    .chat-avatar-sm {
        width: 28px; height: 28px; border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6, #1e3a8a);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; flex-shrink: 0;
    }
    .chat-header-title { font-weight: 700; color: #e2e8f0; }
    .beta-badge {
        background-color: #166534; color: #dcfce7; font-size: 0.65rem;
        padding: 1px 6px; border-radius: 9999px; margin-left: 4px; vertical-align: middle;
    }
    .chat-header-subtitle { font-size: 0.75rem; color: #94a3b8; }
    .bubble-row { display: flex; gap: 8px; margin-bottom: 14px; align-items: flex-end; }
    .bubble-row-user { justify-content: flex-end; }
    .bubble-col { display: flex; flex-direction: column; max-width: 78%; }
    .bubble-row-user .bubble-col { align-items: flex-end; }
    .bubble {
        padding: 0.55rem 0.85rem;
        border-radius: 14px;
        font-size: 0.85rem;
        line-height: 1.5;
        text-align: left;
    }
    .bubble-assistant {
        background-color: #1e293b;
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
    }
    .bubble-user {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border-bottom-right-radius: 4px;
    }
    .bubble-time { font-size: 0.65rem; color: #64748b; margin-top: 3px; padding: 0 4px; }
    .bubble-time-user { color: #a5b4fc; }
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


def explain_test_report(
    client: genai.Client, file_part: types.Part, content_language: str = "Bangla"
) -> dict:
    system_prompt = (
        "You are a medical assistant that reads medical test/lab reports (blood tests, "
        "urine tests, imaging reports, etc.) and explains them to patients in simple, "
        f"easy-to-understand {content_language}. You must always respond ONLY with valid "
        "JSON, no extra text. The JSON must be a single object with exactly these top-level "
        "keys: \"overall_summary\", \"overall_confidence\", \"patient_info\", \"lab_info\", "
        "\"report_date\", \"tests\".\n"
        f"- \"overall_summary\": a 2-3 sentence plain-{content_language} summary of the whole "
        "report, mentioning whether results are mostly normal or if anything stands out.\n"
        "- \"overall_confidence\": one of \"high\", \"medium\", or \"low\" (English, "
        "lowercase), reflecting your overall confidence in reading the entire document.\n"
        "- \"patient_info\": an object with exactly these keys: \"name\" and \"age\", each in "
        f"simple {content_language} if visible, otherwise a short phrase meaning \"not "
        "identified\", written in the target language.\n"
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
        "Below is a medical test/lab report (image or PDF). "
        f"Analyze it fully and respond in simple {content_language} as instructed."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[file_part, user_prompt],
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
        data = {"tests": data}
    elif not isinstance(data, dict):
        data = {}

    tests = data.get("tests")
    if not isinstance(tests, list):
        tests = []

    patient_info_raw = data.get("patient_info")
    if not isinstance(patient_info_raw, dict):
        patient_info_raw = {}
    patient_info = {
        "name": str(patient_info_raw.get("name") or "").strip(),
        "age": str(patient_info_raw.get("age") or "").strip(),
    }

    lab_info_raw = data.get("lab_info")
    if not isinstance(lab_info_raw, dict):
        lab_info_raw = {}
    lab_info = {
        "name": str(lab_info_raw.get("name") or "").strip(),
        "referring_doctor": str(lab_info_raw.get("referring_doctor") or "").strip(),
    }

    return {
        "overall_summary": str(data.get("overall_summary") or "").strip(),
        "overall_confidence": str(data.get("overall_confidence", "low")).strip().lower(),
        "patient_info": patient_info,
        "lab_info": lab_info,
        "report_date": str(data.get("report_date") or "").strip(),
        "tests": tests,
    }


def compare_reports(
    client: genai.Client, prescription_text: str, test_report_text: str, content_language: str
) -> str:
    """Ask Gemini whether the prescribed medicines are consistent with the test results.

    Returns plain text (not JSON) since this is a one-off explanatory answer, not
    something the UI needs to render into structured fields.
    """
    prompt = (
        "You are a medical assistant. Below is a patient's prescription analysis and a "
        "separate medical test report analysis, both already summarized in plain language. "
        f"Compare them and explain, in simple {content_language}, whether the prescribed "
        "medicines appear consistent with the test results (e.g. does a prescribed medicine "
        "make sense given an abnormal result, or does something seem inconsistent or worth "
        "double-checking with a doctor). Keep your answer to 3-6 sentences. Do not provide a "
        "diagnosis, and remind the patient to discuss any concerns with their doctor or "
        "pharmacist.\n\n"
        f"PRESCRIPTION ANALYSIS:\n{prescription_text}\n\n"
        f"TEST REPORT ANALYSIS:\n{test_report_text}"
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.4),
    )
    return response.text.strip()


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


def _render_chat_bubble(role: str, content: str, time_str: str) -> str:
    """Build one WhatsApp-style chat bubble row as an HTML fragment."""
    safe_content = html.escape(content).replace("\n", "<br>")
    if role == "user":
        return f"""
        <div class="bubble-row bubble-row-user">
            <div class="bubble-col">
                <div class="bubble bubble-user">{safe_content}</div>
                <div class="bubble-time bubble-time-user">{time_str} ✓✓</div>
            </div>
        </div>
        """
    return f"""
    <div class="bubble-row bubble-row-assistant">
        <div class="chat-avatar-sm">🤖</div>
        <div class="bubble-col">
            <div class="bubble bubble-assistant">{safe_content}</div>
            <div class="bubble-time">{time_str}</div>
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
            if st.button("🤖", key="chat_fab_button"):
                st.session_state["chat_open"] = True
                st.rerun()
        return

    chat_scope_key = f"{file_hash}_{language}"
    chat_messages_key = f"chat_messages_{chat_scope_key}"
    chat_session_key = f"chat_session_{chat_scope_key}"
    if chat_messages_key not in st.session_state:
        st.session_state[chat_messages_key] = []

    with st.container(key="chat_panel"):
        header_col, close_col = st.columns([6, 1])
        with header_col:
            render_html(
                f"""
                <div class="chat-header">
                    <div class="chat-avatar">🤖</div>
                    <div>
                        <div class="chat-header-title">{T['chat_panel_title']}
                            <span class="beta-badge">{T['chat_beta_badge']}</span></div>
                        <div class="chat-header-subtitle">{T['chat_panel_subtitle']}</div>
                    </div>
                </div>
                """
            )
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
            st.session_state[chat_messages_key].append(
                {"role": "user", "content": user_question, "time": _chat_now()}
            )
            with st.spinner(T["chat_thinking"]):
                try:
                    report_cache_key = f"report_{file_hash}_{language}"
                    report_text = st.session_state[report_cache_key]["text"]
                    chat = get_chat_session(
                        gemini_client, report_text, T["content_language"], chat_session_key
                    )
                    answer = chat.send_message(user_question).text
                except genai_errors.APIError as exc:
                    answer = f"{T['error_api_prefix']}{exc}"
                except httpx.TransportError:
                    answer = T["error_network"]
                except Exception:
                    answer = T["chat_error"]
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
    elderly_mode = st.toggle("🧓 বড় ফন্ট মোড (Elderly Mode)", key="elderly_mode")
    colorblind_mode = st.toggle("🎨 Color Blind Friendly Mode", key="colorblind_mode")

    st.divider()
    st.caption("⚠️ এটি একটি সহায়ক টুল, চিকিৎসা পরামর্শের বিকল্প নয়।")

T = TRANSLATIONS[language]
CONFIDENCE_BADGES = get_confidence_badges(language, colorblind_mode)
DANGER_COLORS = DANGER_COLORS_COLORBLIND if colorblind_mode else DANGER_COLORS_DEFAULT
have_prescription = "cached_result" in st.session_state
nav_page = st.session_state.get("nav_page", "summary") if have_prescription else "summary"

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
            st.caption(step_desc)

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
            st.session_state[report_cache_key] = {
                "text": build_text_report(result, T, valid_medicines, language),
                "pdf": build_pdf_report(result, T, valid_medicines, language),
            }
        report_text = st.session_state[report_cache_key]["text"]
        report_pdf = st.session_state[report_cache_key]["pdf"]

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
                st.download_button(
                    T["download_pdf_button"],
                    data=report_pdf,
                    file_name="prescription_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf",
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

                st.success(f"{T['success_prefix']}{len(valid_medicines)}{T['success_suffix']}")

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
                        {condition_chips}
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

                st.divider()
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    render_stat_card(
                        T["stat_condition_label"], probable_condition, T["ai_inference_badge"]
                    )
                with stat_cols[1]:
                    render_stat_card(
                        T["stat_purpose_label"], treatment_purpose, T["ai_analysis_badge"]
                    )
                with stat_cols[2]:
                    render_stat_card(
                        T["stat_confidence_label"],
                        f"{CONFIDENCE_PERCENT.get(overall_confidence_key, 50)}%",
                        progress_pct=CONFIDENCE_PERCENT.get(overall_confidence_key, 50),
                    )
                with stat_cols[3]:
                    has_high_risk = any(as_bool(m.get("high_risk")) for m in valid_medicines)
                    render_stat_card(
                        T["stat_risk_label"],
                        T["risk_high"] if has_high_risk else T["risk_low"],
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
                    st.subheader(T["timeline_title"])
                    st.caption(T["timeline_intro"])

                    day_ranges = build_treatment_timeline(valid_medicines)
                    tab_labels = []
                    for day_range in day_ranges:
                        if day_range["start_day"] == day_range["end_day"]:
                            tab_labels.append(T["timeline_day_single"].format(day=day_range["start_day"]))
                        else:
                            tab_labels.append(
                                T["timeline_day_range"].format(
                                    start=day_range["start_day"], end=day_range["end_day"]
                                )
                            )
                    if len(day_ranges) > 1:
                        tab_labels.append(f"📋 {T['timeline_full_course']}")

                    tabs = st.tabs(tab_labels)

                    def render_day_range(day_range) -> None:
                        slot_cols = st.columns(3)
                        for slot_col, (flag_key, slot_label) in zip(
                            slot_cols,
                            (
                                ("morning", T["morning_col"]),
                                ("afternoon", T["afternoon_col"]),
                                ("night", T["night_col"]),
                            ),
                        ):
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
                                    f' <span style="color:#94a3b8; font-size:0.8rem;">'
                                    f'({T["timeline_ongoing_badge"]})</span>'
                                    if parse_duration_days(med) is None
                                    else ""
                                )
                                slot_items.append(f"<li>{med_name} — {meal_text}{ongoing_note}</li>")
                            items_html = "".join(slot_items) or (
                                f'<li style="color:#94a3b8;">{T["timeline_no_meds"]}</li>'
                            )
                            with slot_col:
                                render_html(
                                    f"""
                                    <div class="card">
                                        <h3>{slot_label}</h3>
                                        <ul style="margin-top:0; padding-left:1.1rem;">{items_html}</ul>
                                    </div>
                                    """
                                )

                    for tab, day_range in zip(tabs, day_ranges):
                        with tab:
                            render_day_range(day_range)

                    if len(day_ranges) > 1:
                        with tabs[-1]:
                            all_meds_range = {
                                "start_day": 1,
                                "end_day": max(r["end_day"] for r in day_ranges),
                                "medicines": valid_medicines,
                            }
                            render_day_range(all_meds_range)

            elif nav_page == "advice":
                st.subheader(T["nav_advice"])
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

                render_html(
                    f"""
                    <div style="background-color:#451a03; border-left:4px solid #d97706;
                        border-radius:8px; padding:1rem 1.25rem; margin-top:1.25rem; color:#fde68a;">
                        {T['disclaimer']}
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
                                st.session_state["cached_test_file_hash"] = test_file_hash
                                st.session_state["cached_test_language"] = language
                                st.session_state["cached_test_result"] = test_result
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
                                        <span style="background-color:{test_oc_bg}; color:{test_oc_fg};
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
                                                <span style="background-color:{test_status_bg}; color:{test_status_fg};
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
                                compare_test_text = build_test_report_text(
                                    st.session_state["cached_test_result"], T
                                )
                                st.session_state["cached_comparison"] = compare_reports(
                                    gemini_client, report_text, compare_test_text, T["content_language"]
                                )
                            except genai_errors.APIError as exc:
                                st.error(f"{T['error_api_prefix']}{exc}")
                            except httpx.TransportError:
                                st.error(T["error_network"])
                            except Exception as exc:
                                st.error(f"{T['error_unexpected_prefix']}{exc}")

                    if "cached_comparison" in st.session_state:
                        render_html(
                            f"""
                            <div class="card">
                                <p>{html.escape(st.session_state['cached_comparison'])}</p>
                            </div>
                            """
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
