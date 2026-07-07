---
title: MediClarity AI
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# MediClarity AI 💊

A simple web app that reads a photo of a prescription and explains it in **plain Bangla or English** — what each medicine is for, how to take it, side effects, and warnings. Built with Streamlit and Google Gemini Vision.

## What it does

1. You upload a photo of a prescription.
2. Google Gemini (AI) reads the photo directly — no separate OCR step needed.
3. The app shows you a clean, easy-to-read breakdown of everything on the prescription.

## Features

- **Prescription summary** — patient name/age, a plain-language summary, probable diagnosis, treatment purpose, and how many medicines were found.
- **Doctor info** — name, degree, hospital/chamber, phone (if visible on the prescription).
- **Dates** — prescription date and follow-up date (if mentioned).
- **Drug interaction check** and **high-risk medicine warnings**.
- **Daily schedule table** — which medicine to take in the morning/afternoon/night, and before/after/with meals.
- **🗓️ AI Treatment Timeline** — a full day-by-day plan (e.g. "Day 1–5", "Day 6–7") built from each medicine's duration, so you can see exactly which medicines are active on which day and when a medicine's course ends. Medicines with no clear duration are marked "Ongoing". Generated fresh from the current prescription only — no reminders, notifications, accounts, or storage involved.
- **Per-medicine cards** — purpose, dosage, duration, side effects, precautions, and a confidence badge (how sure the AI is about that reading).
- **Export the report** — download as PDF, download as TXT, print, or copy the full result to your clipboard.
- **💬 Chat with your prescription** — a floating chat launcher (the animated 🤖 button pinned to the bottom-right corner) opens a docked assistant panel. Ask follow-up questions in plain language ("Why was this medicine given?", "Should I take it on an empty stomach?"). The chatbot remembers the conversation and stays grounded in your specific prescription.
- **🧪 AI Medical Test Report Interpreter** — upload a lab/test report (image or PDF) separately from the prescription. For each test it explains why it's typically done, what your result means in plain language, and marks it 🟢 Normal / 🟡 Borderline / 🔴 Abnormal — without ever stating a diagnosis. If you've analyzed both a prescription and a test report, click **⚖️ Compare with Prescription** to see whether your medicines line up with your results.
- **Multi-page dashboard** — after a prescription is analyzed, the sidebar navigation switches between separate pages: Summary, Medicines, Daily Schedule, Advice & Warnings, Report Analysis, and About. The app uses a single, eye-friendly dark theme throughout.
- **Accessibility settings** (in the sidebar):
  - 🌐 Switch the whole app between Bangla and English
  - 🔊 Listen — plays the summary out loud as audio
  - 🧓 Elderly mode — bigger text and buttons

## Tech stack

- **Streamlit** — the web app framework
- **Google Gemini 2.5 Flash (Vision)** — reads the prescription image and writes the explanation
- **gTTS** — turns the summary into spoken audio
- **fpdf2** — generates the downloadable PDF report (with a bundled Noto Sans Bengali font so Bangla text renders correctly)
- **Python 3.9+**

---

## How to run this project (for teammates)

### 1. Get the code and open a terminal in the project folder

```powershell
cd MediClarity-AI
```

### 2. Create a virtual environment and activate it

```powershell
python -m venv venv
venv\Scripts\activate
```

You'll know it worked if you see `(venv)` at the start of your terminal line.

### 3. Install all required packages

```powershell
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Copy the example env file:

```powershell
copy .env.example .env
```

Open the new `.env` file in any text editor and paste your key:

```
GEMINI_API_KEY=your-real-gemini-key-here
```

Don't have a key? Get a free one at [Google AI Studio](https://aistudio.google.com/apikey) → "Create API key".

(The `.env` file is never uploaded to GitHub — it's already in `.gitignore`, so your key stays private.)

### 5. Run the app

```powershell
streamlit run app.py
```

Your browser should open automatically. If not, open the link shown in the terminal (usually `http://localhost:8501`).

### 6. Every time after the first setup

You only need steps 2 (activate) and 5 (run) again — no need to reinstall packages unless `requirements.txt` changes:

```powershell
venv\Scripts\activate
streamlit run app.py
```

---

## How to use the app

1. Upload a clear photo of a prescription (JPG/JPEG/PNG).
2. (Optional) In the sidebar, pick your language, or turn on Elderly mode.
3. Click the "Analyze Prescription" button.
4. Once analyzed, use the **sidebar navigation** to move between pages: **Summary** (overview, patient/doctor info, confidence and risk), **Medicines** (per-medicine cards), **Daily Schedule** (timing table + AI Treatment Timeline), **Advice & Warnings**, **Report Analysis**, and **About**.
5. On the Summary page, click "🔊 Listen" to hear the summary read aloud, and use the ⬇️ PDF / ⬇️ TXT / 🖨️ Print / 📋 Copy buttons to save or share the report.
6. Click the floating **🤖 chat button** (bottom-right corner) to open the assistant and ask follow-up questions, either by typing or tapping a suggested question.
7. Open the **Report Analysis** page to upload a lab/test report (image or PDF) under "🧪 Medical Test Report Interpreter" and click "Analyze Test Report" to get the same kind of plain-language breakdown, per test.
8. If you've analyzed both a prescription and a test report, click "⚖️ Compare with Prescription" (on the Report Analysis page) to check whether your medicines are consistent with your test results.

## Customizing the chatbot robot icon (optional)

The floating launcher and the in-chat avatar both fall back to a plain 🤖 emoji, but you can drop in your own robot artwork with **no code changes** — just place a file with the right name in `assets/chatbot/`:

- **Floating launcher button:** `robot.webp` or `robot.gif` (animated is fine; WebP is smallest). A Lottie `robot.json` is detected but not yet rendered.
- **Chat header + reply avatars:** `avatar.png`, `avatar.webp`, or `avatar.gif`.

The floating icon's vertical position and size are controlled by a few constants at the top of `app.py` (`CHATBOT_BOTTOM_PX`, `CHATBOT_SIZE_DESKTOP_PX`, etc.) — change one value to move it up/down or resize it. After swapping an asset file, restart the app so it picks up the new image.

## Notes on accuracy

- Works best on clear, well-lit photos. Messy handwriting is harder to read reliably.
- If the AI isn't sure about something, it says so instead of guessing silently — look for the confidence badges.
- This tool is for informational purposes only and **does not replace professional medical advice**. Always consult a doctor or pharmacist before making decisions about medication.
