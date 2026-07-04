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
- **Per-medicine cards** — purpose, dosage, duration, side effects, precautions, and a confidence badge (how sure the AI is about that reading).
- **Accessibility settings** (in the sidebar):
  - 🌐 Switch the whole app between Bangla and English
  - 🔊 Listen — plays the summary out loud as audio
  - 🧓 Elderly mode — bigger text and buttons
  - 🎨 Color blind friendly mode — different colors for warnings/badges

## Tech stack

- **Streamlit** — the web app framework
- **Google Gemini 2.5 Flash (Vision)** — reads the prescription image and writes the explanation
- **gTTS** — turns the summary into spoken audio
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
2. (Optional) In the sidebar, pick your language, or turn on Elderly mode / Color blind mode.
3. Click the "Analyze Prescription" button.
4. Read the results — summary, schedule table, and medicine cards.
5. Click "🔊 Listen" to hear the summary read aloud.

## Notes on accuracy

- Works best on clear, well-lit photos. Messy handwriting is harder to read reliably.
- If the AI isn't sure about something, it says so instead of guessing silently — look for the confidence badges.
- This tool is for informational purposes only and **does not replace professional medical advice**. Always consult a doctor or pharmacist before making decisions about medication.
