# MediClarity AI

A Streamlit app that reads a prescription image and uses Google Gemini Vision to explain it in simple Bangla — medicine names, purpose, dosage, side effects, and precautions.

## Features

- Upload a prescription image (JPG, JPEG, PNG)
- Gemini Vision reads the image directly and explains it in plain Bangla (no separate OCR step)
- A medicine timing table (সকাল/দুপুর/রাত and before/after meal) summarizing when to take each medicine
- Clean card-based UI showing:
  - Medicine names
  - Purpose of each medicine
  - Dosage instructions
  - Possible side effects
  - Important precautions
  - A confidence badge (উচ্চ/মাঝারি/কম নির্ভরযোগ্যতা) showing how confident Gemini is in its reading of each medicine
- Graceful error handling for missing API keys and API errors

## Prerequisites

- Python 3.9+
- A free [Google Gemini API key](https://aistudio.google.com/apikey) (Google AI Studio → "Create API key")

## Installation

1. Create and activate a virtual environment (recommended):

   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Setting the API keys

1. Copy `.env.example` to `.env`:

   ```powershell
   copy .env.example .env
   ```

2. Open `.env` and set your key:

   ```
   GEMINI_API_KEY=your-real-gemini-key-here
   ```

The `.env` file is already excluded from version control via `.gitignore`.

## Running the app

```powershell
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in your browser.

## Usage

1. Upload a clear photo or scan of a prescription (JPG/JPEG/PNG).
2. Click "প্রেসক্রিপশন বিশ্লেষণ করুন" (Analyze Prescription).
3. Review the Bangla explanation cards for each medicine.

## Notes on accuracy

Gemini Vision works best on clear, well-lit photos. Handwritten prescriptions with messy handwriting are harder to read reliably — if the model is unsure about a medicine, it will note that the reading may be uncertain rather than guessing silently.

## Disclaimer

This tool is for informational purposes only and does not replace professional medical advice. Always consult a doctor or pharmacist before making decisions about medication.
