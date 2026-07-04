# MediClarity AI

A Streamlit app that reads a prescription image, extracts the text with OCR, and uses OpenAI GPT-4o to explain the prescription in simple Bangla — medicine names, purpose, dosage, side effects, and precautions.

## Features

- Upload a prescription image (JPG, JPEG, PNG)
- OCR text extraction with Tesseract
- GPT-4o powered explanation in plain Bangla
- Clean card-based UI showing:
  - Medicine names
  - Purpose of each medicine
  - Dosage instructions
  - Possible side effects
  - Important precautions
- Graceful error handling for missing API keys, OCR failures, and API errors

## Prerequisites

- Python 3.9+
- [Tesseract OCR engine](https://github.com/tesseract-ocr/tesseract) installed on your system
  - **Windows:** install from the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) and add the install folder (e.g. `C:\Program Files\Tesseract-OCR`) to your PATH.
  - **macOS:** `brew install tesseract`
  - **Linux (Debian/Ubuntu):** `sudo apt install tesseract-ocr`
- An OpenAI API key with access to `gpt-4o`

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

## Setting the API key

1. Copy `.env.example` to `.env`:

   ```powershell
   copy .env.example .env
   ```

2. Open `.env` and set your key:

   ```
   OPENAI_API_KEY=sk-your-real-key-here
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
3. Review the OCR text (optional, in the expandable section) and the Bangla explanation cards for each medicine.

## Disclaimer

This tool is for informational purposes only and does not replace professional medical advice. Always consult a doctor or pharmacist before making decisions about medication.
