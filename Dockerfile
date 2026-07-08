# Python version the app's pinned dependencies are proven to install on.
FROM python:3.13-slim

WORKDIR /app

# Install Python dependencies first so this layer is cached across code changes.
# All of the app's deps (streamlit, google-genai, pillow, gTTS, fpdf2,
# uharfbuzz, httpx, python-dotenv) ship as pure-Python or manylinux wheels,
# so no apt-get system packages are needed.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app. This includes assets/ (the bundled Noto Sans
# Bengali font used for Bangla PDF export, and the chatbot/avatar icons).
COPY . .

# Streamlit config via env vars. The port MUST match `app_port: 8501` in the
# README frontmatter, or Hugging Face won't find the running app. Bind to all
# interfaces, run headless, and skip usage-stats collection (which would try
# to write to a home dir that may be read-only in the Space container).
#
# XSRF/CORS are disabled because Hugging Face serves the app behind a reverse
# proxy. With XSRF protection on, Streamlit rejects the file-uploader's POST
# (the prescription/report image upload) with a 403 error, since the request's
# origin doesn't match. Turning both off lets uploads work through the proxy.
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_ENABLE_CORS=false

EXPOSE 8501

# XSRF/CORS flags are also passed on the command line (highest config
# precedence) to be certain they take effect behind HF's proxy - without this
# the file-uploader POST is rejected with a 403.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", \
     "--server.enableCORS=false"]
