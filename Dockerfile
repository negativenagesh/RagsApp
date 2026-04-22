FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

#Install uv and create a project virtual environment managed by uv.
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /usr/local/bin/

COPY . /workspace

RUN uv venv /workspace/.venv && \
    uv pip install --python /workspace/.venv/bin/python \
    fastapi \
    "uvicorn[standard]" \
    python-dotenv \
    httpx \
    requests \
    openai \
    elasticsearch \
    pydantic \
    numpy \
    pandas \
    python-multipart \
    pdfplumber \
    pypdf \
    tiktoken \
    aiohttp \
    pyyaml \
    pytesseract \
    pdf2image \
    pymupdf \
    langchain-text-splitters \
    olefile \
    python-docx \
    odfpy \
    openpyxl \
    networkx \
    pillow

ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"