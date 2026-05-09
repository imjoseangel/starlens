FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for matplotlib/Pillow rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libfontconfig1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app/starlens/ app/starlens/
RUN pip install --no-cache-dir ".[cache]"

COPY . .

EXPOSE 8000

CMD ["python", "app/main.py"]
