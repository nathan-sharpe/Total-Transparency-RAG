# API image (Phase 4). Layers ordered least- to most-changing so a source
# edit re-uses the cached pip install layer.
FROM python:3.12.10-slim

# Unbuffered stdout so container logs stream line-by-line as they happen;
# no .pyc files keeps the image (and COPY --chown) a little leaner.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Root-only build steps end here; the app runs unprivileged.
RUN useradd --create-home app
COPY --chown=app:app . .
USER app

EXPOSE 8000

# python main.py reads API_HOST/API_PORT (and all other config) from the
# environment via rag/config.py — compose supplies the container values.
CMD ["python", "main.py"]
