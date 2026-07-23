# API image (Phase 4). Layers ordered least- to most-changing so a source
# edit re-uses the cached pip install layer.
FROM python:3.12.10-slim

# Unbuffered stdout so container logs stream line-by-line as they happen;
# no .pyc files keeps the image (and COPY --chown) a little leaner.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU embedding profile (EMBEDDING_PROFILE=sentence-transformers) is the Phase 6
# adopted default. torch comes from the PyTorch CPU wheel index first — exactly as
# ci.yml does — so the multi-GB CUDA PyPI wheel is never pulled; requirements-cpu.txt
# then installs the rest (it -r's in the base requirements.txt).
COPY requirements.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-cpu.txt

# Root-only build steps end here; the app runs unprivileged.
RUN useradd --create-home app
USER app

# Pre-bake the embedding model into the image so container startup needs no network
# and builds stay reproducible. Run as `app` so the HF cache lands in this user's
# HOME (where the runtime process, also `app`, finds it); kept above the source COPY
# so the layer survives source edits.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# With the model baked in, force offline: startup loads from the cache and never
# reaches out to the HF Hub, so the container needs no network at boot and builds
# stay reproducible. Set AFTER the bake above (which does need network).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=app:app . .

EXPOSE 8000

# python main.py reads API_HOST/API_PORT (and all other config) from the
# environment via rag/config.py — compose supplies the container values.
CMD ["python", "main.py"]
