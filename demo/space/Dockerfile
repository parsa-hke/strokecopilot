# Stroke Decision Copilot — the public Hugging Face Space (DEC-32). CPU basic hardware.
# Built from demo/space/build/space/ (python -m strokecopilot.demo.space_build), never from the repo root:
# that tree holds no split file, no data, no run folders, and passed the leak scan.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface TOKENIZERS_PARALLELISM=false
WORKDIR $HOME/app

# PyTorch for CPU only (the verifier's retrieval embeds claims with bge-base), then the package with the Space extra.
RUN pip install --upgrade pip && pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user pkg/ pkg/
RUN pip install "./pkg[space]"
# The retrieval model is fetched at build time, so a cold start does not download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

COPY --chown=user static/ static/
COPY --chown=user bank/ bank/
ENV BANK_DIR=bank STATIC_DIR=static PORT=7860
EXPOSE 7860
CMD ["python", "-m", "strokecopilot.demo.server"]
