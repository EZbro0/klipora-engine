FROM python:3.11-slim

# FFmpeg for rendering, Noto fonts so captions work in all 21 languages
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fonts-noto-core fonts-noto-cjk fontconfig ca-certificates \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "app.py"]
