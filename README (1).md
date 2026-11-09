# Klipora video engine

Background worker for Klipora. It takes queued Reddit-story requests, writes the
narration with Gemini, voices it, adds word-by-word captions over the user's
gameplay and uploads the finished vertical video.

Runs as a Docker web service (for example a free Render.com instance).
Opening the service URL shows a small status page in JSON.

Environment variables:
- `SUPABASE_URL` (required)
- `SUPABASE_SERVICE_ROLE_KEY` (required, secret)
- `GEMINI_API_KEY` (required, secret)
- `VIDEO_WIDTH`, `VIDEO_HEIGHT`, `X264_PRESET`, `VIDEO_FPS` (optional; on a small free server use 540, 960, ultrafast, 24)
