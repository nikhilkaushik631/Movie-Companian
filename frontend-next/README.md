# Cinemizer Next.js Frontend

Run locally:

```bash
cd frontend-next
npm install
# optional: preview UI without backend
export NEXT_PUBLIC_DEMO=1
npm run dev
```

Config:
- API base URL: `NEXT_PUBLIC_API_BASE_URL` (defaults to `http://localhost:8000`).
- Demo mode: `NEXT_PUBLIC_DEMO=1` disables live API calls and shows UI-only behavior.
- For full functionality, run the FastAPI backend.

Notes:
- Voice: toggling voice enables TTS; server TTS (Deepgram) is used if available, else browser speechSynthesis.

