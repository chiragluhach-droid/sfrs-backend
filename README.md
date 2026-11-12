# SFRS · Backend (FastAPI + OpenCV)

The face recognition and attendance API for the [SFRS dashboard](https://github.com/chiragluhach-droid/sfrs-frontend).

- **Detector:** YuNet. **Recognizer:** SFace, producing 128-d embeddings compared by cosine similarity. Both are OpenCV DNN models; no dlib is needed.
- **Database:** MongoDB (`students`, `scans` collections)
- **Auth:** a single admin password exchanged for a signed bearer token

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py
.venv/bin/uvicorn app.main:app --port 8000      # docs at http://localhost:8000/docs
```

Locally, auth is **off** unless you set `ADMIN_PASSWORD`.

## Deploy on Render

1. Create a free **MongoDB Atlas** cluster. Copy its `mongodb+srv://…` connection string, and under Network Access allow `0.0.0.0/0`.
2. On Render, choose **New → Blueprint**, pick this repo, and it reads `render.yaml`.
3. Fill in the prompted variables:
   - `MONGO_URL`: the Atlas connection string
   - `ADMIN_PASSWORD`: the password for the dashboard login
   - `CORS_ORIGINS`: your Vercel URL, e.g. `https://sfrs-frontend.vercel.app` (no trailing slash)
4. Deploy, then check `https://<your-service>.onrender.com/api/health`.

On the free tier the service sleeps after about 15 minutes idle, and the first request after that takes 30–60 seconds.

## Environment variables

| Var | Default | |
|---|---|---|
| `MONGO_URL` | `mongodb://localhost:27017` | |
| `MONGO_DB` | `sfrs` | |
| `ADMIN_PASSWORD` | *(empty → auth off)* | required when deployed |
| `SECRET_KEY` | random per boot | signs login tokens |
| `CORS_ORIGINS` | `*` | comma-separated |
| `MATCH_THRESHOLD` | `0.40` | cosine similarity needed for a match |
| `DUPLICATE_THRESHOLD` | `0.55` | blocks enrolling the same face twice |
| `RELOG_SECONDS` | `60` | minimum gap between check-ins per student |
