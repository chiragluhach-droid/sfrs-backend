from __future__ import annotations

import os
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo.errors import DuplicateKeyError

from . import auth, db
from .face import FaceEngine, Gallery, decode_image, portrait_data_uri

MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.40"))  # cosine similarity (SFace)
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.55"))
RELOG_SECONDS = int(os.getenv("RELOG_SECONDS", "60"))  # don't re-log same student within this window
UNKNOWN_LOG_SECONDS = 8

engine: FaceEngine | None = None
gallery = Gallery()
_last_logged: dict[str, datetime] = {}


def refresh_gallery() -> int:
    return gallery.rebuild(db.students.find({}, {"embeddings": 1}))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = FaceEngine()
    db.ensure_indexes()
    n = refresh_gallery()
    print(f"[sfrs] gallery loaded: {n} embeddings · threshold {MATCH_THRESHOLD}")
    yield


app = FastAPI(title="SFRS · Student Face Recognition API", version="1.0.0", lifespan=lifespan)
app.middleware("http")(auth.auth_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────── helpers ─────────────────────────

def now() -> datetime:
    return datetime.now(timezone.utc)


def local_midnight_utc(days_ago: int = 0) -> datetime:
    local = datetime.now().astimezone()
    start = local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_ago)
    return start.astimezone(timezone.utc)


def oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(404, "Student not found")


def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def public_student(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "student_id": doc.get("student_id"),
        "name": doc.get("name"),
        "email": doc.get("email") or "",
        "phone": doc.get("phone") or "",
        "department": doc.get("department") or "",
        "year": doc.get("year") or "",
        "section": doc.get("section") or "",
        "photo": doc.get("photo") or "",
        "samples": doc.get("samples", len(doc.get("embeddings", []))),
        "created_at": iso(doc.get("created_at")),
        "last_seen": iso(doc.get("last_seen")),
        "scan_count": doc.get("scan_count", 0),
    }


def public_scan(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "status": doc["status"],
        "student_ref": doc.get("student_ref"),
        "student_id": doc.get("student_id"),
        "name": doc.get("name"),
        "department": doc.get("department"),
        "photo": doc.get("photo"),
        "score": round(doc.get("score", 0.0), 3),
        "ts": iso(doc["ts"]),
    }


STUDENT_PROJECTION = {"embeddings": 0}
SCAN_PHOTO_LOOKUP_FIELDS = {"photo": 1}


async def extract_embeddings(images: list[UploadFile]):
    """Return (embeddings, portrait, rejected_count)."""
    assert engine is not None
    embeddings, portrait, rejected = [], "", 0
    for up in images:
        data = await up.read()
        if not data:
            continue
        try:
            img = decode_image(data)
        except ValueError:
            rejected += 1
            continue
        faces = engine.analyse(img)
        if not faces:
            rejected += 1
            continue
        face = faces[0]  # largest
        embeddings.append(face.embedding)
        if not portrait:
            portrait = portrait_data_uri(img, face.box)
    return embeddings, portrait, rejected


# ───────────────────────── routes ─────────────────────────

@app.get("/api/health")
def health():
    try:
        db.client.admin.command("ping")
        mongo = True
    except Exception:
        mongo = False
    return {
        "ok": True,
        "mongo": mongo,
        "model": "YuNet + SFace (OpenCV)",
        "embeddings": gallery.size,
        "threshold": MATCH_THRESHOLD,
        "auth_required": auth.AUTH_ENABLED,
    }


class LoginBody(BaseModel):
    password: str


@app.post("/api/login")
def login(body: LoginBody):
    if not auth.AUTH_ENABLED:
        return {"token": "", "expires": 0, "auth_required": False}
    if not auth.check_password(body.password):
        raise HTTPException(401, "Incorrect password")
    token, exp = auth.issue_token()
    return {"token": token, "expires": exp, "auth_required": True}


@app.get("/")
def root():
    return {"service": "SFRS API", "docs": "/docs"}


@app.get("/api/students")
def list_students(q: str = "", department: str = ""):
    filt: dict = {}
    if q.strip():
        rx = {"$regex": re.escape(q.strip()), "$options": "i"}
        filt["$or"] = [{"name": rx}, {"student_id": rx}, {"email": rx}]
    if department:
        filt["department"] = department
    docs = db.students.find(filt, STUDENT_PROJECTION).sort("name", 1)
    return [public_student(d) for d in docs]


@app.get("/api/departments")
def departments():
    return sorted(d for d in db.students.distinct("department") if d)


@app.post("/api/students", status_code=201)
async def create_student(
    name: str = Form(...),
    student_id: str = Form(...),
    email: str = Form(""),
    phone: str = Form(""),
    department: str = Form(""),
    year: str = Form(""),
    section: str = Form(""),
    images: list[UploadFile] = File(...),
):
    name, student_id = name.strip(), student_id.strip().upper()
    if not name or not student_id:
        raise HTTPException(422, "Name and student ID are required")
    if db.students.find_one({"student_id": student_id}, {"_id": 1}):
        raise HTTPException(409, f"Student ID {student_id} is already registered")

    embeddings, portrait, rejected = await extract_embeddings(images)
    if not embeddings:
        raise HTTPException(422, "No face detected. Use a well-lit, front-facing photo.")

    # Guard against enrolling the same face twice under different IDs
    for emb in embeddings:
        owner, score = gallery.match(emb)
        if owner and score >= DUPLICATE_THRESHOLD:
            existing = db.students.find_one({"_id": ObjectId(owner)}, {"name": 1, "student_id": 1})
            if existing:
                raise HTTPException(
                    409,
                    f"This face is already enrolled as {existing['name']} ({existing['student_id']})",
                )

    doc = {
        "name": name,
        "student_id": student_id,
        "email": email.strip(),
        "phone": phone.strip(),
        "department": department.strip(),
        "year": year.strip(),
        "section": section.strip().upper(),
        "photo": portrait,
        "embeddings": [e.tolist() for e in embeddings],
        "samples": len(embeddings),
        "scan_count": 0,
        "last_seen": None,
        "created_at": now(),
    }
    try:
        res = db.students.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, f"Student ID {student_id} is already registered")
    refresh_gallery()
    doc["_id"] = res.inserted_id
    return {**public_student(doc), "rejected_images": rejected}


@app.get("/api/students/{sid}")
def get_student(sid: str):
    doc = db.students.find_one({"_id": oid(sid)}, STUDENT_PROJECTION)
    if not doc:
        raise HTTPException(404, "Student not found")
    history = db.scans.find({"student_ref": sid}).sort("ts", -1).limit(60)
    return {**public_student(doc), "history": [public_scan(s) for s in history]}


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    year: Optional[str] = None
    section: Optional[str] = None


@app.patch("/api/students/{sid}")
def update_student(sid: str, body: StudentUpdate):
    changes = {k: v.strip() for k, v in body.model_dump(exclude_none=True).items()}
    if "name" in changes and not changes["name"]:
        raise HTTPException(422, "Name cannot be empty")
    if "section" in changes:
        changes["section"] = changes["section"].upper()
    if changes:
        r = db.students.update_one({"_id": oid(sid)}, {"$set": changes})
        if not r.matched_count:
            raise HTTPException(404, "Student not found")
        if "name" in changes or "department" in changes:
            db.scans.update_many(
                {"student_ref": sid},
                {"$set": {k: changes[k] for k in ("name", "department") if k in changes}},
            )
    return get_student(sid)


@app.post("/api/students/{sid}/faces")
async def add_faces(sid: str, images: list[UploadFile] = File(...)):
    if not db.students.find_one({"_id": oid(sid)}, {"_id": 1}):
        raise HTTPException(404, "Student not found")
    embeddings, _, rejected = await extract_embeddings(images)
    if not embeddings:
        raise HTTPException(422, "No face detected in the uploaded images")
    db.students.update_one(
        {"_id": oid(sid)},
        {"$push": {"embeddings": {"$each": [e.tolist() for e in embeddings]}},
         "$inc": {"samples": len(embeddings)}},
    )
    refresh_gallery()
    return {"added": len(embeddings), "rejected_images": rejected}


@app.delete("/api/students/{sid}", status_code=204)
def delete_student(sid: str):
    r = db.students.delete_one({"_id": oid(sid)})
    if not r.deleted_count:
        raise HTTPException(404, "Student not found")
    db.scans.delete_many({"student_ref": sid})
    refresh_gallery()


@app.post("/api/recognize")
async def recognize(image: UploadFile = File(...), log: bool = Query(True)):
    assert engine is not None
    try:
        img = decode_image(await image.read())
    except ValueError:
        raise HTTPException(400, "Invalid image")

    h, w = img.shape[:2]
    faces = engine.analyse(img)
    results = []
    ts = now()
    for face in faces[:5]:
        owner, score = gallery.match(face.embedding)
        entry = {"box": list(face.box), "score": round(score, 3), "student": None, "logged": False}
        if owner and score >= MATCH_THRESHOLD:
            doc = db.students.find_one({"_id": ObjectId(owner)}, STUDENT_PROJECTION)
            if doc:
                entry["student"] = public_student(doc)
                last = _last_logged.get(owner)
                if log and (not last or (ts - last).total_seconds() > RELOG_SECONDS):
                    _last_logged[owner] = ts
                    db.scans.insert_one({
                        "status": "recognized",
                        "student_ref": owner,
                        "student_id": doc["student_id"],
                        "name": doc["name"],
                        "department": doc.get("department", ""),
                        "score": score,
                        "ts": ts,
                    })
                    db.students.update_one(
                        {"_id": doc["_id"]}, {"$set": {"last_seen": ts}, "$inc": {"scan_count": 1}}
                    )
                    entry["logged"] = True
                    entry["student"]["last_seen"] = iso(ts)
                entry["first_today"] = db.scans.count_documents(
                    {"student_ref": owner, "ts": {"$gte": local_midnight_utc()}}
                ) <= 1 and entry["logged"]
        elif log:
            last = _last_logged.get("__unknown__")
            if not last or (ts - last).total_seconds() > UNKNOWN_LOG_SECONDS:
                _last_logged["__unknown__"] = ts
                db.scans.insert_one({"status": "unknown", "score": score, "ts": ts})
        results.append(entry)
    return {"width": w, "height": h, "faces": results, "threshold": MATCH_THRESHOLD}


def attach_photos(scans: list[dict]) -> list[dict]:
    refs = {s["student_ref"] for s in scans if s.get("student_ref")}
    photos = {
        str(d["_id"]): d.get("photo", "")
        for d in db.students.find({"_id": {"$in": [ObjectId(r) for r in refs]}}, SCAN_PHOTO_LOOKUP_FIELDS)
    }
    for s in scans:
        if s.get("student_ref"):
            s["photo"] = photos.get(s["student_ref"], "")
    return scans


@app.get("/api/scans")
def list_scans(status: str = "", limit: int = Query(100, le=500), today: bool = False):
    filt: dict = {}
    if status in ("recognized", "unknown"):
        filt["status"] = status
    if today:
        filt["ts"] = {"$gte": local_midnight_utc()}
    docs = list(db.scans.find(filt).sort("ts", -1).limit(limit))
    return [public_scan(s) for s in attach_photos(docs)]


@app.get("/api/stats")
def stats():
    today = local_midnight_utc()
    total = db.students.count_documents({})
    present = len(db.scans.distinct("student_ref", {"status": "recognized", "ts": {"$gte": today}}))
    scans_today = db.scans.count_documents({"ts": {"$gte": today}})
    unknown_today = db.scans.count_documents({"status": "unknown", "ts": {"$gte": today}})

    # last 7 local days, unique students per day
    start = local_midnight_utc(6)
    per_day: dict[str, set] = defaultdict(set)
    for s in db.scans.find({"status": "recognized", "ts": {"$gte": start}}, {"student_ref": 1, "ts": 1}):
        per_day[s["ts"].astimezone().date().isoformat()].add(s["student_ref"])
    week = []
    for i in range(6, -1, -1):
        day = (datetime.now().astimezone() - timedelta(days=i)).date()
        week.append({"date": day.isoformat(), "present": len(per_day.get(day.isoformat(), ()))})

    # department roster vs present today
    present_ids = set(db.scans.distinct("student_ref", {"status": "recognized", "ts": {"$gte": today}}))
    dept: dict[str, dict] = defaultdict(lambda: {"total": 0, "present": 0})
    for d in db.students.find({}, {"department": 1}):
        key = d.get("department") or "Unassigned"
        dept[key]["total"] += 1
        if str(d["_id"]) in present_ids:
            dept[key]["present"] += 1
    departments = sorted(
        ({"name": k, **v} for k, v in dept.items()), key=lambda x: x["total"], reverse=True
    )

    recent = attach_photos(list(db.scans.find().sort("ts", -1).limit(8)))
    return {
        "total_students": total,
        "present_today": present,
        "scans_today": scans_today,
        "unknown_today": unknown_today,
        "week": week,
        "departments": departments,
        "recent": [public_scan(s) for s in recent],
    }
