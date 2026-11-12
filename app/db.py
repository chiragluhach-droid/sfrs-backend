import os

from pymongo import ASCENDING, DESCENDING, MongoClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "sfrs")

client = MongoClient(MONGO_URL, tz_aware=True, serverSelectionTimeoutMS=3000)
db = client[MONGO_DB]

students = db["students"]
scans = db["scans"]


def ensure_indexes() -> None:
    students.create_index([("student_id", ASCENDING)], unique=True)
    students.create_index([("name", ASCENDING)])
    students.create_index([("department", ASCENDING)])
    scans.create_index([("ts", DESCENDING)])
    scans.create_index([("student_ref", ASCENDING), ("ts", DESCENDING)])
