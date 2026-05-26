import csv
import os
from pathlib import Path
from typing import Any, Dict, List

from pymongo import MongoClient
from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

# Install: pip install pymongo zenml
#
# How to get your LinkedIn data:
# 1. Go to LinkedIn → Me (profile icon) → Settings & Privacy
# 2. Data Privacy → Get a copy of your data
# 3. Select: Profile, Positions, Education, Certifications, Skills
# 4. LinkedIn will email you a download link (can take up to 24h)
# 5. Extract the ZIP into ./linkedin/


def _read_csv(filepath: Path) -> List[Dict[str, str]]:
    if not filepath.exists():
        logger.warning(f"File not found, skipping: {filepath}")
        return []
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@step
def extract_linkedin_data(data_dir: str) -> Dict[str, Any]:
    """Read LinkedIn export CSVs and combine into a single profile dict."""
    root = Path(data_dir)

    profile_rows = _read_csv(root / "Profile.csv")
    profile = profile_rows[0] if profile_rows else {}

    data: Dict[str, Any] = {
        "first_name": profile.get("First Name", ""),
        "last_name": profile.get("Last Name", ""),
        "headline": profile.get("Headline", ""),
        "summary": profile.get("Summary", ""),
        "location": profile.get("Geo Location", ""),
        "industry": profile.get("Industry", ""),
        "websites": profile.get("Websites", ""),
        "twitter_handles": profile.get("Twitter Handles", ""),
        "positions": [
            {
                "title": row.get("Title", ""),
                "company": row.get("Company Name", ""),
                "description": row.get("Description", ""),
                "location": row.get("Location", ""),
                "started_on": row.get("Started On", ""),
                "finished_on": row.get("Finished On", ""),
            }
            for row in _read_csv(root / "Positions.csv")
        ],
        "education": [
            {
                "school": row.get("School Name", ""),
                "degree": row.get("Degree Name", ""),
                "notes": row.get("Notes", ""),
                "activities": row.get("Activities", ""),
                "started_on": row.get("Start Date", ""),
                "finished_on": row.get("End Date", ""),
            }
            for row in _read_csv(root / "Education.csv")
        ],
        "certifications": [
            {
                "name": row.get("Name", ""),
                "authority": row.get("Authority", ""),
                "url": row.get("Url", ""),
                "license_number": row.get("License Number", ""),
                "started_on": row.get("Started On", ""),
                "finished_on": row.get("Finished On", ""),
            }
            for row in _read_csv(root / "Certifications.csv")
        ],
        "skills": [
            row.get("Name", "") for row in _read_csv(root / "Skills.csv")
        ],
    }

    logger.info(
        f"Extracted: {data['first_name']} {data['last_name']} — "
        f"{len(data['positions'])} positions, "
        f"{len(data['education'])} education, "
        f"{len(data['certifications'])} certifications, "
        f"{len(data['skills'])} skills"
    )
    return data


@step
def extract_linkedin_messages(data_dir: str) -> List[Dict[str, Any]]:
    """Read Messages.csv and group into conversation threads."""
    from collections import defaultdict
    from datetime import datetime

    root = Path(data_dir)
    raw = _read_csv(root / "messages.csv")
    if not raw:
        logger.warning("No messages found")
        return []

    convos: Dict[str, list] = defaultdict(list)
    for row in raw:
        conv_id = row.get("CONVERSATION ID", "")
        convos[conv_id].append({
            "from": row.get("FROM", ""),
            "sender_profile_url": row.get("SENDER PROFILE URL", ""),
            "date": row.get("DATE", ""),
            "subject": row.get("SUBJECT", ""),
            "content": row.get("CONTENT", ""),
        })

    conversations = []
    for conv_id, msgs in convos.items():
        msgs.sort(key=lambda m: m["date"])
        conversations.append({
            "conversation_id": conv_id,
            "conversation_title": raw[0].get("CONVERSATION TITLE", ""),
            "participants": list({m["from"] for m in msgs if m["from"]}),
            "message_count": len(msgs),
            "first_message_date": msgs[0]["date"],
            "last_message_date": msgs[-1]["date"],
            "messages": msgs,
        })

    logger.info(f"Extracted {len(conversations)} conversations with {len(raw)} total messages")
    return conversations


@step(enable_cache=False)
def store_messages_in_mongodb(conversations: List[Dict[str, Any]]) -> str:
    """Store message conversations in MongoDB for future training and search."""
    mongo_host = os.environ.get("MONGO_HOST", "localhost")
    mongo_port = int(os.environ.get("MONGO_PORT", "27017"))
    mongo_user = os.environ.get("MONGO_USER", "llm_engineering")
    mongo_pass = os.environ.get("MONGO_PASS", "llm_engineering")
    db_name = os.environ.get("MONGO_DB", "linkedin_warehouse")

    client = MongoClient(
        host=mongo_host,
        port=mongo_port,
        username=mongo_user,
        password=mongo_pass,
        authSource="admin",
    )
    collection = client[db_name]["messages"]

    inserted = 0
    updated = 0
    for convo in conversations:
        result = collection.replace_one(
            {"conversation_id": convo["conversation_id"]},
            convo,
            upsert=True,
        )
        if result.modified_count:
            updated += 1
        else:
            inserted += 1

    msg = f"Messages: {inserted} inserted, {updated} updated across {len(conversations)} conversations"
    logger.info(msg)
    client.close()
    return msg


@step(enable_cache=False)
def store_in_mongodb(profile: Dict[str, Any]) -> str:
    """Upsert the profile into MongoDB (avoids duplicates on re-runs)."""
    mongo_host = os.environ.get("MONGO_HOST", "localhost")
    mongo_port = int(os.environ.get("MONGO_PORT", "27017"))
    mongo_user = os.environ.get("MONGO_USER", "llm_engineering")
    mongo_pass = os.environ.get("MONGO_PASS", "llm_engineering")
    db_name = os.environ.get("MONGO_DB", "linkedin_warehouse")
    collection_name = os.environ.get("MONGO_COLLECTION", "profiles")

    client = MongoClient(
        host=mongo_host,
        port=mongo_port,
        username=mongo_user,
        password=mongo_pass,
        authSource="admin",
    )
    db = client[db_name]
    collection = db[collection_name]

    key = f"{profile['first_name']}_{profile['last_name']}".lower()
    result = collection.replace_one(
        {"_key": key},
        {**profile, "_key": key},
        upsert=True,
    )

    action = "Updated" if result.modified_count else "Inserted"
    msg = f"{action} profile for {key} in {db_name}.{collection_name}"
    logger.info(msg)

    client.close()
    return msg


@pipeline
def linkedin_to_mongodb_pipeline(data_dir: str):
    profile = extract_linkedin_data(data_dir=data_dir)
    store_in_mongodb(profile=profile)
    conversations = extract_linkedin_messages(data_dir=data_dir)
    store_messages_in_mongodb(conversations=conversations)


if __name__ == "__main__":
    linkedin_export_dir = os.environ.get("LINKEDIN_DATA_DIR", "./linkedin")

    run = linkedin_to_mongodb_pipeline(data_dir=linkedin_export_dir)
    logger.info("Pipeline finished")
