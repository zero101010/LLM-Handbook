from typing import Any

import hashlib
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

from zenml import pipeline, step


load_dotenv()


# =========================
# Config
# =========================

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "linkedin_warehouse")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "profiles")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "profiles")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))


# =========================
# Helpers
# =========================

def stable_id(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def serialize_mongo_doc(doc: dict) -> dict:
    """
    Converts MongoDB ObjectId and other non-JSON-safe values to strings.
    ZenML artifacts should be serializable between steps.
    """
    doc["_id"] = str(doc["_id"])
    return doc


def build_profile_chunks(profile: dict) -> list[dict]:
    profile_id = safe_str(profile.get("_id"))
    profile_key = safe_str(profile.get("_key"))

    first_name = safe_str(profile.get("first_name"))
    last_name = safe_str(profile.get("last_name"))
    full_name = f"{first_name} {last_name}".strip()

    industry = safe_str(profile.get("industry"))
    profile_location = safe_str(profile.get("location"))

    base_metadata = {
        "profile_id": profile_id,
        "profile_key": profile_key,
        "full_name": full_name,
        "industry": industry,
        "profile_location": profile_location,
        "source": "mongodb",
        "embedding_model": EMBEDDING_MODEL,
    }

    chunks = []

    profile_summary = f"""
Name: {full_name}
Headline: {safe_str(profile.get("headline"))}
Summary: {safe_str(profile.get("summary"))}
Location: {profile_location}
Industry: {industry}
Website: {safe_str(profile.get("websites"))}
""".strip()

    chunks.append(
        {
            "id": stable_id(f"{profile_id}:profile_summary"),
            "text": profile_summary,
            "metadata": {
                **base_metadata,
                "section": "profile_summary",
            },
        }
    )

    for index, position in enumerate(profile.get("positions", [])):
        title = safe_str(position.get("title"))
        company = safe_str(position.get("company"))
        description = safe_str(position.get("description"))
        started_on = safe_str(position.get("started_on"))
        finished_on = safe_str(position.get("finished_on")) or "Present"
        position_location = safe_str(position.get("location"))

        position_overview = f"""
{full_name} worked as {title} at {company}.
Period: {started_on} - {finished_on}
Location: {position_location}
""".strip()

        chunks.append(
            {
                "id": stable_id(f"{profile_id}:position_overview:{index}:{company}:{title}"),
                "text": position_overview,
                "metadata": {
                    **base_metadata,
                    "section": "position_overview",
                    "position_index": index,
                    "company": company,
                    "title": title,
                    "started_on": started_on,
                    "finished_on": finished_on,
                    "position_location": position_location,
                },
            }
        )

        # Split bullet-heavy descriptions into smaller semantic chunks.
        # Your descriptions use " - " between bullets, so we normalize them here.
        raw_bullets = [
            item.strip()
            for item in description.replace(" - ", "\n- ").split("\n- ")
            if item.strip()
        ]

        if not raw_bullets and description:
            raw_bullets = [description]

        for bullet_index, bullet in enumerate(raw_bullets):
            text = f"""
{full_name} worked as {title} at {company}.
Period: {started_on} - {finished_on}

Achievement/responsibility:
{bullet}
""".strip()

            chunks.append(
                {
                    "id": stable_id(
                        f"{profile_id}:position_bullet:{index}:{bullet_index}:{company}:{title}"
                    ),
                    "text": text,
                    "metadata": {
                        **base_metadata,
                        "section": "position_bullet",
                        "position_index": index,
                        "bullet_index": bullet_index,
                        "company": company,
                        "title": title,
                        "started_on": started_on,
                        "finished_on": finished_on,
                        "position_location": position_location,
                    },
                }
            )

    for index, education in enumerate(profile.get("education", [])):
        school = safe_str(education.get("school"))
        degree = safe_str(education.get("degree"))
        started_on = safe_str(education.get("started_on"))
        finished_on = safe_str(education.get("finished_on"))

        text = f"""
{full_name} studied at {school}.
Degree: {degree}
Started: {started_on}
Finished: {finished_on}
Notes: {safe_str(education.get("notes"))}
Activities: {safe_str(education.get("activities"))}
""".strip()

        chunks.append(
            {
                "id": stable_id(f"{profile_id}:education:{index}:{school}"),
                "text": text,
                "metadata": {
                    **base_metadata,
                    "section": "education",
                    "education_index": index,
                    "school": school,
                    "degree": degree,
                    "started_on": started_on,
                    "finished_on": finished_on,
                },
            }
        )

    for index, cert in enumerate(profile.get("certifications", [])):
        name = safe_str(cert.get("name"))
        authority = safe_str(cert.get("authority"))
        started_on = safe_str(cert.get("started_on"))
        finished_on = safe_str(cert.get("finished_on"))
        url = safe_str(cert.get("url"))
        license_number = safe_str(cert.get("license_number"))

        text = f"""
Certification: {name}
Authority: {authority}
Started: {started_on}
Finished: {finished_on}
License number: {license_number}
URL: {url}
""".strip()

        chunks.append(
            {
                "id": stable_id(f"{profile_id}:certification:{index}:{name}"),
                "text": text,
                "metadata": {
                    **base_metadata,
                    "section": "certification",
                    "certification_index": index,
                    "certification_name": name,
                    "authority": authority,
                    "started_on": started_on,
                    "finished_on": finished_on,
                    "url": url,
                    "license_number": license_number,
                },
            }
        )

    skills = profile.get("skills", [])

    skills_text = f"""
{full_name}'s skills include:
{", ".join(skills)}
""".strip()

    chunks.append(
        {
            "id": stable_id(f"{profile_id}:skills"),
            "text": skills_text,
            "metadata": {
                **base_metadata,
                "section": "skills",
                "skills": skills,
            },
        }
    )

    return chunks


# =========================
# ZenML steps
# =========================

@step(enable_cache=False)
def load_profiles_step() -> list[dict]:
    """
    Extracts profiles from MongoDB.

    Cache disabled because this step reads external data.
    Otherwise ZenML may reuse a previous run's artifact.
    """

    if not MONGO_URI:
        raise ValueError("MONGO_URI is missing from .env")

    try:
        mongo = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
        )

        mongo.admin.command("ping")

        db = mongo[MONGO_DB]
        collection = db[MONGO_COLLECTION]

        profiles = [serialize_mongo_doc(doc) for doc in collection.find({})]

        print(f"MongoDB connected successfully")
        print(f"Database: {MONGO_DB}")
        print(f"Collection: {MONGO_COLLECTION}")
        print(f"Profiles loaded: {len(profiles)}")

        return profiles

    except OperationFailure as e:
        print("MongoDB authentication failed.")
        print("Check MONGO_URI username, password, database, and authSource.")
        raise e

    except ServerSelectionTimeoutError as e:
        print("Could not connect to MongoDB.")
        print("Check if MongoDB is running and the host/port are correct.")
        raise e


@step
def build_chunks_step(profiles: list[dict]) -> list[dict]:
    """
    Converts MongoDB profiles into RAG-friendly chunks.
    """

    chunks = []

    for profile in profiles:
        profile_chunks = build_profile_chunks(profile)
        chunks.extend(profile_chunks)

    print(f"Profiles received: {len(profiles)}")
    print(f"Chunks created: {len(chunks)}")

    return chunks


@step
def embed_chunks_step(chunks: list[dict]) -> list[dict]:
    """
    Embeds all chunks using sentence-transformers/all-MiniLM-L6-v2.
    """

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    if not texts:
        print("No chunks to embed.")
        return []

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
    ).tolist()

    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append(
            {
                **chunk,
                "vector": vector,
            }
        )

    print(f"Chunks embedded: {len(embedded_chunks)}")

    return embedded_chunks


@step(enable_cache=False)
def upsert_to_qdrant_step(embedded_chunks: list[dict]) -> int:
    """
    Creates the Qdrant collection if needed and upserts embedded chunks.
    """

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    existing_collections = qdrant.get_collections().collections
    existing_names = [collection.name for collection in existing_collections]

    if QDRANT_COLLECTION not in existing_names:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created Qdrant collection: {QDRANT_COLLECTION}")
    else:
        print(f"Qdrant collection already exists: {QDRANT_COLLECTION}")

    if not embedded_chunks:
        print("No embedded chunks to upsert.")
        return 0

    points = []

    for chunk in embedded_chunks:
        payload = {
            **chunk["metadata"],
            "text": chunk["text"],
        }

        points.append(
            PointStruct(
                id=chunk["id"],
                vector=chunk["vector"],
                payload=payload,
            )
        )

    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    print(f"Points upserted into Qdrant: {len(points)}")

    return len(points)


# =========================
# ZenML pipeline
# =========================

@pipeline
def mongo_to_qdrant_ingestion_pipeline():
    profiles = load_profiles_step()
    chunks = build_chunks_step(profiles)
    embedded_chunks = embed_chunks_step(chunks)
    upsert_to_qdrant_step(embedded_chunks)


if __name__ == "__main__":
    mongo_to_qdrant_ingestion_pipeline()