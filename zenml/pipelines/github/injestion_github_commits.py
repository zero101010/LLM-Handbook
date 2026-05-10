import hashlib
import os
import re
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

from zenml import pipeline, step


try:
    from zenml import log_metadata
except ImportError:
    log_metadata = None


load_dotenv()


# =========================
# Environment config
# =========================

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "github_warehouse")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "github_commits")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "github_commits")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))


# =========================
# Cleaning helpers
# =========================

NOISY_FILE_PATTERNS = [
    ".gitignore",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "terraform.tfstate",
    "terraform.tfstate.backup",
]

SECRET_PATTERNS = [
    (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        "[REDACTED_AWS_ACCESS_KEY]",
    ),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(r'(?i)(password\s*=\s*)["\'][^"\']+["\']'),
        r'\1"[REDACTED_PASSWORD]"',
    ),
    (
        re.compile(r'(?i)(token\s*=\s*)["\'][^"\']+["\']'),
        r'\1"[REDACTED_TOKEN]"',
    ),
    (
        re.compile(r"mongodb(\+srv)?://[^:\s]+:[^@\s]+@"),
        "mongodb://[REDACTED_CREDENTIALS]@",
    ),
]


def stable_id(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def serialize_mongo_doc(doc: dict[str, Any]) -> dict[str, Any]:
    doc["_id"] = str(doc["_id"])
    return doc


def normalize_message(message: str) -> str:
    message = safe_str(message)
    message = re.sub(r"\s+", " ", message)
    return message.strip()


def redact_secrets(text: str) -> str:
    if not text:
        return ""

    cleaned = text

    for pattern, replacement in SECRET_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)

    return cleaned


def get_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext.lower()


def detect_language(filename: str, patch: str = "") -> str:
    filename_lower = filename.lower()
    ext = get_extension(filename)

    if ext == ".tf":
        return "terraform"

    if ext in [".yaml", ".yml"]:
        patch_lower = patch.lower()

        if "networking.istio.io" in patch_lower or "kind: gateway" in patch_lower:
            return "istio_yaml"

        if "apiversion:" in patch_lower or "kind:" in patch_lower:
            return "kubernetes_yaml"

        return "yaml"

    if ext == ".sh":
        return "shell"

    if ext == ".py":
        return "python"

    if ext in [".js", ".jsx"]:
        return "javascript"

    if ext in [".ts", ".tsx"]:
        return "typescript"

    if ext == ".md":
        return "markdown"

    if filename_lower.endswith(".gitignore"):
        return "gitignore"

    return "unknown"


def is_noisy_file(filename: str) -> bool:
    filename_lower = filename.lower()

    if filename_lower in NOISY_FILE_PATTERNS:
        return True

    if filename_lower.endswith(".lock"):
        return True

    if "/dist/" in filename_lower:
        return True

    if "/build/" in filename_lower:
        return True

    if "/node_modules/" in filename_lower:
        return True

    if "/vendor/" in filename_lower:
        return True

    return False


def detect_change_type(patch: str) -> str:
    if not patch:
        return "unknown"

    if patch.startswith("@@ -0,0"):
        return "added"

    if re.search(r"@@ -\d+,\d+ \+0,0 @@", patch):
        return "deleted"

    return "modified"


def detect_topics(filename: str, patch: str, message: str) -> list[str]:
    text = f"{filename}\n{message}\n{patch}".lower()
    topics = set()

    topic_rules = {
        "terraform": [
            "terraform",
            ".tf",
            "resource ",
            "provider ",
            "variable ",
            "output ",
        ],
        "gke": [
            "google_container_cluster",
            "google_container_node_pool",
            "gke",
            "container clusters",
        ],
        "google_cloud": [
            "google_",
            "gcloud",
            "gcp",
            "google cloud",
        ],
        "kubernetes": [
            "apiversion:",
            "kind:",
            "kubectl",
            "deployment",
            "service",
        ],
        "helm": [
            "helm",
            "chart",
            "values.yaml",
        ],
        "istio": [
            "istio",
            "virtualservice",
            "gateway",
            "networking.istio.io",
        ],
        "nginx": [
            "nginx",
        ],
        "rancher": [
            "rancher",
        ],
        "shell": [
            ".sh",
            "#!/bin/bash",
            "bash",
            "sh ",
        ],
        "local_exec": [
            "local-exec",
            "provisioner",
        ],
        "docker": [
            "dockerfile",
            "docker-compose",
            "docker ",
        ],
        "github_actions": [
            ".github/workflows",
            "github actions",
        ],
        "ci_cd": [
            "ci/cd",
            "pipeline",
            "workflow",
        ],
    }

    for topic, keywords in topic_rules.items():
        if any(keyword in text for keyword in keywords):
            topics.add(topic)

    return sorted(topics)


def summarize_patch(filename: str, language: str, patch: str) -> str:
    patch = patch or ""

    added_lines = []
    removed_lines = []

    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue

        if line.startswith("+"):
            added_lines.append(line[1:].strip())

        elif line.startswith("-"):
            removed_lines.append(line[1:].strip())

    important_added = [
        line
        for line in added_lines
        if line and not line.startswith("#") and not line.startswith("//")
    ]

    summary_parts = [
        f"File {filename} changed.",
        f"Detected language: {language}.",
    ]

    added_text = "\n".join(added_lines)

    if language == "terraform":
        resources = re.findall(
            r'resource\s+"([^"]+)"\s+"([^"]+)"',
            added_text,
        )
        providers = re.findall(
            r'provider\s+"([^"]+)"',
            added_text,
        )
        variables = re.findall(
            r'variable\s+"([^"]+)"',
            added_text,
        )
        outputs = re.findall(
            r'output\s+"([^"]+)"',
            added_text,
        )

        if resources:
            resource_names = [f"{rtype}.{name}" for rtype, name in resources]
            summary_parts.append(
                f"Adds or changes Terraform resources: {', '.join(resource_names)}."
            )

        if providers:
            summary_parts.append(
                f"Configures Terraform providers: {', '.join(sorted(set(providers)))}."
            )

        if variables:
            summary_parts.append(
                f"Defines Terraform variables: {', '.join(sorted(set(variables)))}."
            )

        if outputs:
            summary_parts.append(
                f"Defines Terraform outputs: {', '.join(sorted(set(outputs)))}."
            )

    elif language in ["kubernetes_yaml", "istio_yaml", "yaml"]:
        kinds = re.findall(
            r"kind:\s*([A-Za-z0-9_-]+)",
            added_text,
        )
        names = re.findall(
            r"name:\s*([A-Za-z0-9_.-]+)",
            added_text,
        )

        if kinds:
            summary_parts.append(
                f"Adds or changes Kubernetes/Istio resource kinds: {', '.join(sorted(set(kinds)))}."
            )

        if names:
            summary_parts.append(
                f"Resource names include: {', '.join(sorted(set(names)))}."
            )

    elif language == "shell":
        commands = important_added[:8]

        if commands:
            summary_parts.append(
                f"Adds shell commands: {'; '.join(commands)}."
            )

    if important_added:
        preview = "; ".join(important_added[:10])
        summary_parts.append(f"Important added lines: {preview}.")

    if removed_lines:
        summary_parts.append(f"Removed {len(removed_lines)} lines.")

    if added_lines:
        summary_parts.append(f"Added {len(added_lines)} lines.")

    return " ".join(summary_parts)


def clean_commit(raw_commit: dict) -> dict:
    commit_id = safe_str(raw_commit.get("_id"))
    commit_key = safe_str(raw_commit.get("_key"))
    repo = safe_str(raw_commit.get("repo"))
    sha = safe_str(raw_commit.get("sha"))
    short_sha = sha[:12] if sha else ""
    message = normalize_message(raw_commit.get("message"))

    cleaned_files = []

    for file_item in raw_commit.get("files", []):
        filename = safe_str(file_item.get("filename"))
        raw_patch = safe_str(file_item.get("patch"))

        if not filename:
            continue

        redacted_patch = redact_secrets(raw_patch)
        language = detect_language(filename, redacted_patch)
        change_type = detect_change_type(redacted_patch)
        noisy = is_noisy_file(filename)
        topics = detect_topics(filename, redacted_patch, message)
        patch_summary = summarize_patch(filename, language, redacted_patch)

        cleaned_files.append(
            {
                "filename": filename,
                "extension": get_extension(filename),
                "language": language,
                "change_type": change_type,
                "is_noisy": noisy,
                "topics": topics,
                "patch_summary": patch_summary,
                "raw_patch": redacted_patch,
            }
        )

    all_topics = sorted(
        set(
            topic
            for file_item in cleaned_files
            for topic in file_item.get("topics", [])
        )
    )

    return {
        "commit_id": commit_id,
        "commit_key": commit_key,
        "repo": repo,
        "sha": sha,
        "short_sha": short_sha,
        "message": message,
        "message_normalized": message.lower(),
        "first_word": message.split(" ")[0] if message else "",
        "files_changed": int(raw_commit.get("files_changed") or len(cleaned_files)),
        "additions": int(raw_commit.get("additions") or 0),
        "deletions": int(raw_commit.get("deletions") or 0),
        "authored_at": safe_str(raw_commit.get("authored_at")),
        "fetched_at": safe_str(raw_commit.get("fetched_at")),
        "topics": all_topics,
        "files": cleaned_files,
        "source": "mongodb",
    }


def build_commit_chunks(cleaned_commit: dict) -> list[dict]:
    repo = cleaned_commit["repo"]
    sha = cleaned_commit["sha"]
    short_sha = cleaned_commit["short_sha"]
    commit_key = cleaned_commit["commit_key"]
    message = cleaned_commit["message"]

    base_metadata = {
        "document_type": "github_commit",
        "commit_id": cleaned_commit["commit_id"],
        "commit_key": commit_key,
        "repo": repo,
        "sha": sha,
        "short_sha": short_sha,
        "message": message,
        "files_changed": cleaned_commit["files_changed"],
        "additions": cleaned_commit["additions"],
        "deletions": cleaned_commit["deletions"],
        "authored_at": cleaned_commit["authored_at"],
        "fetched_at": cleaned_commit["fetched_at"],
        "topics": cleaned_commit["topics"],
        "source": "mongodb",
        "embedding_model": EMBEDDING_MODEL,
    }

    chunks = []

    important_files = [
        file_item
        for file_item in cleaned_commit["files"]
        if not file_item["is_noisy"]
    ]

    file_list = "\n".join(
        f"- {file_item['filename']} ({file_item['language']}, {file_item['change_type']})"
        for file_item in important_files[:30]
    )

    commit_summary_text = f"""
Repository: {repo}
Commit SHA: {sha}
Commit message: {message}
Files changed: {cleaned_commit["files_changed"]}
Additions: {cleaned_commit["additions"]}
Deletions: {cleaned_commit["deletions"]}
Authored at: {cleaned_commit["authored_at"]}
Topics: {", ".join(cleaned_commit["topics"])}

Important changed files:
{file_list}
""".strip()

    chunks.append(
        {
            "id": stable_id(f"{commit_key}:commit_summary"),
            "text": commit_summary_text,
            "metadata": {
                **base_metadata,
                "chunk_type": "commit_summary",
            },
        }
    )

    message_style_text = f"""
Commit message example:
{message}

Commit message style:
The message starts with "{cleaned_commit["first_word"]}".
It is a descriptive commit message explaining the main implementation change.
Use this as a style reference when suggesting future commit messages.

Repository: {repo}
Topics: {", ".join(cleaned_commit["topics"])}
""".strip()

    chunks.append(
        {
            "id": stable_id(f"{commit_key}:commit_message_style"),
            "text": message_style_text,
            "metadata": {
                **base_metadata,
                "chunk_type": "commit_message_style",
                "first_word": cleaned_commit["first_word"],
                "message_style": "imperative_descriptive",
            },
        }
    )

    for index, file_item in enumerate(cleaned_commit["files"]):
        if file_item["is_noisy"]:
            continue

        filename = file_item["filename"]
        language = file_item["language"]
        change_type = file_item["change_type"]
        topics = file_item["topics"]

        file_text = f"""
Commit message:
{message}

Repository:
{repo}

File:
{filename}

Language:
{language}

Change type:
{change_type}

Topics:
{", ".join(topics)}

Patch summary:
{file_item["patch_summary"]}
""".strip()

        chunks.append(
            {
                "id": stable_id(f"{commit_key}:file_patch_summary:{index}:{filename}"),
                "text": file_text,
                "metadata": {
                    **base_metadata,
                    "chunk_type": "file_patch_summary",
                    "file_index": index,
                    "filename": filename,
                    "extension": file_item["extension"],
                    "language": language,
                    "change_type": change_type,
                    "file_topics": topics,
                    "raw_patch": file_item["raw_patch"],
                },
            }
        )

    return chunks


# =========================
# ZenML steps
# =========================

@step(enable_cache=False)
def load_github_commits_step(limit: int = 100) -> list[dict]:
    if not MONGO_URI:
        raise ValueError("MONGO_URI is missing from .env")

    try:
        mongo = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
        )

        mongo.admin.command("ping")

        collection = mongo[MONGO_DB][MONGO_COLLECTION]

        raw_commits = [
            serialize_mongo_doc(doc)
            for doc in collection.find({}).limit(limit)
        ]

        print(f"Loaded raw commits: {len(raw_commits)}")
        print(f"Mongo database: {MONGO_DB}")
        print(f"Mongo collection: {MONGO_COLLECTION}")

        if log_metadata:
            log_metadata(
                metadata={
                    "mongo_database": MONGO_DB,
                    "mongo_collection": MONGO_COLLECTION,
                    "raw_commits_loaded": len(raw_commits),
                }
            )

        return raw_commits

    except OperationFailure as e:
        print("MongoDB authentication failed.")
        print("Check MONGO_URI username, password, database, and authSource.")
        raise e

    except ServerSelectionTimeoutError as e:
        print("Could not connect to MongoDB.")
        print("Check if MongoDB is running and the host/port are correct.")
        raise e


@step
def clean_github_commits_step(raw_commits: list[dict]) -> list[dict]:
    cleaned_commits = []

    for raw_commit in raw_commits:
        cleaned_commits.append(clean_commit(raw_commit))

    total_files = sum(len(commit["files"]) for commit in cleaned_commits)

    noisy_files = sum(
        1
        for commit in cleaned_commits
        for file_item in commit["files"]
        if file_item["is_noisy"]
    )

    clean_files = total_files - noisy_files

    all_topics = sorted(
        set(
            topic
            for commit in cleaned_commits
            for topic in commit.get("topics", [])
        )
    )

    all_languages = sorted(
        set(
            file_item["language"]
            for commit in cleaned_commits
            for file_item in commit["files"]
        )
    )

    print(f"Cleaned commits: {len(cleaned_commits)}")
    print(f"Total files discovered: {total_files}")
    print(f"Noisy files detected: {noisy_files}")
    print(f"Files that can become chunks: {clean_files}")
    print(f"Detected topics: {all_topics}")
    print(f"Detected languages: {all_languages}")

    if log_metadata:
        log_metadata(
            metadata={
                "cleaned_commits": len(cleaned_commits),
                "total_files": total_files,
                "noisy_files": noisy_files,
                "clean_files": clean_files,
                "detected_topics": ", ".join(all_topics[:100]),
                "detected_languages": ", ".join(all_languages[:50]),
            }
        )

    return cleaned_commits


@step
def build_github_commit_chunks_step(cleaned_commits: list[dict]) -> list[dict]:
    chunks = []

    for commit in cleaned_commits:
        chunks.extend(build_commit_chunks(commit))

    chunk_type_counts = {}

    for chunk in chunks:
        chunk_type = chunk["metadata"]["chunk_type"]
        chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1

    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk type counts: {chunk_type_counts}")

    if log_metadata:
        log_metadata(
            metadata={
                "chunks_created": len(chunks),
                "commit_summary_chunks": chunk_type_counts.get("commit_summary", 0),
                "commit_message_style_chunks": chunk_type_counts.get(
                    "commit_message_style",
                    0,
                ),
                "file_patch_summary_chunks": chunk_type_counts.get(
                    "file_patch_summary",
                    0,
                ),
            }
        )

    return chunks


@step
def embed_github_commit_chunks_step(chunks: list[dict]) -> list[dict]:
    if not chunks:
        print("No chunks to embed.")
        return []

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

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

    print(f"Embedded chunks: {len(embedded_chunks)}")

    if log_metadata:
        log_metadata(
            metadata={
                "embedded_chunks": len(embedded_chunks),
                "embedding_model": EMBEDDING_MODEL,
                "vector_size": VECTOR_SIZE,
            }
        )

    return embedded_chunks


@step(enable_cache=False)
def upsert_github_chunks_to_qdrant_step(embedded_chunks: list[dict]) -> int:
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

    if log_metadata:
        log_metadata(
            metadata={
                "qdrant_collection": QDRANT_COLLECTION,
                "qdrant_points_upserted": len(points),
                "qdrant_url": QDRANT_URL,
            }
        )

    return len(points)


# =========================
# Pipeline
# =========================

@pipeline
def github_commits_to_qdrant_pipeline(limit: int = 100):
    raw_commits = load_github_commits_step(limit=limit)

    cleaned_commits = clean_github_commits_step(raw_commits)

    chunks = build_github_commit_chunks_step(cleaned_commits)

    embedded_chunks = embed_github_commit_chunks_step(chunks)

    upsert_github_chunks_to_qdrant_step(embedded_chunks)


if __name__ == "__main__":
    github_commits_to_qdrant_pipeline(limit=500)