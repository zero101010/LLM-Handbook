# GitHub Data Pipeline Proposal

## Goal

Extract GitHub data to train a model that acts like you. This complements the LinkedIn pipeline (professional identity) with how you **think, communicate, code, and work**.

---

## Data to Extract (by training value)

### Tier 1 — High Signal (your voice & thinking)

| Data | Why |
|---|---|
| **PR descriptions & review comments** | Shows how you explain decisions, critique code, and reason about tradeoffs |
| **Issue comments** | How you communicate problems, ask questions, propose solutions |
| **Commit messages** | Your writing style in technical context, how you summarize work |
| **README / docs you authored** | Long-form writing style, how you explain things |

### Tier 2 — Medium Signal (your patterns & expertise)

| Data | Why |
|---|---|
| **Repositories metadata** | Topics, descriptions, languages — maps your interests & expertise |
| **Code contributions (diffs)** | Coding style, patterns you favor, languages you use |
| **Stars & watched repos** | What you find interesting — shapes the model's "taste" |
| **Gists** | Snippets you thought worth sharing |

### Tier 3 — Context (supporting data)

| Data | Why |
|---|---|
| **Profile bio & social links** | Identity, how you present yourself |
| **Contribution activity timeline** | Work patterns, focus areas over time |
| **Organizations** | Professional context |

---

## MongoDB Structure

### Why multiple collections?

The LinkedIn pipeline uses a single `profiles` collection with embedded arrays. GitHub data is much larger and more relational, so we use **one collection per entity type**:

1. **BSON 16MB limit** — a prolific user can exceed this with all commits embedded in one doc
2. **Incremental updates** — re-sync just commits or just PRs without reprocessing everything
3. **Training flexibility** — weight different data sources differently during fine-tuning
4. **Queryability** — filter by date, repo, or type easily

### Collections

```
Database: linkedin_warehouse

Collections:
├── profiles                  # (existing) LinkedIn data
├── github_profile            # 1 doc per user — bio, orgs, stats
├── github_repositories       # 1 doc per repo
├── github_commits            # 1 doc per commit (your commits only)
├── github_pull_requests      # 1 doc per PR (authored + reviewed)
├── github_issues             # 1 doc per issue interaction
├── github_gists              # 1 doc per gist
└── github_stars              # 1 doc per starred repo
```

### Document Schemas

#### github_profile

```json
{
    "_key": "github_username",
    "username": "string",
    "name": "string",
    "bio": "string",
    "company": "string",
    "location": "string",
    "blog": "string",
    "organizations": [
        {"name": "string", "description": "string"}
    ],
    "public_repos_count": "int",
    "followers": "int",
    "following": "int",
    "created_at": "string (ISO date)",
    "fetched_at": "datetime"
}
```

#### github_repositories

```json
{
    "_key": "username/repo_name",
    "owner": "string",
    "name": "string",
    "full_name": "string",
    "description": "string",
    "language": "string (primary)",
    "languages": {"Python": 45000, "Shell": 1200},
    "topics": ["string"],
    "is_fork": "bool",
    "is_owner": "bool (you own it vs just contributed)",
    "stars": "int",
    "forks": "int",
    "readme_content": "string (only for your own repos)",
    "created_at": "string (ISO date)",
    "updated_at": "string (ISO date)",
    "fetched_at": "datetime"
}
```

#### github_commits

```json
{
    "_key": "repo_full_name/sha_short",
    "repo": "string",
    "sha": "string",
    "message": "string  <-- key training signal",
    "files_changed": "int",
    "additions": "int",
    "deletions": "int",
    "files": [
        {"filename": "string", "patch": "string (the diff)"}
    ],
    "authored_at": "string (ISO date)",
    "fetched_at": "datetime"
}
```

> **Note on diffs:** Storing patch/diff data per commit is the most storage-heavy part but very valuable for learning coding style. This can be toggled off if storage is a concern.

#### github_pull_requests

```json
{
    "_key": "repo_full_name/pr_number",
    "repo": "string",
    "number": "int",
    "title": "string",
    "body": "string  <-- key training signal",
    "state": "string (open/closed/merged)",
    "role": "string (author | reviewer | commenter)",
    "comments": [
        {
            "author": "string",
            "body": "string  <-- key training signal",
            "created_at": "string (ISO date)",
            "is_review_comment": "bool",
            "diff_context": "string (the code being commented on)"
        }
    ],
    "labels": ["string"],
    "created_at": "string (ISO date)",
    "merged_at": "string (ISO date)",
    "fetched_at": "datetime"
}
```

> **Why comments are embedded:** PR comments are always read together with the PR and rarely exceed a few hundred per item — no risk of hitting the 16MB limit.

#### github_issues

```json
{
    "_key": "repo_full_name/issue_number",
    "repo": "string",
    "number": "int",
    "title": "string",
    "body": "string",
    "role": "string (author | commenter)",
    "comments": [
        {
            "author": "string",
            "body": "string",
            "created_at": "string (ISO date)"
        }
    ],
    "labels": ["string"],
    "state": "string",
    "created_at": "string (ISO date)",
    "fetched_at": "datetime"
}
```

#### github_stars

```json
{
    "_key": "username/starred_repo_full_name",
    "repo_full_name": "string",
    "description": "string",
    "language": "string",
    "topics": ["string"],
    "starred_at": "string (ISO date)",
    "fetched_at": "datetime"
}
```

#### github_gists

```json
{
    "_key": "gist_id",
    "description": "string",
    "files": [
        {"filename": "string", "language": "string", "content": "string"}
    ],
    "public": "bool",
    "comments": "int",
    "created_at": "string (ISO date)",
    "updated_at": "string (ISO date)",
    "fetched_at": "datetime"
}
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| `_key` on every doc | Consistent with LinkedIn pipeline, enables idempotent upserts |
| `fetched_at` on every doc | Enables incremental syncs ("give me everything since last fetch") |
| `role` field on PRs/issues | Critical for training — the model needs to know when **you** wrote something vs someone else |
| Comments embedded in parent | Always read together, no 16MB risk |
| Diffs stored in commits | Most valuable for coding style, can be toggled off |
| Separate collections per entity | Avoids 16MB limit, enables independent sync and flexible training |

### Recommended Indexes

```javascript
// On every collection
{"_key": 1}           // unique, for upserts
{"fetched_at": 1}     // incremental sync queries

// On commits, PRs, issues
{"repo": 1}           // filter by repository

// On commits specifically
{"authored_at": -1}   // recent commits first
```

---

## Open Questions (need your input)

1. **GitHub username** — hardcoded or read from env var?
2. **Authentication** — GitHub personal access token via env var? (needed for private repos, higher rate limits, and accessing review comments)
3. **Scope** — all repos you've ever touched, or just ones you own? How far back for commits?
4. **Diffs** — store full patch data per commit, or skip to save storage?
5. **Database name** — keep `linkedin_warehouse` or rename to something broader like `digital_twin_warehouse`?
