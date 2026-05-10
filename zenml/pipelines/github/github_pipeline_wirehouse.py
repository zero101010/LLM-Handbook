import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from github import Auth, Github
from pymongo import MongoClient
from zenml import pipeline, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONGO_URI = os.environ.get(
    "MONGO_URI", "mongodb://llm_engineering:llm_engineering@localhost:27017"
)
MONGO_DB = os.environ.get("MONGO_DB", "github_warehouse")
EXCLUDED_REPOS = {"zero101010/api-rest-go"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gh_client() -> Github:
    """Build an authenticated PyGithub client from the ZenML GITHUB secret."""
    secret = Client().get_secret("GITHUB")
    token = secret.secret_values["secret"]
    return Github(auth=Auth.Token(token))


def _get_username() -> str:
    secret = Client().get_secret("GITHUB")
    return secret.secret_values["user_name"]


def _mongo_upsert(collection_name: str, docs: List[Dict[str, Any]]) -> int:
    """Upsert a list of documents into a MongoDB collection by _key."""
    if not docs:
        return 0
    client = MongoClient(MONGO_URI)
    col = client[MONGO_DB][collection_name]
    count = 0
    for doc in docs:
        result = col.replace_one({"_key": doc["_key"]}, doc, upsert=True)
        if result.upserted_id or result.modified_count:
            count += 1
    client.close()
    return count


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@step
def extract_github_profile() -> Dict[str, Any]:
    """Fetch GitHub profile and organizations."""
    gh = _gh_client()
    username = _get_username()
    user = gh.get_user(username)

    orgs = [
        {"name": org.login, "description": org.description or ""}
        for org in user.get_orgs()
    ]

    profile = {
        "_key": username,
        "username": username,
        "name": user.name or "",
        "bio": user.bio or "",
        "company": user.company or "",
        "location": user.location or "",
        "blog": user.blog or "",
        "organizations": orgs,
        "public_repos_count": user.public_repos,
        "followers": user.followers,
        "following": user.following,
        "created_at": user.created_at.isoformat() if user.created_at else "",
        "fetched_at": _now(),
    }
    logger.info(f"Extracted profile for {username} ({user.public_repos} repos)")
    gh.close()
    return profile


@step
def extract_github_repositories() -> List[Dict[str, Any]]:
    """Fetch all repos the user owns or has contributed to."""
    gh = _gh_client()
    username = _get_username()
    user = gh.get_user(username)
    repos = []

    for i, repo in enumerate(user.get_repos(type="all"), 1):
        if repo.name in EXCLUDED_REPOS:
            logger.info(f"[Repos] Skipping excluded repo: {repo.full_name}")
            continue
        logger.info(f"[Repos] Fetching {i}: {repo.full_name}")
        is_owner = repo.owner.login == username

        # Fetch README only for owned repos
        readme_content = ""
        if is_owner:
            try:
                readme_content = repo.get_readme().decoded_content.decode("utf-8")
            except Exception:
                pass

        # Fetch language breakdown
        try:
            languages = dict(repo.get_languages())
        except Exception:
            languages = {}

        repos.append(
            {
                "_key": repo.full_name,
                "owner": repo.owner.login,
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description or "",
                "language": repo.language or "",
                "languages": languages,
                "topics": repo.get_topics(),
                "is_fork": repo.fork,
                "is_owner": is_owner,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "readme_content": readme_content,
                "created_at": repo.created_at.isoformat() if repo.created_at else "",
                "updated_at": repo.updated_at.isoformat() if repo.updated_at else "",
                "fetched_at": _now(),
            }
        )

    logger.info(f"Extracted {len(repos)} repositories for {username}")
    gh.close()
    return repos


@step
def extract_github_commits() -> List[Dict[str, Any]]:
    """Fetch commits authored by the user across their own repos."""
    gh = _gh_client()
    username = _get_username()
    user = gh.get_user(username)
    commits = []

    for repo in user.get_repos(type="owner"):
        try:
            for commit in repo.get_commits(author=username):
                c = commit.commit
                files = []
                for f in commit.files or []:
                    files.append(
                        {
                            "filename": f.filename,
                            "patch": (f.patch or "")[:5000],  # cap large diffs
                        }
                    )

                commits.append(
                    {
                        "_key": f"{repo.full_name}/{commit.sha[:12]}",
                        "repo": repo.full_name,
                        "sha": commit.sha,
                        "message": c.message,
                        "files_changed": commit.stats.total if commit.stats else 0,
                        "additions": commit.stats.additions if commit.stats else 0,
                        "deletions": commit.stats.deletions if commit.stats else 0,
                        "files": files,
                        "authored_at": (
                            c.author.date.isoformat() if c.author and c.author.date else ""
                        ),
                        "fetched_at": _now(),
                    }
                )
        except Exception as e:
            logger.warning(f"Skipping commits for {repo.full_name}: {e}")

    logger.info(f"Extracted {len(commits)} commits for {username}")
    gh.close()
    return commits


@step
def extract_github_pull_requests() -> List[Dict[str, Any]]:
    """Fetch PRs authored or reviewed by the user."""
    gh = _gh_client()
    username = _get_username()
    prs = []

    # Search for PRs authored by user (last 3 years)
    cutoff = (datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 3)).strftime("%Y-%m-%d")
    logger.info(f"[PRs] Fetching authored PRs since {cutoff}")
    for i, issue in enumerate(gh.search_issues(f"type:pr author:{username} created:>={cutoff}"), 1):
        pr = issue.as_pull_request()
        repo_label = pr.base.repo.full_name if pr.base and pr.base.repo else "unknown"
        repo_short = repo_label.split("/")[-1] if "/" in repo_label else repo_label
        if repo_short in EXCLUDED_REPOS:
            logger.info(f"[PRs authored] Skipping excluded repo: {repo_label}#{pr.number}")
            continue
        logger.info(f"[PRs authored] {i}: {repo_label}#{pr.number} — {pr.title}")
        comments = []
        try:
            for comment in pr.get_issue_comments():
                comments.append(
                    {
                        "author": comment.user.login if comment.user else "",
                        "body": comment.body or "",
                        "created_at": comment.created_at.isoformat() if comment.created_at else "",
                        "is_review_comment": False,
                        "diff_context": "",
                    }
                )
            for comment in pr.get_review_comments():
                comments.append(
                    {
                        "author": comment.user.login if comment.user else "",
                        "body": comment.body or "",
                        "created_at": comment.created_at.isoformat() if comment.created_at else "",
                        "is_review_comment": True,
                        "diff_context": comment.diff_hunk or "",
                    }
                )
        except Exception as e:
            logger.warning(f"Could not fetch comments for PR {pr.html_url}: {e}")

        repo_name = pr.base.repo.full_name if pr.base and pr.base.repo else ""
        prs.append(
            {
                "_key": f"{repo_name}/{pr.number}",
                "repo": repo_name,
                "number": pr.number,
                "title": pr.title or "",
                "body": pr.body or "",
                "state": pr.state or "",
                "role": "author",
                "comments": comments,
                "labels": [l.name for l in pr.labels],
                "created_at": pr.created_at.isoformat() if pr.created_at else "",
                "merged_at": pr.merged_at.isoformat() if pr.merged_at else "",
                "fetched_at": _now(),
            }
        )

    # Search for PRs where user left a review (last 3 years)
    logger.info(f"[PRs] Fetching reviewed PRs since {cutoff}")
    for i, issue in enumerate(gh.search_issues(f"type:pr reviewed-by:{username} -author:{username} created:>={cutoff}"), 1):
        pr = issue.as_pull_request()
        repo_label = pr.base.repo.full_name if pr.base and pr.base.repo else "unknown"
        repo_short = repo_label.split("/")[-1] if "/" in repo_label else repo_label
        if repo_short in EXCLUDED_REPOS:
            logger.info(f"[PRs reviewed] Skipping excluded repo: {repo_label}#{pr.number}")
            continue
        logger.info(f"[PRs reviewed] {i}: {repo_label}#{pr.number} — {pr.title}")
        comments = []
        try:
            for comment in pr.get_review_comments():
                if comment.user and comment.user.login == username:
                    comments.append(
                        {
                            "author": username,
                            "body": comment.body or "",
                            "created_at": comment.created_at.isoformat() if comment.created_at else "",
                            "is_review_comment": True,
                            "diff_context": comment.diff_hunk or "",
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not fetch review comments for PR {pr.html_url}: {e}")

        repo_name = pr.base.repo.full_name if pr.base and pr.base.repo else ""
        key = f"{repo_name}/{pr.number}"
        # Only add if not already captured as author
        if not any(p["_key"] == key for p in prs):
            prs.append(
                {
                    "_key": key,
                    "repo": repo_name,
                    "number": pr.number,
                    "title": pr.title or "",
                    "body": pr.body or "",
                    "state": pr.state or "",
                    "role": "reviewer",
                    "comments": comments,
                    "labels": [l.name for l in pr.labels],
                    "created_at": pr.created_at.isoformat() if pr.created_at else "",
                    "merged_at": pr.merged_at.isoformat() if pr.merged_at else "",
                    "fetched_at": _now(),
                }
            )

    logger.info(f"Extracted {len(prs)} pull requests for {username}")
    gh.close()
    return prs


@step
def extract_github_issues() -> List[Dict[str, Any]]:
    """Fetch issues authored or commented on by the user."""
    gh = _gh_client()
    username = _get_username()
    issues = []

    for issue in gh.search_issues(f"type:issue author:{username}"):
        if issue.pull_request:
            continue
        comments = []
        try:
            for comment in issue.get_comments():
                comments.append(
                    {
                        "author": comment.user.login if comment.user else "",
                        "body": comment.body or "",
                        "created_at": comment.created_at.isoformat() if comment.created_at else "",
                    }
                )
        except Exception as e:
            logger.warning(f"Could not fetch comments for issue {issue.html_url}: {e}")

        repo_name = issue.repository.full_name if issue.repository else ""
        issues.append(
            {
                "_key": f"{repo_name}/{issue.number}",
                "repo": repo_name,
                "number": issue.number,
                "title": issue.title or "",
                "body": issue.body or "",
                "role": "author",
                "comments": comments,
                "labels": [l.name for l in issue.labels],
                "state": issue.state or "",
                "created_at": issue.created_at.isoformat() if issue.created_at else "",
                "fetched_at": _now(),
            }
        )

    # Issues where user commented but didn't author
    for issue in gh.search_issues(f"type:issue commenter:{username} -author:{username}"):
        if issue.pull_request:
            continue
        comments = []
        try:
            for comment in issue.get_comments():
                if comment.user and comment.user.login == username:
                    comments.append(
                        {
                            "author": username,
                            "body": comment.body or "",
                            "created_at": comment.created_at.isoformat() if comment.created_at else "",
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not fetch comments for issue {issue.html_url}: {e}")

        repo_name = issue.repository.full_name if issue.repository else ""
        key = f"{repo_name}/{issue.number}"
        if not any(i["_key"] == key for i in issues):
            issues.append(
                {
                    "_key": key,
                    "repo": repo_name,
                    "number": issue.number,
                    "title": issue.title or "",
                    "body": issue.body or "",
                    "role": "commenter",
                    "comments": comments,
                    "labels": [l.name for l in issue.labels],
                    "state": issue.state or "",
                    "created_at": issue.created_at.isoformat() if issue.created_at else "",
                    "fetched_at": _now(),
                }
            )

    logger.info(f"Extracted {len(issues)} issues for {username}")
    gh.close()
    return issues


@step
def extract_github_stars() -> List[Dict[str, Any]]:
    """Fetch repos starred by the user."""
    gh = _gh_client()
    username = _get_username()
    user = gh.get_user(username)
    stars = []

    for repo in user.get_starred():
        stars.append(
            {
                "_key": f"{username}/{repo.full_name}",
                "repo_full_name": repo.full_name,
                "description": repo.description or "",
                "language": repo.language or "",
                "topics": repo.get_topics(),
                "starred_at": "",  # GitHub API doesn't return this by default
                "fetched_at": _now(),
            }
        )

    logger.info(f"Extracted {len(stars)} starred repos for {username}")
    gh.close()
    return stars


@step
def extract_github_gists() -> List[Dict[str, Any]]:
    """Fetch gists authored by the user."""
    gh = _gh_client()
    username = _get_username()
    user = gh.get_user(username)
    gists = []

    for gist in user.get_gists():
        files = []
        for filename, gist_file in gist.files.items():
            files.append(
                {
                    "filename": filename,
                    "language": gist_file.language or "",
                    "content": gist_file.content or "",
                }
            )
        gists.append(
            {
                "_key": gist.id,
                "description": gist.description or "",
                "files": files,
                "public": gist.public,
                "comments": gist.comments,
                "created_at": gist.created_at.isoformat() if gist.created_at else "",
                "updated_at": gist.updated_at.isoformat() if gist.updated_at else "",
                "fetched_at": _now(),
            }
        )

    logger.info(f"Extracted {len(gists)} gists for {username}")
    gh.close()
    return gists


# ---------------------------------------------------------------------------
# Store steps
# ---------------------------------------------------------------------------


@step
def store_github_profile(profile: Dict[str, Any]) -> str:
    count = _mongo_upsert("github_profile", [profile])
    msg = f"Stored github_profile: {count} doc(s)"
    logger.info(msg)
    return msg


@step
def store_github_repositories(repos: List[Dict[str, Any]]) -> str:
    count = _mongo_upsert("github_repositories", repos)
    msg = f"Stored github_repositories: {count} doc(s)"
    logger.info(msg)
    return msg


@step
def store_github_commits(commits: List[Dict[str, Any]]) -> str:
    count = _mongo_upsert("github_commits", commits)
    msg = f"Stored github_commits: {count} doc(s)"
    logger.info(msg)
    return msg


@step
def store_github_pull_requests(prs: List[Dict[str, Any]]) -> str:
    count = _mongo_upsert("github_pull_requests", prs)
    msg = f"Stored github_pull_requests: {count} doc(s)"
    logger.info(msg)
    return msg


@step
def store_github_issues(issues: List[Dict[str, Any]]) -> str:
    count = _mongo_upsert("github_issues", issues)
    msg = f"Stored github_issues: {count} doc(s)"
    logger.info(msg)
    return msg


@step
def store_github_stars(stars: List[Dict[str, Any]]) -> str:
    count = _mongo_upsert("github_stars", stars)
    msg = f"Stored github_stars: {count} doc(s)"
    logger.info(msg)
    return msg


@step
def store_github_gists(gists: List[Dict[str, Any]]) -> str:
    count = _mongo_upsert("github_gists", gists)
    msg = f"Stored github_gists: {count} doc(s)"
    logger.info(msg)
    return msg


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@pipeline
def github_to_mongodb_pipeline():
    """Extract GitHub data and store in MongoDB for digital twin training."""
    # Extract (independent steps run in parallel where ZenML can schedule them)
    profile = extract_github_profile()
    repos = extract_github_repositories()
    commits = extract_github_commits()
    prs = extract_github_pull_requests()
    issues = extract_github_issues()
    stars = extract_github_stars()
    gists = extract_github_gists()

    # Store (each depends on its extract step)
    store_github_profile(profile=profile)
    store_github_repositories(repos=repos)
    store_github_commits(commits=commits)
    store_github_pull_requests(prs=prs)
    store_github_issues(issues=issues)
    store_github_stars(stars=stars)
    store_github_gists(gists=gists)


if __name__ == "__main__":
    run = github_to_mongodb_pipeline()
    logger.info("GitHub pipeline finished")
