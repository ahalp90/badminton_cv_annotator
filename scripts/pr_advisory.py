#!/usr/bin/env python3
"""Non-blocking PR quick read using the Gemini API.

Called by the `advisory` job in .github/workflows/pr-quality.yml. Reads the PR's
commit messages, description and implementation diff, asks a cheap/fast LLM to
explain the change, then posts a short comment.

Design goals (see docs/ci.md):
  * NEVER blocks a merge -- this script always exits 0.
  * If the model is rate-limited, unreachable, mis-named or quota-exhausted, that
    is surfaced as a GitHub ``::warning::`` annotation, not a failure.
  * Stdlib only -- no pip install in the workflow.

Environment:
  GEMINI_API_KEY     required; if empty the script no-ops (the workflow already
                     guards on this, this is just belt-and-suspenders). The
                     workflow fills it from the ``PR_MESSAGE_BOT_KEY`` repo secret.
  GEMINI_MODEL       optional model id; defaults to ``gemini-2.5-flash``. Filled
                     from the ``PR_MESSAGE_BOT_MODEL`` repo variable.
  GITHUB_EVENT_PATH  path to the PR event payload (provided by Actions).
  GITHUB_TOKEN       optional; if present, post/update a sticky PR comment.
  GITHUB_REPOSITORY  "owner/repo" (provided by Actions).
  GITHUB_STEP_SUMMARY  optional; the quick read is also written here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

try:
    from scripts.pr_main_files import rank_changed_files
except ModuleNotFoundError:  # Direct execution puts scripts/, not the repo root, on sys.path.
    from pr_main_files import rank_changed_files

DEFAULT_MODEL = "gemini-2.5-flash"
COMMENT_MARKER = "<!-- pr-advisory-bot -->"
HTTP_TIMEOUT = 360  # seconds
MAX_DIFF_FILES = 6
MAX_DIFF_CHARS = 15_000
MAX_FILE_DIFF_CHARS = 5_000
MAX_REVIEW_WORDS = 450
MAX_OUTPUT_TOKENS = 32_000

RUBRIC = """\
Write a short PR note that a tired person can read quickly.

First, silently decide if the PR text feels human-written or mostly AI/agent-written.
- If human: keep its voice where it helps.
- If AI/agentic: keep the facts and rewrite it plainly.

Use the diff, tests and changed docs as the main source. Use commits and PR text for extra context.

Write like this:
- Start with the big picture.
- Keep the important technical details.
- Use simple words where simple words will do.
- Explain technical terms or operations when their importance is not obvious.
- Avoid corporate or project-management language.
- Each sentence sticks to one main idea. Same rule for paragraphs.
- Prefer saying what IS, not what ISN'T.
- Sound warm and natural.
- Never mention the human/AI judgment.

### Summary
One short paragraph: what changed, why it matters, and the main takeaway.

### What changed
2-5 short bullets.

### Worth knowing
0-3 bullets for important experiments, results, open problems, or next steps.
Routine test/lint passes get one short footnote at most.

Stick to facts supported by the diff, tests, docs, commits, or PR text.

Keep it as short as the PR allows.
- Small diff or concept: ~100-180 words.
- Large or complex: up to ~450 words when needed to keep the important technical detail.
"""

OUTPUT_CONTRACT = """\
Return only the finished PR note. Do not echo or analyse the rubric, source
material, audience, task, or your reasoning. The human/AI decision in the rubric
is private and must not appear in the note.

The first characters must be `### Summary`. Use one prose paragraph there, then
`### What changed` with 2-5 short `- ` bullets. Use `### Worth knowing` only when
there are 1-3 useful `- ` bullets. Use no other headings, code fences, or tables.
Do not include a separate diff analysis. End with a complete sentence or bullet.
"""


def warn(message: str) -> None:
    """Emit a non-fatal GitHub Actions warning annotation (single line)."""
    one_line = " ".join(message.split())
    print(f"::warning title=AI quick read unavailable::{one_line}")
    _write_summary(f"### 🤖 AI quick read\n\n> ⚠️ Skipped: {one_line}\n")


def _write_summary(markdown: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
    except OSError:
        pass  # summary is best-effort


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return ""


def _truncate_with_marker(text: str, limit: int, marker: str) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(marker)] + marker


def _ranked_implementation_diff(base: str, head: str) -> str:
    numstat = _git(["diff", "--numstat", f"{base}...{head}"])
    ranked_files, _ = rank_changed_files(numstat)
    selected_files = ranked_files[:MAX_DIFF_FILES]
    if not selected_files:
        return "(implementation diff unavailable)"

    file_sections: list[str] = []
    for ranked_file in selected_files:
        file_diff = _git(
            [
                "diff",
                "--unified=3",
                f"{base}...{head}",
                "--",
                ranked_file.path,
            ]
        )
        if not file_diff:
            continue
        file_marker = f"\n\n[File diff truncated at {MAX_FILE_DIFF_CHARS:,} characters.]"
        file_sections.append(_truncate_with_marker(file_diff, MAX_FILE_DIFF_CHARS, file_marker))

    if not file_sections:
        return "(implementation diff unavailable)"

    section_parts = []
    if len(ranked_files) > MAX_DIFF_FILES:
        section_parts.append(
            f"[Implementation sample limited to the top {MAX_DIFF_FILES} of "
            f"{len(ranked_files)} ranked files.]"
        )
    section_parts.extend(file_sections)
    combined = "\n\n".join(section_parts)
    overall_marker = f"\n\n[Implementation diff truncated at {MAX_DIFF_CHARS:,} characters.]"
    return _truncate_with_marker(combined, MAX_DIFF_CHARS, overall_marker)


def review_format_problem(review: str) -> str | None:
    """Return why an obviously malformed note should not be posted."""
    if not review.startswith("### Summary\n") or "\n### What changed\n" not in review:
        return "response is missing the required sections"
    if "```" in review or "~~~" in review:
        return "response contains a code fence"
    if len(review.split()) > MAX_REVIEW_WORDS:
        return f"response exceeds {MAX_REVIEW_WORDS} words"
    return None


def gather_context(pr: dict) -> str:
    """Build the prompt input from PR prose, commits and the implementation diff."""
    base = pr.get("base", {}).get("sha", "")
    head = pr.get("head", {}).get("sha", "")
    rng = f"{base}..{head}" if base and head else "HEAD~20..HEAD"

    # Commit subjects + bodies, one block each, capped to keep the request small.
    commits = _git(["log", "--no-merges", "--format=%h%x1f%s%x1f%b%x1e", rng])
    blocks = []
    for entry in filter(None, commits.split("\x1e")):
        parts = entry.strip("\n").split("\x1f")
        if len(parts) < 2:
            continue
        short, subject = parts[0].strip(), parts[1].strip()
        body = (parts[2].strip() if len(parts) > 2 else "")[:200]
        blocks.append(f"- {short} {subject}" + (f"\n    {body}" if body else ""))
        if len(blocks) >= 25:
            break
    commit_text = "\n".join(blocks) or "(no commits found in range)"

    diffstat = _git(["diff", "--stat", f"{base}...{head}"]) if base and head else ""
    diffstat = "\n".join(diffstat.splitlines()[:100]) or "(diffstat unavailable)"

    diff = _ranked_implementation_diff(base, head) if base and head else "(implementation diff unavailable)"

    title = pr.get("title", "") or "(no title)"
    body = (pr.get("body") or "(empty PR description)")[:4000]

    return (
        f"## PR title\n{title}\n\n"
        f"## PR description\n{body}\n\n"
        f"## Commits\n{commit_text}\n\n"
        f"## Changed files (diffstat)\n{diffstat}\n\n"
        f"## Implementation diff\n{diff}\n"
    )


def _candidate_text(candidate: dict) -> str:
    finish_reason = candidate.get("finishReason", "")
    if finish_reason and finish_reason != "STOP":
        raise RuntimeError(f"response ended with finish reason {finish_reason}")

    answer_parts = []
    for part in candidate.get("content", {}).get("parts", []):
        if not part.get("thought", False):
            answer_parts.append(part.get("text", ""))
    text = "".join(answer_parts).strip()
    if not text:
        raise RuntimeError("empty response text")
    return text


def call_gemini(model: str, api_key: str, prompt: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback", {})
        raise RuntimeError(f"no candidates returned (feedback: {feedback})")
    return _candidate_text(candidates[0])


def post_comment(repo: str, number: int, token: str, body: str) -> None:
    """Create or update a single sticky quick-read comment (best-effort)."""
    api = "https://api.github.com"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }
    marked = f"{COMMENT_MARKER}\n{body}"

    def _request(method: str, url: str, data: dict | None = None):
        raw = json.dumps(data).encode("utf-8") if data is not None else None
        req = urllib.request.Request(url, data=raw, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        existing = _request(
            "GET", f"{api}/repos/{repo}/issues/{number}/comments?per_page=100"
        )
        for comment in existing:
            if COMMENT_MARKER in (comment.get("body") or ""):
                _request("PATCH", comment["url"], {"body": marked})
                return
        _request(
            "POST",
            f"{api}/repos/{repo}/issues/{number}/comments",
            {"body": marked},
        )
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        # Comment posting is a nicety; the step summary still carries the review.
        warn(f"could not post PR comment ({exc}); see the step summary instead")


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY not set -- AI quick read is dormant, nothing to do.")
        return 0

    model = os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    try:
        with open(event_path, encoding="utf-8") as fh:
            pr = json.load(fh).get("pull_request", {})
    except (OSError, ValueError) as exc:
        warn(f"could not read PR event payload ({exc})")
        return 0
    if not pr:
        warn("event payload has no pull_request object")
        return 0

    prompt = (
        RUBRIC
        + "\n\n"
        + OUTPUT_CONTRACT
        + "\n\n<source_material>\n"
        + gather_context(pr)
        + "</source_material>\n"
    )

    try:
        review = call_gemini(model, api_key, prompt)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001 - best-effort detail only
            detail = ""
        if exc.code == 429:
            warn(f"Gemini API rate-limited (HTTP 429) for model '{model}'. {detail}")
        elif exc.code in (400, 404):
            warn(
                f"Gemini rejected the request (HTTP {exc.code}) -- the model name "
                f"'{model}' may be wrong; set the PR_MESSAGE_BOT_MODEL repo variable "
                f"to a current free-tier model. {detail}"
            )
        elif exc.code in (401, 403):
            warn(f"Gemini auth/quota problem (HTTP {exc.code}) for model '{model}'. {detail}")
        else:
            warn(f"Gemini API error (HTTP {exc.code}) for model '{model}'. {detail}")
        return 0
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        warn(f"Gemini API unreachable ({exc}).")
        return 0
    except (ValueError, RuntimeError, KeyError) as exc:
        warn(f"Unexpected response from Gemini ({exc}).")
        return 0

    format_problem = review_format_problem(review)
    if format_problem:
        warn(f"Gemini returned a malformed quick read ({format_problem})")
        return 0

    note = (
        "🤖 **AI quick read** — generated from the PR and implementation diff\n\n"
        f"{review}\n"
    )
    _write_summary("### " + note)

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    number = pr.get("number")
    if token and repo and number:
        post_comment(repo, int(number), token, note)
    else:
        print("No GITHUB_TOKEN/repo/number -- quick read written to the step summary only.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
