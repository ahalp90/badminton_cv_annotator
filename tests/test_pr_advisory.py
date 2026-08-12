from __future__ import annotations

import json

import pytest

from scripts import pr_advisory


def test_ranked_diff_samples_main_files_within_limits(monkeypatch) -> None:
    calls: list[list[str]] = []
    numstat = (
        "100\t0\tsrc/a.py\n"
        "140\t0\tscripts/b.py\n"
        "250\t0\tconfig.toml\n"
        "50\t0\tsrc/c.py\n"
        "70\t0\ttests/d.py\n"
        "130\t0\tdocs/e.md\n"
        "40\t0\tsrc/f.py\n"
        "1000\t0\tdata/ignored.csv"
    )

    def fake_git(args: list[str]) -> str:
        calls.append(args)
        return numstat if "--numstat" in args else "x" * 6_000

    monkeypatch.setattr(pr_advisory, "_git", fake_git)
    diff = pr_advisory._ranked_implementation_diff("base-sha", "head-sha")

    diff_paths = [args[-1] for args in calls if "--unified=3" in args]
    assert diff_paths == [
        "src/a.py",
        "scripts/b.py",
        "config.toml",
        "src/c.py",
        "tests/d.py",
        "docs/e.md",
    ]
    assert len(diff) == pr_advisory.MAX_DIFF_CHARS
    assert "[File diff truncated at 5,000 characters.]" in diff
    assert diff.endswith("[Implementation diff truncated at 15,000 characters.]")


def test_candidate_text_excludes_thought_parts() -> None:
    candidate = {
        "finishReason": "STOP",
        "content": {
            "parts": [
                {"thought": True, "text": "Tired person. Analyse the diff first."},
                {"text": "### Summary\nA concise explanation."},
            ]
        },
    }

    assert pr_advisory._candidate_text(candidate) == "### Summary\nA concise explanation."


def test_candidate_text_rejects_cut_off_response() -> None:
    candidate = {
        "finishReason": "MAX_TOKENS",
        "content": {"parts": [{"text": "### Summary\nCut off"}]},
    }

    with pytest.raises(RuntimeError, match="finish reason MAX_TOKENS"):
        pr_advisory._candidate_text(candidate)


def test_main_posts_valid_quick_read(monkeypatch, tmp_path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"pull_request": {"number": 17}}), encoding="utf-8")
    review = (
        "### Summary\nA useful note.\n\n"
        "### What changed\n"
        "- Reads the implementation diff.\n"
        "- Explains the main changes."
    )
    posted: dict[str, object] = {}

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(pr_advisory, "gather_context", lambda pr: "prompt context")
    monkeypatch.setattr(pr_advisory, "call_gemini", lambda model, api_key, prompt: review)

    def fake_post_comment(repo: str, number: int, token: str, body: str) -> None:
        posted.update(repo=repo, number=number, token=token, body=body)

    monkeypatch.setattr(pr_advisory, "post_comment", fake_post_comment)

    assert pr_advisory.main() == 0
    assert posted["body"] == (
        "🤖 **AI quick read** — generated from the PR and implementation diff\n\n"
        f"{review}\n"
    )
