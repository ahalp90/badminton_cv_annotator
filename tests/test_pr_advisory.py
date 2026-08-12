from __future__ import annotations

import json

from scripts import pr_advisory, pr_main_files


def test_gather_context_supplies_ranked_limited_implementation_diff(monkeypatch) -> None:
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
    oversized_file_diff = "x" * (pr_advisory.MAX_FILE_DIFF_CHARS + 1)

    def fake_git(args: list[str]) -> str:
        calls.append(args)
        if args[0] == "log":
            return ""
        if "--stat" in args:
            return "scripts/pr_advisory.py | 40 +++++++++++++++++++++"
        if "--numstat" in args:
            return numstat
        return oversized_file_diff

    monkeypatch.setattr(pr_advisory, "_git", fake_git)
    context = pr_advisory.gather_context(
        {
            "base": {"sha": "base-sha"},
            "head": {"sha": "head-sha"},
            "title": "Explain the implementation",
            "body": "Use the code as the main evidence.",
        }
    )

    diff_paths = [args[-1] for args in calls if "--unified=3" in args]
    assert diff_paths == [
        "src/a.py",
        "scripts/b.py",
        "config.toml",
        "src/c.py",
        "tests/d.py",
        "docs/e.md",
    ]
    supplied_diff = context.split("## Implementation diff\n", maxsplit=1)[1].removesuffix("\n")
    assert len(supplied_diff) == pr_advisory.MAX_DIFF_CHARS
    assert supplied_diff.startswith("[Implementation sample limited to the top 6 of 7 ranked files.]")
    assert f"[File diff truncated at {pr_advisory.MAX_FILE_DIFF_CHARS:,} characters.]" in supplied_diff
    assert supplied_diff.endswith(
        f"[Implementation diff truncated at {pr_advisory.MAX_DIFF_CHARS:,} characters.]"
    )


def test_rank_changed_files_uses_churn_and_path_relevance() -> None:
    ranked, total = pr_main_files.rank_changed_files(
        "10\t0\tconfig.toml\n"
        "4\t0\tsrc/small.py\n"
        "100\t0\tdata/results.csv\n"
        "2\t0\tsrc/trivial.py\n"
        "-\t-\tmodel.pt"
    )

    assert total == 5
    assert [file.path for file in ranked] == ["src/small.py", "config.toml"]


def test_gather_context_limits_commit_count_and_body_length(monkeypatch) -> None:
    long_body = "b" * 250
    commits = "".join(
        f"hash{index:02d}\x1fSubject {index:02d}\x1f{long_body}\x1e"
        for index in range(30)
    )

    def fake_git(args: list[str]) -> str:
        return commits if args[0] == "log" else ""

    monkeypatch.setattr(pr_advisory, "_git", fake_git)
    context = pr_advisory.gather_context(
        {
            "base": {"sha": "base-sha"},
            "head": {"sha": "head-sha"},
        }
    )

    assert "Subject 24" in context
    assert "Subject 25" not in context
    assert "b" * 200 in context
    assert "b" * 201 not in context


def test_main_posts_ai_quick_read_heading(monkeypatch, tmp_path) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"number": 17}}),
        encoding="utf-8",
    )
    posted: dict[str, object] = {}

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setattr(pr_advisory, "gather_context", lambda pr: "prompt context")
    monkeypatch.setattr(pr_advisory, "call_gemini", lambda model, api_key, prompt: "### Summary\nA useful note.")

    def fake_post_comment(repo: str, number: int, token: str, body: str) -> None:
        posted.update(repo=repo, number=number, token=token, body=body)

    monkeypatch.setattr(pr_advisory, "post_comment", fake_post_comment)

    assert pr_advisory.main() == 0
    assert posted == {
        "repo": "owner/repo",
        "number": 17,
        "token": "test-token",
        "body": (
            "🤖 **AI quick read** — generated from the PR and implementation diff\n\n"
            "### Summary\nA useful note.\n"
        ),
    }
