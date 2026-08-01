# Decisions — deviations and open questions

## Deviations and open questions

1. **PyCharm MCP for delegates.** Ariel asked delegates to use the PyCharm MCP for
   static analysis. Headless Codex workers cannot call MCP tools (codex-delegate
   skill constraint). Substitute: delegates get `pyrefly_nav.py` (find-references /
   goto-definition over the pyrefly LSP) copied into local_scratch/, per the skill's
   runtime notes. PyCharm MCP find-usages runs Claude-side on every finding the
   report recommends acting on. Cost: delegates' reference checks are pyrefly-grade,
   not PyCharm-grade (re-exports and dynamic dispatch may slip). Benefit: keeps the
   sweep parallel and headless. The Claude-side verification pass covers the gap on
   anything actionable.
2. **Model routing at xhigh/max.** The codex-delegate skill allows xhigh/max only on
   Luna. Sweep delegates therefore run gpt-5.6-luna at the efforts Ariel named;
   Sol stays at medium (plan review) and high (report audit). One-liner, no
   behavioural edge.
3. **Scope calls.** docs/**/*.py and scripts/archive are out of audit scope
   (evidence artefacts and already-archived code). Confirmed nothing else was
   requested in scope beyond src/ and scripts/.
4. **No commits, no edits outside docs/dead_code_clean/.** Audit writes docs only.
   The refactor itself is a separate, later pass.
5. **Sol plan review outcome.** All ten criticisms adopted into plan rev 2, with
   one partial: Sol called the doc apparatus overweight; the worklog, decisions
   doc, and Sol report-audit stay because Ariel asked for them explicitly. What
   was trimmed instead: per-WP findings files (raw returns stay in the session
   scratchpad; the repo gets one merged ledger), the RUF100 gate, and one Sol
   review rather than two of the report.
