# Serve identification by shuttle lookback

The investigation found a server-identification rule worth testing on unseen
rallies. It did not find a broad, safe way to recover the exact visible start
of every rally.

The preferred rule gets 170 of 239 server sides right on the development set.
PR #82 gets 163 right. The new rule repairs 20 PR #82 mistakes and introduces
13 new mistakes. This is a development result, so the rule now needs a frozen
test on unseen rallies.

## The important distinction

The work measures two different answers:

- **Server attribution:** which court side served
- **Visible-start attribution:** whether the chosen frame is the visible serve,
  or the first return when the serve happened before the video began

A rule can choose the right side for the wrong timing reason. PR #82 does this
in 67 of the 239 rallies. Report both scores rather than treating one as a
substitute for the other.

| Development result | Server side | Visible start | Both correct |
| --- | ---: | ---: | ---: |
| PR #82 | 163/239 | 125/239 | 96/239 |
| Preferred layered rule | **170/239** | **132/239** | **117/239** |
| Nearby alternative: first-contact fallback | 171/239 | 131/239 | 117/239 |

The preferred rule gives up one server hit compared with this alternative. It
keeps one additional correct visible-start answer and uses the already checked
PR #82 fallback. Freeze the preferred rule for the first unseen test.

![Server attribution across the core approaches](figures/server_attribution.png)

## Read by available attention

- **Two minutes:** this file
- **Ten minutes:** [report.md](report.md), which explains what was tried and why
- **Implementation:** [HANDOVER.md](HANDOVER.md) and
  [docs/next_steps.md](docs/next_steps.md)
- **Numbers:** [docs/results.md](docs/results.md)
- **Reproduction:** [docs/reproducibility.md](docs/reproducibility.md)
- **Past investigation state:** [archive/ARCHIVE_MAP.md](archive/ARCHIVE_MAP.md)

`worklog.md` is the short live record for agent re-entry. The original worklog
is preserved in full beneath an archive notice at
`archive/original_investigation/worklog.md`.

## Current decision

Freeze the preferred layered rule now. Test it next on unseen rallies, without
using these 239 development rallies for any more tuning.

Keep the narrow high-shot timing correction as a separate timing hypothesis.
Do not add the proposed curved-path rescue to the server rule. The supplied
material does not reproduce a safe server-side gain from that proposal.

If exact start timing still needs a large improvement, add a new observation.
Local racket or hitting-arm motion is a more promising next input than another
threshold over the same two-dimensional shuttle-distance trace.

## House rules

- `README.md` remains the single entry point
- `HANDOVER.md` and `worklog.md` remain short and current
- Full session history belongs in `archive/`
- Explain a technical idea in plain language before giving its threshold
- Use descriptive names in live material; historical shorthand stays in archive
- Store CSV and JSON records as `.gz`, and NumPy records as `.npy.xz`
- Put generated figures in `figures/` and keep their source in `scripts/`
