# Figure generation

`render_report_figures.py` reads the checked compressed records and regenerates
the four PNG report figures.

Run from the parent directory:

```bash
MPLCONFIGDIR=/tmp/badminton-matplotlib \
  ~/.venvs/badminton-cicd/bin/python scripts/render_report_figures.py
```

Render the Mermaid decision flow with:

```bash
/home/ariel/.venvs/skill-utils/bin/mermaidx \
  -i figures/preferred_server_rule.mmd \
  -o figures/preferred_server_rule.svg
```
