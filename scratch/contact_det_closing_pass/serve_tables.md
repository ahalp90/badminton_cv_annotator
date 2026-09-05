# Serve discovery and server attribution

**The recommended detector finds 2,781 / 3,422 (81.3%) labelled serves at ±10.** It both finds the serve and identifies the server in 2,647 / 3,422 (77.4%) retained rallies. Among nonempty proposals, 2,624 / 3,725 (70.4%) start at the serve, and 2,536 / 3,725 (68.1%) also name the right server. The other 257 proposed sections are empty.

These tables recount saved predictions. The serve is the first labelled contact of each retained rally. The proposed start is the first event of a nonempty section. Matching uses the full video contact stream once at each tolerance; it does not match against serve labels alone. Tolerances use a 30 fps clock and scale once to source fps. Raw sides are wrist/net guesses; final sides use the existing alternating-sequence vote.

Development contains 32 grouped videos. The broader comparison contains 47 previously examined videos. Old cached detector scores retain cross-group dependence; these are not fresh independent test estimates.

Unmatched starts inside an unambiguous retained rally's contact envelope count as extra leading events. Unmatched starts outside that support remain unknown. Unknowns stay in the all-start denominator. Empty sections are listed separately. Missing predicted sides are failures; missing label sides are unknown.

- [Development](#development)
- [Broader comparison](#broader)
- [Serve timing](#serve-timing)

## Development

### Find the serve anywhere in the full stream

| Detector | Tolerance | Serve timing / all retained starts | Timing + raw side / known-side starts | Timing + final side / known-side starts |
| --- | --- | --- | --- | --- |
| Original contacts | ±10 | 1,484 / 2,691 (55.1%) | 1,325 / 2,691 (49.2%) | 1,265 / 2,691 (47.0%) |
| Original contacts | ±5 | 1,354 / 2,691 (50.3%) | 1,218 / 2,691 (45.3%) | 1,158 / 2,691 (43.0%) |
| Preceding detector | ±10 | 1,893 / 2,691 (70.3%) | 1,507 / 2,691 (56.0%) | 1,778 / 2,691 (66.1%) |
| Preceding detector | ±5 | 1,560 / 2,691 (58.0%) | 1,294 / 2,691 (48.1%) | 1,475 / 2,691 (54.8%) |
| Guarded edges only | ±10 | 1,893 / 2,691 (70.3%) | 1,507 / 2,691 (56.0%) | 1,778 / 2,691 (66.1%) |
| Guarded edges only | ±5 | 1,560 / 2,691 (58.0%) | 1,294 / 2,691 (48.1%) | 1,475 / 2,691 (54.8%) |
| Local insertion + guarded edges | ±10 | 1,894 / 2,691 (70.4%) | 1,515 / 2,691 (56.3%) | 1,790 / 2,691 (66.5%) |
| Local insertion + guarded edges | ±5 | 1,565 / 2,691 (58.2%) | 1,302 / 2,691 (48.4%) | 1,487 / 2,691 (55.3%) |
| Wider early shortlist + edges | ±10 | 1,906 / 2,691 (70.8%) | 1,516 / 2,691 (56.3%) | 1,806 / 2,691 (67.1%) |
| Wider early shortlist + edges | ±5 | 1,574 / 2,691 (58.5%) | 1,302 / 2,691 (48.4%) | 1,500 / 2,691 (55.7%) |

### Identify the server among timing-matched serves

| Detector | Tolerance | Side answer | Correct / wrong / missing prediction / missing label | Correct / answered | Answered / matched serves |
| --- | --- | --- | --- | --- | --- |
| Original contacts | ±10 | raw | 1,325 / 120 / 39 / 0 | 1,325 / 1,445 (91.7%) | 1,445 / 1,484 (97.4%) |
| Original contacts | ±10 | final | 1,265 / 187 / 32 / 0 | 1,265 / 1,452 (87.1%) | 1,452 / 1,484 (97.8%) |
| Original contacts | ±5 | raw | 1,218 / 99 / 37 / 0 | 1,218 / 1,317 (92.5%) | 1,317 / 1,354 (97.3%) |
| Original contacts | ±5 | final | 1,158 / 166 / 30 / 0 | 1,158 / 1,324 (87.5%) | 1,324 / 1,354 (97.8%) |
| Preceding detector | ±10 | raw | 1,507 / 140 / 246 / 0 | 1,507 / 1,647 (91.5%) | 1,647 / 1,893 (87.0%) |
| Preceding detector | ±10 | final | 1,778 / 110 / 5 / 0 | 1,778 / 1,888 (94.2%) | 1,888 / 1,893 (99.7%) |
| Preceding detector | ±5 | raw | 1,294 / 106 / 160 / 0 | 1,294 / 1,400 (92.4%) | 1,400 / 1,560 (89.7%) |
| Preceding detector | ±5 | final | 1,475 / 80 / 5 / 0 | 1,475 / 1,555 (94.9%) | 1,555 / 1,560 (99.7%) |
| Guarded edges only | ±10 | raw | 1,507 / 140 / 246 / 0 | 1,507 / 1,647 (91.5%) | 1,647 / 1,893 (87.0%) |
| Guarded edges only | ±10 | final | 1,778 / 110 / 5 / 0 | 1,778 / 1,888 (94.2%) | 1,888 / 1,893 (99.7%) |
| Guarded edges only | ±5 | raw | 1,294 / 106 / 160 / 0 | 1,294 / 1,400 (92.4%) | 1,400 / 1,560 (89.7%) |
| Guarded edges only | ±5 | final | 1,475 / 80 / 5 / 0 | 1,475 / 1,555 (94.9%) | 1,555 / 1,560 (99.7%) |
| Local insertion + guarded edges | ±10 | raw | 1,515 / 135 / 244 / 0 | 1,515 / 1,650 (91.8%) | 1,650 / 1,894 (87.1%) |
| Local insertion + guarded edges | ±10 | final | 1,790 / 99 / 5 / 0 | 1,790 / 1,889 (94.8%) | 1,889 / 1,894 (99.7%) |
| Local insertion + guarded edges | ±5 | raw | 1,302 / 105 / 158 / 0 | 1,302 / 1,407 (92.5%) | 1,407 / 1,565 (89.9%) |
| Local insertion + guarded edges | ±5 | final | 1,487 / 73 / 5 / 0 | 1,487 / 1,560 (95.3%) | 1,560 / 1,565 (99.7%) |
| Wider early shortlist + edges | ±10 | raw | 1,516 / 137 / 253 / 0 | 1,516 / 1,653 (91.7%) | 1,653 / 1,906 (86.7%) |
| Wider early shortlist + edges | ±10 | final | 1,806 / 96 / 4 / 0 | 1,806 / 1,902 (95.0%) | 1,902 / 1,906 (99.8%) |
| Wider early shortlist + edges | ±5 | raw | 1,302 / 107 / 165 / 0 | 1,302 / 1,409 (92.4%) | 1,409 / 1,574 (89.5%) |
| Wider early shortlist + edges | ±5 | final | 1,500 / 70 / 4 / 0 | 1,500 / 1,570 (95.5%) | 1,570 / 1,574 (99.7%) |

### Start the proposed output at the serve

| Detector | Tolerance | Correct / all nonempty starts | Correct / judgeable starts | Later hit | Extra leading | Unknown | Empty sections |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original contacts | ±10 | 1,382 / 2,621 (52.7%) | 1,382 / 2,360 (58.6%) | 830 | 148 | 261 | 229 |
| Original contacts | ±5 | 1,259 / 2,621 (48.0%) | 1,259 / 2,310 (54.5%) | 823 | 228 | 311 | 229 |
| Preceding detector | ±10 | 1,804 / 2,621 (68.8%) | 1,804 / 2,308 (78.2%) | 148 | 356 | 313 | 229 |
| Preceding detector | ±5 | 1,484 / 2,621 (56.6%) | 1,484 / 2,131 (69.6%) | 133 | 514 | 490 | 229 |
| Guarded edges only | ±10 | 1,804 / 2,621 (68.8%) | 1,804 / 2,308 (78.2%) | 148 | 356 | 313 | 229 |
| Guarded edges only | ±5 | 1,484 / 2,621 (56.6%) | 1,484 / 2,131 (69.6%) | 133 | 514 | 490 | 229 |
| Local insertion + guarded edges | ±10 | 1,803 / 2,621 (68.8%) | 1,803 / 2,311 (78.0%) | 143 | 365 | 310 | 229 |
| Local insertion + guarded edges | ±5 | 1,486 / 2,621 (56.7%) | 1,486 / 2,135 (69.6%) | 127 | 522 | 486 | 229 |
| Wider early shortlist + edges | ±10 | 1,818 / 2,621 (69.4%) | 1,818 / 2,302 (79.0%) | 115 | 369 | 319 | 229 |
| Wider early shortlist + edges | ±5 | 1,498 / 2,621 (57.2%) | 1,498 / 2,130 (70.3%) | 101 | 531 | 491 | 229 |

### Start at the serve and identify its server

| Detector | Tolerance | Raw timing + side / all starts | Final timing + side / all starts | Final timing + side / judgeable starts |
| --- | --- | --- | --- | --- |
| Original contacts | ±10 | 1,262 / 2,621 (48.1%) | 1,218 / 2,621 (46.5%) | 1,218 / 2,360 (51.6%) |
| Original contacts | ±5 | 1,157 / 2,621 (44.1%) | 1,114 / 2,621 (42.5%) | 1,114 / 2,310 (48.2%) |
| Preceding detector | ±10 | 1,442 / 2,621 (55.0%) | 1,719 / 2,621 (65.6%) | 1,719 / 2,308 (74.5%) |
| Preceding detector | ±5 | 1,235 / 2,621 (47.1%) | 1,423 / 2,621 (54.3%) | 1,423 / 2,131 (66.8%) |
| Guarded edges only | ±10 | 1,442 / 2,621 (55.0%) | 1,719 / 2,621 (65.6%) | 1,719 / 2,308 (74.5%) |
| Guarded edges only | ±5 | 1,235 / 2,621 (47.1%) | 1,423 / 2,621 (54.3%) | 1,423 / 2,131 (66.8%) |
| Local insertion + guarded edges | ±10 | 1,452 / 2,621 (55.4%) | 1,732 / 2,621 (66.1%) | 1,732 / 2,311 (74.9%) |
| Local insertion + guarded edges | ±5 | 1,244 / 2,621 (47.5%) | 1,435 / 2,621 (54.8%) | 1,435 / 2,135 (67.2%) |
| Wider early shortlist + edges | ±10 | 1,455 / 2,621 (55.5%) | 1,748 / 2,621 (66.7%) | 1,748 / 2,302 (75.9%) |
| Wider early shortlist + edges | ±5 | 1,246 / 2,621 (47.5%) | 1,448 / 2,621 (55.2%) | 1,448 / 2,130 (68.0%) |

## Broader

### Find the serve anywhere in the full stream

| Detector | Tolerance | Serve timing / all retained starts | Timing + raw side / known-side starts | Timing + final side / known-side starts |
| --- | --- | --- | --- | --- |
| Original contacts | ±10 | 1,986 / 3,422 (58.0%) | 1,757 / 3,422 (51.3%) | 1,722 / 3,422 (50.3%) |
| Original contacts | ±5 | 1,844 / 3,422 (53.9%) | 1,650 / 3,422 (48.2%) | 1,600 / 3,422 (46.8%) |
| Preceding detector | ±10 | 2,769 / 3,422 (80.9%) | 2,202 / 3,422 (64.3%) | 2,611 / 3,422 (76.3%) |
| Preceding detector | ±5 | 2,350 / 3,422 (68.7%) | 1,953 / 3,422 (57.1%) | 2,228 / 3,422 (65.1%) |
| Guarded edges only | ±10 | 2,769 / 3,422 (80.9%) | 2,202 / 3,422 (64.3%) | 2,611 / 3,422 (76.3%) |
| Guarded edges only | ±5 | 2,350 / 3,422 (68.7%) | 1,953 / 3,422 (57.1%) | 2,228 / 3,422 (65.1%) |
| Local insertion + guarded edges | ±10 | 2,781 / 3,422 (81.3%) | 2,222 / 3,422 (64.9%) | 2,647 / 3,422 (77.4%) |
| Local insertion + guarded edges | ±5 | 2,371 / 3,422 (69.3%) | 1,972 / 3,422 (57.6%) | 2,263 / 3,422 (66.1%) |
| Wider early shortlist + edges | ±10 | 2,784 / 3,422 (81.4%) | 2,212 / 3,422 (64.6%) | 2,652 / 3,422 (77.5%) |
| Wider early shortlist + edges | ±5 | 2,366 / 3,422 (69.1%) | 1,961 / 3,422 (57.3%) | 2,262 / 3,422 (66.1%) |

### Identify the server among timing-matched serves

| Detector | Tolerance | Side answer | Correct / wrong / missing prediction / missing label | Correct / answered | Answered / matched serves |
| --- | --- | --- | --- | --- | --- |
| Original contacts | ±10 | raw | 1,757 / 193 / 36 / 0 | 1,757 / 1,950 (90.1%) | 1,950 / 1,986 (98.2%) |
| Original contacts | ±10 | final | 1,722 / 235 / 29 / 0 | 1,722 / 1,957 (88.0%) | 1,957 / 1,986 (98.5%) |
| Original contacts | ±5 | raw | 1,650 / 159 / 35 / 0 | 1,650 / 1,809 (91.2%) | 1,809 / 1,844 (98.1%) |
| Original contacts | ±5 | final | 1,600 / 215 / 29 / 0 | 1,600 / 1,815 (88.2%) | 1,815 / 1,844 (98.4%) |
| Preceding detector | ±10 | raw | 2,202 / 247 / 320 / 0 | 2,202 / 2,449 (89.9%) | 2,449 / 2,769 (88.4%) |
| Preceding detector | ±10 | final | 2,611 / 147 / 11 / 0 | 2,611 / 2,758 (94.7%) | 2,758 / 2,769 (99.6%) |
| Preceding detector | ±5 | raw | 1,953 / 197 / 200 / 0 | 1,953 / 2,150 (90.8%) | 2,150 / 2,350 (91.5%) |
| Preceding detector | ±5 | final | 2,228 / 113 / 9 / 0 | 2,228 / 2,341 (95.2%) | 2,341 / 2,350 (99.6%) |
| Guarded edges only | ±10 | raw | 2,202 / 247 / 320 / 0 | 2,202 / 2,449 (89.9%) | 2,449 / 2,769 (88.4%) |
| Guarded edges only | ±10 | final | 2,611 / 147 / 11 / 0 | 2,611 / 2,758 (94.7%) | 2,758 / 2,769 (99.6%) |
| Guarded edges only | ±5 | raw | 1,953 / 197 / 200 / 0 | 1,953 / 2,150 (90.8%) | 2,150 / 2,350 (91.5%) |
| Guarded edges only | ±5 | final | 2,228 / 113 / 9 / 0 | 2,228 / 2,341 (95.2%) | 2,341 / 2,350 (99.6%) |
| Local insertion + guarded edges | ±10 | raw | 2,222 / 250 / 309 / 0 | 2,222 / 2,472 (89.9%) | 2,472 / 2,781 (88.9%) |
| Local insertion + guarded edges | ±10 | final | 2,647 / 128 / 6 / 0 | 2,647 / 2,775 (95.4%) | 2,775 / 2,781 (99.8%) |
| Local insertion + guarded edges | ±5 | raw | 1,972 / 200 / 199 / 0 | 1,972 / 2,172 (90.8%) | 2,172 / 2,371 (91.6%) |
| Local insertion + guarded edges | ±5 | final | 2,263 / 102 / 6 / 0 | 2,263 / 2,365 (95.7%) | 2,365 / 2,371 (99.7%) |
| Wider early shortlist + edges | ±10 | raw | 2,212 / 245 / 327 / 0 | 2,212 / 2,457 (90.0%) | 2,457 / 2,784 (88.3%) |
| Wider early shortlist + edges | ±10 | final | 2,652 / 126 / 6 / 0 | 2,652 / 2,778 (95.5%) | 2,778 / 2,784 (99.8%) |
| Wider early shortlist + edges | ±5 | raw | 1,961 / 195 / 210 / 0 | 1,961 / 2,156 (91.0%) | 2,156 / 2,366 (91.1%) |
| Wider early shortlist + edges | ±5 | final | 2,262 / 98 / 6 / 0 | 2,262 / 2,360 (95.8%) | 2,360 / 2,366 (99.7%) |

### Start the proposed output at the serve

| Detector | Tolerance | Correct / all nonempty starts | Correct / judgeable starts | Later hit | Extra leading | Unknown | Empty sections |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original contacts | ±10 | 1,795 / 3,725 (48.2%) | 1,795 / 2,927 (61.3%) | 1060 | 72 | 798 | 257 |
| Original contacts | ±5 | 1,667 / 3,725 (44.8%) | 1,667 / 2,900 (57.5%) | 1047 | 186 | 825 | 257 |
| Preceding detector | ±10 | 2,613 / 3,725 (70.1%) | 2,613 / 2,905 (89.9%) | 168 | 124 | 820 | 257 |
| Preceding detector | ±5 | 2,214 / 3,725 (59.4%) | 2,214 / 2,691 (82.3%) | 150 | 327 | 1034 | 257 |
| Guarded edges only | ±10 | 2,613 / 3,725 (70.1%) | 2,613 / 2,905 (89.9%) | 168 | 124 | 820 | 257 |
| Guarded edges only | ±5 | 2,214 / 3,725 (59.4%) | 2,214 / 2,691 (82.3%) | 150 | 327 | 1034 | 257 |
| Local insertion + guarded edges | ±10 | 2,624 / 3,725 (70.4%) | 2,624 / 2,910 (90.2%) | 161 | 125 | 815 | 257 |
| Local insertion + guarded edges | ±5 | 2,234 / 3,725 (60.0%) | 2,234 / 2,709 (82.5%) | 144 | 331 | 1016 | 257 |
| Wider early shortlist + edges | ±10 | 2,625 / 3,725 (70.5%) | 2,625 / 2,901 (90.5%) | 151 | 125 | 824 | 257 |
| Wider early shortlist + edges | ±5 | 2,227 / 3,725 (59.8%) | 2,227 / 2,694 (82.7%) | 133 | 334 | 1031 | 257 |

### Start at the serve and identify its server

| Detector | Tolerance | Raw timing + side / all starts | Final timing + side / all starts | Final timing + side / judgeable starts |
| --- | --- | --- | --- | --- |
| Original contacts | ±10 | 1,618 / 3,725 (43.4%) | 1,597 / 3,725 (42.9%) | 1,597 / 2,927 (54.6%) |
| Original contacts | ±5 | 1,520 / 3,725 (40.8%) | 1,486 / 3,725 (39.9%) | 1,486 / 2,900 (51.2%) |
| Preceding detector | ±10 | 2,080 / 3,725 (55.8%) | 2,503 / 3,725 (67.2%) | 2,503 / 2,905 (86.2%) |
| Preceding detector | ±5 | 1,842 / 3,725 (49.4%) | 2,133 / 3,725 (57.3%) | 2,133 / 2,691 (79.3%) |
| Guarded edges only | ±10 | 2,080 / 3,725 (55.8%) | 2,503 / 3,725 (67.2%) | 2,503 / 2,905 (86.2%) |
| Guarded edges only | ±5 | 1,842 / 3,725 (49.4%) | 2,133 / 3,725 (57.3%) | 2,133 / 2,691 (79.3%) |
| Local insertion + guarded edges | ±10 | 2,100 / 3,725 (56.4%) | 2,536 / 3,725 (68.1%) | 2,536 / 2,910 (87.1%) |
| Local insertion + guarded edges | ±5 | 1,860 / 3,725 (49.9%) | 2,167 / 3,725 (58.2%) | 2,167 / 2,709 (80.0%) |
| Wider early shortlist + edges | ±10 | 2,089 / 3,725 (56.1%) | 2,541 / 3,725 (68.2%) | 2,541 / 2,901 (87.6%) |
| Wider early shortlist + edges | ±5 | 1,849 / 3,725 (49.6%) | 2,166 / 3,725 (58.1%) | 2,166 / 2,694 (80.4%) |

## Serve timing

![Serve timing errors and missed serves for the recommended detector.](figures/serve_timing.png)

[Per-video counts](results/serve_followups/serve_per_video.csv.gz) accompany the full saved rows and identity comparisons.
