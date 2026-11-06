# Automatic badminton annotation

Start with [`annotate_video.py`](annotate_video.py). It coordinates one video's
saved shuttle positions, player poses and court geometry into rally boundaries,
contact times, top/bottom hitters, landings and estimated winners.

| To change or inspect… | Open… |
| --- | --- |
| The complete annotation flow | [`annotate_video.py`](annotate_video.py) |
| Inputs prepared for contact inference | [`inference/prepare_contact_inference.py`](inference/prepare_contact_inference.py) |
| Rule-based hit detection | [`contacts/heuristics.py`](contacts/heuristics.py) |
| Measurements supplied to the contact model | [`contacts/build_features.py`](contacts/build_features.py) |
| Contact probabilities and competing nearby hits | [`contacts/score_candidates.py`](contacts/score_candidates.py) |
| Wrist distances, hitter attribution and alternating player order | [`contacts/player_attribution.py`](contacts/player_attribution.py) |
| Following the top/bottom players | [`player_tracking.py`](player_tracking.py) |
| Initial rally boundaries | [`rallies/find_rallies.py`](rallies/find_rallies.py) |
| Alternative contact lists for each rally | [`rallies/build_contact_proposals.py`](rallies/build_contact_proposals.py) |
| Choosing a contact list and finishing rally bounds | [`rallies/refine_rallies.py`](rallies/refine_rallies.py) |
| Camera geometry and recurring views | [`courts/`](courts/) |
| Excluded footage and fabricated shuttle positions | [`excluded_frames/`](excluded_frames/) |
| Landings, point winners and contact heights | [`rally_results/`](rally_results/) |
| Fitting and evaluating the current learned models | [`training/`](training/) |
| Ground-truth scoring used by earlier fixture experiments | [`evaluation/`](evaluation/) |

`python -m annotator` runs the full annotator. `python -m annotator.training`
fits or evaluates the three learned models. See the
[quickstart](../../docs/annotator/quickstart.md) for input paths and commands.

Manual editors are in [`labelling`](../labelling/). Video rendering is in
[`video_overlay`](../video_overlay/). Historical pinned benchmarks are in
[`experiments/annotator/shuttleset_benchmarks`](../../experiments/annotator/shuttleset_benchmarks/).
