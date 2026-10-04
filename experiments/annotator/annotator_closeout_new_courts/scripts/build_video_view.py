"""Build a self-contained per-video view from the saved closeout tables."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIDE_NAMES = {"Top": "Far", "Bot": "Near"}
STATE_KEYS = {"Court rejected": "court_rejected", "Court accepted; a player pick missing": "accepted_missing_pick",
              "Court accepted; both players picked": "accepted_both_picked"}
TEMPLATE = '''<!doctype html>
<html lang="en-AU"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Per-Video Annotation Results</title>
<style>
:root{--ink:#202b35;--muted:#435465;--page:#f6f8fa;--panel:white;--line:#d5dde3;--head:#edf2f6;--accent:#00659e;--note:#fff0db;--note-edge:#a65b00;--shade:0,114,178}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--ink:#e3e9ee;--muted:#a9b6c1;--page:#141a1f;--panel:#1d252c;--line:#34414b;--head:#26313a;--accent:#5fb4ea;--note:#3a2a14;--note-edge:#e09b4a;--shade:95,180,234}}
:root[data-theme="dark"]{--ink:#e3e9ee;--muted:#a9b6c1;--page:#141a1f;--panel:#1d252c;--line:#34414b;--head:#26313a;--accent:#5fb4ea;--note:#3a2a14;--note-edge:#e09b4a;--shade:95,180,234}
body{font:17px/1.5 system-ui,sans-serif;color:var(--ink);background:var(--page);margin:0}
main{max-width:1060px;margin:32px auto;padding:0 16px 48px}h1{font-size:30px;line-height:1.2}h2{font-size:23px;margin-top:30px}
p{max-width:850px}.intro{color:var(--muted)}.controls{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}
label{font-weight:650}select{display:block;margin-top:6px;padding:10px;font:inherit;color:var(--ink);background:var(--panel);border:1px solid #71808a;border-radius:5px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.card{background:var(--panel);padding:18px;border:1px solid var(--line);border-radius:7px}
.big{font-size:29px;font-weight:700;display:block;color:var(--accent)}.detail{font-size:15px;color:var(--muted)}.note{padding:14px 18px;background:var(--note);border-left:4px solid var(--note-edge)}
table{width:100%;border-collapse:collapse;background:var(--panel)}caption{text-align:left;margin-bottom:10px;font-weight:650}
th,td{padding:14px 10px;border:1px solid var(--line);text-align:right}th{background:var(--head)}th:first-child{text-align:left}thead th{text-align:center}
.matrix td{font-size:23px;min-width:110px;text-align:center}.scroll{overflow-x:auto}.footnote{font-size:15px;color:var(--muted)}.axis{font-size:15px;margin:12px 0 5px}
@media(max-width:700px){.cards{grid-template-columns:1fr}h1{font-size:26px}.matrix td{min-width:65px;font-size:18px}th,td{padding:10px 5px}}
@media print{body{background:white}main{margin:0}.controls{margin:10px 0}.card{break-inside:avoid}table{break-inside:avoid}}
</style></head><body><main>
<h1>Where each video's annotations succeed and fail</h1>
<p class="intro">46 ShuttleSet22 videos · new-court model · cleaned labels · ±10 frames at 30 fps.
Choose a video to see its contact, player and rally results. These counts compare saved outputs with labels; they are not a fresh test on unseen matches.</p>
<div class="controls"><label>Video<select id="video"></select></label></div>
<div id="warning"></div><div class="cards" id="cards"></div>
<p id="courts" class="footnote"></p>
<h2>Which player was named for each labelled hit?</h2>
<p>The rows say who the label names. The columns say what the detector returned within the timing allowance.
A correct hit needs both the right time and the right player. Far and near refer to the image, not an athlete's identity across court-end changes.</p>
<p class="axis"><strong>Rows:</strong> labelled player &nbsp; · &nbsp; <strong>Columns:</strong> detector's answer</p>
<div class="scroll"><table class="matrix"><thead><tr><th scope="col">Labelled player</th><th scope="col">Far player</th><th scope="col">Near player</th><th scope="col">Hit found;<br>no player</th><th scope="col">No timing<br>match</th></tr></thead><tbody id="matrix"></tbody></table></div>
<p class="footnote">Every labelled hit appears once. Shading shows the share within each row; numbers are counts. One cleaned label across all 46 videos has no side; it appears in its video's total only.</p>
<p id="extras" class="note"></p>
<h2>What was available at the missed contact times?</h2>
<div class="scroll"><table><caption>Saved input state at each labelled frame</caption><thead><tr><th>Input state</th><th>Timing matched</th><th>Missed</th><th>Total labels</th></tr></thead><tbody id="inputs"></tbody></table></div>
<p class="footnote">These are linked pipeline stages, not separate proven causes. A rejected frame can still match a nearby event. Label errors can coexist with court rejection.</p>
<h2>How much of each rally is usable?</h2>
<table><thead><tr><th>Requirement</th><th>Labelled rallies meeting it</th></tr></thead><tbody id="rallies"></tbody></table>
<p class="footnote">A clip can contain the whole rally yet have extra or missing hits. An exact sequence has every labelled hit matched once and no extra hit. Fully correct also requires the right players.</p>
<p id="selection"></p>
<p class="footnote">The source tables also contain ±5-frame results. Video IDs belong to ShuttleSet22 and differ from original ShuttleSet IDs. Video 15 is excluded because its labels and footage disagree.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const video=document.getElementById('video');
const number=value=>Number(value).toLocaleString('en-AU');
const rate=(count,total)=>(100*count/total).toFixed(1)+'%';
const rows=[...data].sort((a,b)=>a.fully_correct_rallies/a.labelled_rallies-b.fully_correct_rallies/b.labelled_rallies);
for(const row of rows){const option=document.createElement('option');option.value=row.video;option.textContent='Video '+row.video;video.append(option);}
video.value='53';
const warnings={53:'<p class="note"><strong>Video 53: the earlier court failure is fixed.</strong> The new court is accepted in the scene that the old outline lost. See court_checks.md.</p>',
17:'<p class="note"><strong>Video 17: the far player is now picked, but player errors remain.</strong> Many matched hits still name the wrong side. See court_checks.md.</p>'};
function update(){
const row=data.find(item=>item.video===Number(video.value));
document.getElementById('warning').innerHTML=warnings[row.video]||'';
const cards=[['Fully correct rallies',row.fully_correct_rallies,row.labelled_rallies],['Contacts: time + player',row.confirmed_contacts,row.labelled_contacts],['Serves: time + player',row.confirmed_serves,row.labelled_rallies]];
document.getElementById('cards').innerHTML=cards.map(([title,count,total])=>'<div class="card">'+title+'<span class="big">'+rate(count,total)+'</span><span class="detail">'+number(count)+' / '+number(total)+'</span></div>').join('');
document.getElementById('courts').textContent='Fully correct rallies with old courts (fresh refit): '+number(row.old_court_correct)+'. With new courts: '+number(row.fully_correct_rallies)+'. Rallies gained: '+number(row.gained)+'; lost: '+number(row.lost)+'.';
const targets=['Far','Near'];const predictions=['Far','Near','Unassigned','Missed prediction'];
document.getElementById('matrix').innerHTML=targets.map(target=>{const counts=predictions.map(prediction=>row.matrix[target+'|'+prediction]||0);const total=counts.reduce((sum,value)=>sum+value,0);return '<tr><th scope="row">'+target+' player</th>'+counts.map(count=>'<td style="background:rgba(var(--shade),'+(total?0.05+0.22*count/total:0)+')">'+number(count)+'</td>').join('')+'</tr>';}).join('');
document.getElementById('extras').textContent=number(row.unmatched_events)+' emitted events have no cleaned-label match. These are separate from the matrix, which starts from labelled hits. Missing labels can create unmatched events; they are not all proven false physical hits.';
const states=[['Court rejected','court_rejected'],['Court accepted; a player pick missing','accepted_missing_pick'],['Court accepted; both players picked','accepted_both_picked']];
document.getElementById('inputs').innerHTML=states.map(([title,key])=>'<tr><th scope="row">'+title+'</th><td>'+number(row['matched_'+key])+'</td><td>'+number(row['missed_'+key])+'</td><td>'+number(row['labelled_'+key])+'</td></tr>').join('');
const rallyRows=[['Whole rally fits in a clip',row.contained_rallies],['Exact contact sequence; players not required',row.exact_sequence_rallies],['Fully correct contact sequence and players',row.fully_correct_rallies]];
document.getElementById('rallies').innerHTML=rallyRows.map(([title,count])=>'<tr><th scope="row">'+title+'</th><td>'+number(count)+' / '+number(row.labelled_rallies)+' ('+rate(count,row.labelled_rallies)+')</td></tr>').join('');
document.getElementById('selection').textContent='The historical review cutoff keeps '+number(row.selected_correct+row.selected_wrong+row.selected_unknown)+' clips: '+number(row.selected_correct)+' correct, '+number(row.selected_wrong)+' wrong and '+number(row.selected_unknown)+' unknown.';
}
video.addEventListener('change',update);update();
</script></main></body></html>'''


def test_rows(name: str) -> pd.DataFrame:
    table = pd.read_csv(ROOT / f"results/{name}.csv.gz", dtype={"video": str})
    table = table[(table.split == "test") & (table.tolerance_base30 == 10)]
    return table.assign(video=table.video.astype(int))


def video_record(video: int, contacts: pd.DataFrame, rallies: pd.DataFrame, per_video: pd.Series,
                 court_change: dict) -> dict:
    target = contacts.target_side.map(SIDE_NAMES)
    answer = contacts.predicted_side.map(SIDE_NAMES).fillna("Unassigned").where(contacts.matched, "Missed prediction")
    matrix = pd.crosstab(target, answer).stack()
    record = {
        "video": video,
        "labelled_rallies": len(rallies),
        "fully_correct_rallies": int(rallies.fully_correct.sum()),
        "exact_sequence_rallies": int(rallies.exact_sequence.sum()),
        "contained_rallies": int(rallies.contained.sum()),
        "labelled_contacts": len(contacts),
        "confirmed_contacts": int(contacts.player_correct.sum()),
        "confirmed_serves": int(contacts[contacts.position == "serve"].player_correct.sum()),
        "unmatched_events": int(per_video.predictions - per_video.matched_contacts),
        "selected_correct": int(per_video.selected_correct),
        "selected_wrong": int(per_video.selected_wrong),
        "selected_unknown": int(per_video.selected_unknown),
        "old_court_correct": court_change["old"],
        "gained": court_change["gained"],
        "lost": court_change["lost"],
        "matrix": {f"{labelled}|{returned}": int(count) for (labelled, returned), count in matrix.items()},
    }
    assert record["fully_correct_rallies"] == per_video.correct_rallies == court_change["new"]
    for state, key in STATE_KEYS.items():
        in_state = contacts.input_state == state
        record[f"labelled_{key}"] = int(in_state.sum())
        record[f"matched_{key}"] = int((in_state & contacts.matched).sum())
        record[f"missed_{key}"] = int((in_state & ~contacts.matched).sum())
    return record


def run() -> None:
    context = pd.read_csv(ROOT / "results/contexts.csv.gz")
    contacts = test_rows("contacts").merge(context, on=["video", "source_frame"], validate="many_to_one")
    rallies = test_rows("rallies")
    per_video = pd.read_csv(ROOT / "results/per_video.csv.gz", index_col="video")
    with gzip.open(ROOT / "results/summary.json.gz", "rt") as source:
        court_changes = json.load(source)["court_comparison"]["fresh_old_courts"]["per_video"]
    records = [video_record(int(video), group, rallies[rallies.video == video], per_video.loc[video],
                            court_changes[str(video)])
               for video, group in contacts.groupby("video")]
    assert len(records) == 46
    assert sum(record["labelled_contacts"] for record in records) == 37184
    payload = json.dumps(records, separators=(",", ":"), allow_nan=False)
    (ROOT / "VIDEO_BREAKDOWN.html").write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print("Wrote 46 videos; contact totals match the headline count")


if __name__ == "__main__":
    run()
