"""Portable session review page. No server, external scripts, or network requests."""
from __future__ import annotations
from html import escape
import os
import tempfile
from pathlib import Path
from urllib.parse import quote


def write_review(report: dict, destination: str) -> Path:
    page = Path(destination).resolve()
    def link(path):
        return escape(quote(os.path.relpath(path, page.parent), safe='/'), quote=True)
    stats, tracking = report['stats'], report['tracking']
    rows = ''.join(
        f'<tr><td><button class="seek" data-time="{event["time_s"]:.6f}" '
        f'aria-label="Seek to {event["type"]} at {event["time_s"]:.2f} seconds">'
        f'{event["time_s"]:.2f}s ↗</button></td><td>{escape(event["type"].title())}</td>'
        f'<td>{"Unattributed" if event["player_id"] is None else "P" + str(event["player_id"])}</td>'
        f'<td>{event["frame_idx"]}</td></tr>' for event in report['events'])
    if not rows:
        rows = '<tr><td colspan="4" class="empty">No scoring events detected. Review the footage and tracking coverage before interpreting this as a score.</td></tr>'
    player_ids = sorted(set(stats['player_touch_counts']) | set(stats['player_errors']),
                        key=lambda p: -stats['player_touch_counts'].get(p, 0))
    players = ''.join(f'<li><strong>P{escape(str(pid))}</strong><span>{stats["player_touch_counts"].get(pid, 0)} touches</span><span>{stats["player_errors"].get(pid, 0)} errors</span></li>' for pid in player_ids)
    if not players:
        players = '<li>No player-attributed events yet.</li>'
    mode = 'Synthetic demo' if report['synthetic'] else 'Gameplay review'
    content = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Footbag · Session review</title><style>
:root{--ink:#203d48;--paper:#edf5f6;--white:#fff;--court:#be5b3a;--line:#bfd1d5;--muted:#526b73}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 'Avenir Next',Avenir,system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:36px 32px 60px}header{display:flex;align-items:center;justify-content:space-between;gap:20px;border-bottom:2px solid var(--ink);padding-bottom:20px}
.wordmark{white-space:nowrap;flex-shrink:0;font-weight:850;font-size:24px;letter-spacing:-1px}.mode{font:12px ui-monospace,monospace;text-transform:uppercase;letter-spacing:1px}
h1{font-family:'Arial Rounded MT Bold','Trebuchet MS',sans-serif;font-size:clamp(36px,6vw,66px);line-height:1.05;letter-spacing:-2px;margin:34px 0 12px}
.subtitle{margin:0 0 28px;color:var(--muted);overflow-wrap:anywhere}.layout{display:grid;grid-template-columns:minmax(0,2.2fr) minmax(240px,1fr);gap:28px}
video{display:block;width:100%;max-height:70vh;background:var(--ink);border-radius:10px}h2{font-size:19px;margin:0 0 14px}.caption{font-size:13px;color:var(--muted)}
.score{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:24px}
.score div{background:var(--white);padding:16px}.score strong{display:block;font:600 32px ui-monospace,monospace}.score span{font-size:13px;color:var(--muted)}
.coverage{height:10px;background:var(--line);border-radius:5px;overflow:hidden}.coverage div{height:100%;background:var(--court)}
ul{list-style:none;padding:0}li{display:flex;gap:10px;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--line);font-size:14px}
.events{margin-top:32px}table{width:100%;border-collapse:collapse;text-align:left}th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.8px}th,td{padding:12px 10px;border-bottom:1px solid var(--line)}
button,a{color:var(--court)}button{background:var(--white);border:1px solid var(--line);border-radius:6px;padding:8px 12px;cursor:pointer;font:inherit}button:hover{border-color:var(--court)}:focus-visible{outline:3px solid var(--court);outline-offset:3px}
.notice{padding:14px 16px;background:#f8e8df;border-left:4px solid var(--court);font-size:14px;margin-top:24px}.empty{color:var(--muted);padding:24px 10px}.links{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}footer{margin-top:40px;border-top:1px solid var(--line);padding-top:16px;font-size:13px;color:var(--muted)}
@media(max-width:760px){main{padding:24px 18px}.layout{grid-template-columns:1fr}header{align-items:flex-start}.mode{text-align:right}h1{letter-spacing:-1px}th,td{padding:10px 5px}.score{grid-template-columns:repeat(4,1fr)}.score div{padding:10px}.score strong{font-size:24px}}
@media(max-width:390px){.score{grid-template-columns:1fr 1fr}}
</style></head><body><main>
'''
    content += f'''<header><div class="wordmark">footbag / cv</div><div class="mode">{mode}<br>{report['duration_s']:.1f} seconds · {report['processed_frames']:,} frames</div></header>
<h1>Replay the round.</h1><p class="subtitle">Session review · {escape(Path(report['input']).name)}</p>
<div class="layout"><section aria-label="Session video"><video id="video" controls preload="metadata" src="{link(report['output'])}"></video>
<p id="video-error" class="notice" hidden>This browser cannot play the video codec. Open the annotated video in a desktop player, or install FFmpeg and run the analysis again for browser-compatible H.264 output.</p>
<p class="caption">Select an event below to jump to the moment. The overlay distinguishes observations from predicted positions.</p>
<div class="links"><a href="{link(report['output'])}" download>Save annotated video</a><a href="{link(page.with_suffix('.json'))}" download>Save session data</a></div>
</section><aside aria-label="Session statistics"><h2>On the scorecard</h2><div class="score">
<div><strong>{stats['total_touches']}</strong><span>Touches</span></div><div><strong>{stats['best_round_touches']}</strong><span>Best round</span></div>
<div><strong>{stats['total_drops']}</strong><span>Drops</span></div><div><strong>{stats['current_round_touches']}</strong><span>Current round</span></div></div>
<h2>Tracking coverage</h2><div class="coverage" role="img" aria-label="Observed on {tracking['observed_fraction']:.1%} of frames"><div style="width:{tracking['observed_fraction']*100:.2f}%"></div></div>
<p class="caption">{tracking['observed_fraction']:.1%} observed · {tracking['coasted_frames']:,} predicted frames · {tracking['missing_frames']:,} missing frames. Coverage is not an accuracy score.</p>
<h2>Players</h2><ul>{players}</ul></aside></div>
<div class="notice">{'Synthetic positions: these scores do not describe actual gameplay. ' if report['synthetic'] else ''}{'This is a shortened session. ' if report['stopped_early'] else ''}Scores are estimates. Missed detections can hide touches; ground calibration affects drops. Review events before using the totals.</div>
<section class="events"><h2>Event log</h2><table><thead><tr><th scope="col">Time</th><th scope="col">Event</th><th scope="col">Player</th><th scope="col">Frame</th></tr></thead><tbody>{rows}</tbody></table></section>
<footer>Processed locally · Ground line {report['config']['ground_line_ratio']:.0%} of frame height · Speed and height use assumed player height, not calibrated geometry.</footer>
'''
    content += '''</main><script>
const video = document.getElementById('video');
video.addEventListener('error', () => { document.getElementById('video-error').hidden = false; });
for (const button of document.querySelectorAll('.seek')) {
  button.addEventListener('click', () => {
    video.currentTime = Math.max(0, Number(button.dataset.time) - 0.5);
    video.focus();
    video.play().catch(() => { document.getElementById('video-error').hidden = false; });
  });
}
</script></body></html>'''
    page.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=page.parent,
                                         prefix=f".{page.name}-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, page)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return page
