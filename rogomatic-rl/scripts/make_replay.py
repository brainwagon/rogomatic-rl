"""Turn a recorded game into an HTML replay that can be embedded in a web page.

Record a game first:

    python scripts/eval_ppo.py --record game.json --record-best 20

then make the snippet:

    python scripts/make_replay.py game.json -o replay.html --id rogue-1

The output is a self-contained block of HTML, CSS and JavaScript with no
outside dependencies.  Paste it into a page or a Markdown post as is; it
contains no blank lines, so Markdown treats it as one raw-HTML block.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rogomatic_rl.replay import describe_outcome, load, render_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make an embeddable HTML replay from a recorded Rogue game."
    )
    parser.add_argument("recording", help="JSON file written by eval_ppo.py --record")
    parser.add_argument(
        "-o", "--out", default=None, help="Output file (default: recording with .html)"
    )
    parser.add_argument(
        "--id",
        default="rogue-replay",
        help="HTML id of the replay; must be unique on the page it is pasted into",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Text under the replay (default: a summary of how the game went)",
    )
    parser.add_argument(
        "--fps", type=int, default=12, help="Moves per second at 1x speed"
    )
    args = parser.parse_args()

    recording = load(args.recording)
    out = Path(args.out) if args.out else Path(args.recording).with_suffix(".html")
    snippet = render_html(
        recording, element_id=args.id, caption=args.caption, fps=args.fps
    )
    out.write_text(snippet)
    print(f"{describe_outcome(recording.outcome)}")
    print(f"Wrote {out} ({len(snippet) / 1024:.0f} KB, {len(recording.frames)} screens)")


if __name__ == "__main__":
    main()
