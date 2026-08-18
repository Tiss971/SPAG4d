"""Rebuild a benchmarks/<scene>/comparison.html from the scene01 template,
substituting per-scene DATA/FG_DATA JSON, the embedded flicker video, and the
scene-specific text (name, video path, frame/point counts, generator script name).

Usage:
    python scripts/build_comparison_html.py scene01 \\
        /raid/mb273924/_DATASETS/uptale/data/videos/scene01.mp4
    python scripts/build_comparison_html.py accident_electrique_02 \\
        /raid/mb273924/_DATASETS/uptale/data/videos/accident_electrique_02.mp4 \\
        --template benchmarks/scene01/comparison.html
"""
import argparse
import base64
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="Scene name, e.g. scene01 or accident_electrique_02")
    parser.add_argument("video_path", help="Original source video path (for the footer text)")
    parser.add_argument("--template", type=Path, default=Path("benchmarks/scene01/comparison.html"))
    parser.add_argument("--bench-root", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()

    scene_dir = args.bench_root / args.scene
    html = args.template.read_text()

    bg_affine = json.loads((scene_dir / "affine_stability.json").read_text())
    bg_bglock = json.loads((scene_dir / "bglock_stability.json").read_text())
    fg_affine = json.loads((scene_dir / "affine_fg_stability.json").read_text())
    fg_bglock = json.loads((scene_dir / "bglock_fg_stability.json").read_text())

    # 1. Replace DATA / FG_DATA JS blobs (each is a single-line `const NAME = {...};`).
    new_data = json.dumps({"affine": bg_affine, "bglock": bg_bglock})
    new_fg_data = json.dumps({"affine": fg_affine, "bglock": fg_bglock})
    html = re.sub(r"const DATA = \{.*?\};", f"const DATA = {new_data};", html, count=1, flags=re.DOTALL)
    html = re.sub(r"const FG_DATA = \{.*?\};", f"const FG_DATA = {new_fg_data};", html, count=1, flags=re.DOTALL)

    # 2. Replace embedded video base64.
    video_path = scene_dir / "fg_depth_flicker.mp4"
    b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    html = re.sub(
        r'src="data:video/mp4;base64,[^"]*"',
        f'src="data:video/mp4;base64,{b64}"',
        html, count=1,
    )

    # 3. Replace scene-name text occurrences (template is scene01-specific).
    template_scene = args.template.parent.name
    html = html.replace(template_scene, args.scene)

    # 4. Replace footer stats (frame/point counts differ per scene).
    n_frames = bg_affine["n_frames"]
    n_bg_points = bg_affine["n_points"]
    n_fg_seeded = fg_affine["n_points_seeded"]
    footer_re = re.compile(
        r"stride 8, skip_step 4, \d+ frames processed\. Background: ~[\d.]+k static points tracked\s*"
        r"\(grid-subsampled, background in every frame\), drift-window 15\. Foreground: ~[\d.]+k dynamic\s*"
        r"seed points per config, flow-propagated, tracks ≥20 frames kept, drift-window 15\."
    )
    new_footer_stats = (
        f"stride 8, skip_step 4, {n_frames} frames processed. Background: ~{n_bg_points/1000:.1f}k static points tracked\n"
        f"    (grid-subsampled, background in every frame), drift-window 15. Foreground: ~{n_fg_seeded/1000:.1f}k dynamic\n"
        f"    seed points per config, flow-propagated, tracks ≥20 frames kept, drift-window 15."
    )
    html, n = footer_re.subn(new_footer_stats, html, count=1)
    if n != 1:
        raise RuntimeError("footer stats pattern not found/replaced")

    html = html.replace(
        "scripts/run_scene01_bg_stability_bench.py",
        "scripts/run_bg_stability_bench.py",
    )
    html = html.replace(
        f"<span class=\"mono\">/raid/mb273924/_DATASETS/uptale/data/videos/{template_scene}.mp4</span>",
        f'<span class="mono">{args.video_path}</span>',
    )

    out_path = scene_dir / "comparison.html"
    out_path.write_text(html)
    print(f"Wrote {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
