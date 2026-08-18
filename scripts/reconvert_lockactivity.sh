#!/usr/bin/env bash
# Re-convert the four _data scenes that ran alignement_mask="sam_and_activity"
# before SPAG_LOCK_ACTIVITY existed, so their background depth is locked to
# depth_ref on activity-only pixels (see spag4d/video.py's SPAG_LOCK_ACTIVITY
# comment). accident_electrique_02 is deliberately absent: it ran
# alignement_mask="sam", where the flag is a no-op.
#
# Writes in place into _data/<scene>/. SPAG4D only creates images/, depth_maps/
# and gaussians/ -- it never removes anything -- so each scene's colmap/ rig and
# normalize_transform.npy survive untouched, which is what keeps the rebuilt NPZ
# inits comparable to the ones measured before.
#
# --stride matches whatever produced each scene's existing frame_0.ply
# (fast2 was built at 2, the rest at 4).
set -euo pipefail

PY=/home/mb273924/miniforge3/envs/spag4d/bin/python
VIDEOS=/raid/mb273924/_DATASETS/uptale/data/videos
DATA=/raid/mb273924/FreeTimeGsVanilla/_data
export CUDA_VISIBLE_DEVICES=1

# The source clips are split across two folders (data/videos/ and data/ itself).
find_video() {
    local name=$1
    for d in "$VIDEOS" "$(dirname "$VIDEOS")"; do
        [ -f "$d/$name.mp4" ] && { echo "$d/$name.mp4"; return; }
    done
    echo "no source video found for $name" >&2
    return 1
}

convert_scene() {
    local name=$1 stride=$2
    local video
    video=$(find_video "$name")
    echo "=== [$(date +%H:%M:%S)] $name (stride=$stride) <- $video"
    "$PY" -m spag4d convert \
        "$video" \
        "$DATA/$name" \
        --stride="$stride" \
        --skip-step 1 \
        --depth-correction bglock \
        --alignement-mask sam_and_activity \
        --depth-raw
    echo "=== [$(date +%H:%M:%S)] $name done"
}

convert_scene accident_electrique_fast2 2
convert_scene tissc 4
convert_scene scene01 4
echo "=== all scenes reconverted"
