#!/bin/bash
# One detection process per video on Carmack's GPU. Usage: run_extract.sh WORK_DIR CHECKOUT_ROOT
set -u
WORK=$1
CHECKOUT=$2
NVIDIA=$HOME/.venvs/venv-rtmlib/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NVIDIA/cudnn/lib:$NVIDIA/cu13/lib:/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}
PLAYER=${REMOTE_ROOT}/player_guided_20260908/videos
SSET=${SHUTTLESET_SOURCES}
declare -A VIDEOS=(
    [gx]=${REMOTE_ROOT}/paint_geometry_20260909/gx_extension/videos/gxBQ_HwdgN4.mp4
    [am1]=$PLAYER/am1_h264.mp4 [am2]=$PLAYER/am2_h264.mp4 [am3]=$PLAYER/am3_h264.mp4
    [letterboxed]=$PLAYER/Cb-xs5rPyxI_gameplay.mp4
    [sset03]="$SSET/3 Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final.mp4"
    [sset21]="$SSET/21 An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals.mp4"
)
mkdir -p "$WORK/logs"
for KEY in "${!VIDEOS[@]}"; do
    ~/.venvs/venv-rtmlib/bin/python "$WORK/extract_window_people.py" "$WORK/views.json" "$KEY" "${VIDEOS[$KEY]}" \
        "$CHECKOUT" "$WORK/people" > "$WORK/logs/$KEY.log" 2>&1 &
done
wait
touch "$WORK/extract.done"
