#!/bin/bash
# Run VLM evaluation for RMMBench
# Leave a value empty to use the default from scripts/vlm_eval.py.
# Toggle switches: set to "1" to enable, "0" or empty to disable.

SAVE_DIR=""

VLM_URL=""

API_KEY=""

HISTORY_MAXLEN=""

MAX_SKILLS_NUM=""

NUM_WORKERS=""

USE_GRASPNET="0"

SAVE_VIDEO="0"

cd "$(dirname "$0")" || exit 1

ARGS=()

[ -n "${SAVE_DIR}" ] && ARGS+=(--save-dir "${SAVE_DIR}")
[ -n "${VLM_URL}" ] && ARGS+=(--vlm-url "${VLM_URL}")
[ -n "${API_KEY}" ] && ARGS+=(--api-key "${API_KEY}")
[ -n "${HISTORY_MAXLEN}" ] && ARGS+=(--history-maxlen "${HISTORY_MAXLEN}")
[ -n "${MAX_SKILLS_NUM}" ] && ARGS+=(--max-skills-num "${MAX_SKILLS_NUM}")
[ -n "${NUM_WORKERS}" ] && ARGS+=(--num-workers "${NUM_WORKERS}")

[ "${USE_GRASPNET}" = "1" ] && ARGS+=(--use-graspnet)
[ "${SAVE_VIDEO}" = "1" ] && ARGS+=(--save-video)

python scripts/vlm_eval.py "${ARGS[@]}"
