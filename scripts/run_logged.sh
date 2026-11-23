
#!/usr/bin/env bash
set -o pipefail

# Usage: bash scripts/run_logged.sh <command> [args...]

if [ "$#" -eq 0 ]; then
    echo "Usage: bash scripts/run_logged.sh <command> [args...]"
    exit 1
fi

LOG_DIR="${LOG_DIR:-runs/multimodal}"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
RUN_NAME="${RUN_NAME:-experiment}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Command: $*" | tee -a "$LOG_FILE"

"$@" 2>&1 | tee -a "$LOG_FILE"

STATUS=${PIPESTATUS[0]}
echo "Exit status: $STATUS" | tee -a "$LOG_FILE"

exit "$STATUS"
