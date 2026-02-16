#!/usr/bin/env bash
# poller_control.sh — Start, stop, check status, or tail logs of the GTFS poller VM service.
#
# Usage:
#   bash infra/poller_control.sh start
#   bash infra/poller_control.sh stop
#   bash infra/poller_control.sh status
#   bash infra/poller_control.sh logs          # last 50 lines
#   bash infra/poller_control.sh logs -f       # follow (Ctrl-C to quit)
#
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-realtime-headway-prediction}"
ZONE="us-east1-b"
VM_NAME="gtfs-poller"

ACTION="${1:-status}"
shift || true

ssh_cmd() {
    gcloud compute ssh "${VM_NAME}" \
        --zone="${ZONE}" \
        --project="${PROJECT_ID}" \
        --command="$1"
}

case "${ACTION}" in
    start)
        echo "Starting poller on ${VM_NAME}..."
        ssh_cmd "sudo systemctl start gtfs-poller"
        sleep 2
        ssh_cmd "sudo systemctl status gtfs-poller --no-pager"
        ;;
    stop)
        echo "Stopping poller on ${VM_NAME}..."
        ssh_cmd "sudo systemctl stop gtfs-poller"
        echo "Poller stopped."
        ;;
    restart)
        echo "Restarting poller on ${VM_NAME}..."
        ssh_cmd "sudo systemctl restart gtfs-poller"
        sleep 2
        ssh_cmd "sudo systemctl status gtfs-poller --no-pager"
        ;;
    status)
        ssh_cmd "sudo systemctl status gtfs-poller --no-pager" || true
        ;;
    logs)
        EXTRA_ARGS="${*}"
        if [[ "${EXTRA_ARGS}" == *"-f"* ]]; then
            echo "Following poller logs (Ctrl-C to stop)..."
            gcloud compute ssh "${VM_NAME}" \
                --zone="${ZONE}" \
                --project="${PROJECT_ID}" \
                --command="sudo journalctl -u gtfs-poller -f --no-pager"
        else
            ssh_cmd "sudo journalctl -u gtfs-poller -n 50 --no-pager"
        fi
        ;;
    update)
        echo "Updating poller code on ${VM_NAME}..."
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
        POLLER_DIR="${PROJECT_ROOT}/realtime_ingestion/poller"
        gcloud compute scp --recurse "${POLLER_DIR}" "${VM_NAME}:~/poller" \
            --zone="${ZONE}" --project="${PROJECT_ID}"
        ssh_cmd "cd ~/poller && source .venv/bin/activate && pip install --quiet -r requirements.txt"
        ssh_cmd "sudo systemctl restart gtfs-poller"
        sleep 2
        ssh_cmd "sudo systemctl status gtfs-poller --no-pager"
        echo "Poller updated and restarted."
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|update}"
        echo ""
        echo "  start    — start the poller service"
        echo "  stop     — stop the poller service"
        echo "  restart  — restart the poller service"
        echo "  status   — show service status"
        echo "  logs     — show last 50 log lines (add -f to follow)"
        echo "  update   — re-copy poller code from local, reinstall deps, restart"
        exit 1
        ;;
esac
