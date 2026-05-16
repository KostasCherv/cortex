#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="${ROOT_DIR}/ui"
VENV_ACTIVATE="source ${ROOT_DIR}/.venv/bin/activate"

escape_for_applescript() {
  local cmd="$1"
  local escaped="${cmd//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"
  printf '%s' "$escaped"
}

CMD_API="$(escape_for_applescript "cd ${ROOT_DIR} && ${VENV_ACTIVATE} && INNGEST_DEV=1 python -m src.main serve --reload")"
CMD_INNGEST="$(escape_for_applescript "cd ${ROOT_DIR} && npx --ignore-scripts=false inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery")"
CMD_OUTBOX="$(escape_for_applescript "cd ${ROOT_DIR} && ${VENV_ACTIVATE} && while true; do INNGEST_DEV=1 python -m src.main rag-dispatch-outbox --limit 100; sleep 2; done")"
CMD_UI="$(escape_for_applescript "cd ${UI_DIR} && npm run dev")"

echo "Starting backend, workers, and frontend..."

osascript <<APPLESCRIPT
tell application "iTerm"
    tell (create window with default profile)
        tell current session
            write text "${CMD_API}"
        end tell
        create tab with default profile
        tell current session
            write text "${CMD_INNGEST}"
        end tell
        create tab with default profile
        tell current session
            write text "${CMD_OUTBOX}"
        end tell
        create tab with default profile
        tell current session
            write text "${CMD_UI}"
        end tell
    end tell
end tell
APPLESCRIPT

echo "All services started in iTerm2 tabs (API | Inngest | Outbox | UI)."
echo "Close the window or press Ctrl+C in each tab to stop."
