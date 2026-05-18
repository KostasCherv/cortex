#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="source ${ROOT_DIR}/.venv/bin/activate"

escape_for_applescript() {
  local cmd="$1"
  local escaped="${cmd//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"
  printf '%s' "$escaped"
}

CMD_API="$(escape_for_applescript "cd ${ROOT_DIR} && ${VENV_ACTIVATE} && INNGEST_DEV=1 uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000 --reload")"
CMD_INNGEST="$(escape_for_applescript "cd ${ROOT_DIR} && npx --ignore-scripts=false inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery")"

echo "Starting all backend components..."

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
    end tell
end tell
APPLESCRIPT

echo "All backend components started in iTerm2 tabs (API | Inngest)."
echo "The Inngest dev server fires the outbox-dispatcher cron automatically every 2 minutes."
echo "Close the window or press Ctrl+C in each tab to stop the services."
