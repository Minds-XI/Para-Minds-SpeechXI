#!/usr/bin/env bash
set -euo pipefail

# -------- Config --------
CONNECT_URL="${CONNECT_URL:-http://connect:8083}"
CONNECTOR_DIR="${CONNECTOR_DIR:-/connectors}"
# Space-separated list of required plugin classes (optional)
REQUIRED_PLUGIN_CLASSES="${REQUIRED_PLUGIN_CLASSES:-com.mongodb.kafka.connect.MongoSinkConnector}"

# -------- Tools --------
if command -v apt-get >/dev/null 2>&1; then
  apt-get update && apt-get install -y curl jq >/dev/null
elif command -v apk >/dev/null 2>&1; then
  apk add --no-cache curl jq >/dev/null
fi

# -------- Wait for Connect REST --------
echo "[init] Waiting for Connect at ${CONNECT_URL} ..."
until curl -fsS "${CONNECT_URL}/connector-plugins" >/dev/null; do
  sleep 2
done
echo "[init] Connect REST is up."

# -------- Ensure required plugins are available --------
if [[ -n "${REQUIRED_PLUGIN_CLASSES}" ]]; then
  for cls in ${REQUIRED_PLUGIN_CLASSES}; do
    if ! curl -fsS "${CONNECT_URL}/connector-plugins" \
      | jq -e --arg C "$cls" '.[] | select(.class == $C)' >/dev/null; then
      echo "[init] Required plugin not found: ${cls}" >&2
      echo "[init] Available plugins:" >&2
      curl -fsS "${CONNECT_URL}/connector-plugins" | jq . >&2
      exit 1
    fi
  done
  echo "[init] Required plugins detected: ${REQUIRED_PLUGIN_CLASSES}"
fi

# -------- Apply all connector JSON configs --------
shopt -s nullglob
applied=0
for f in "${CONNECTOR_DIR}"/*.json; do
  name="$(jq -r '.name // empty' "$f")"
  if [[ -z "${name}" ]]; then
    echo "[init] Skipping ${f} (missing .name)"
    continue
  fi

  echo "[init] Processing connector: ${name}"

  http_code="$(curl -s -o /dev/null -w "%{http_code}" "${CONNECT_URL}/connectors/${name}")"
  if [[ "${http_code}" == "200" ]]; then
    echo "[init] Updating connector: ${name}"
    jq -r '.config' "$f" > /tmp/config.json
    curl -fsS -X PUT "${CONNECT_URL}/connectors/${name}/config" \
      -H "Content-Type: application/json" \
      --data @/tmp/config.json >/dev/null
  else
    echo "[init] Creating connector: ${name}"
    curl -fsS -X POST "${CONNECT_URL}/connectors" \
      -H "Content-Type: application/json" \
      --data @"$f" >/dev/null
  fi
  ((applied++))
done

echo "[init] Done. Applied ${applied} connector definition(s)."
