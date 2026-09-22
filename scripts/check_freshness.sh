#!/bin/bash
# Liquidhub data freshness check.
#
# - Healthy: silent, exit 0.
# - Stale:   prints "[STALE] ..." lines to stdout, exit 1, and Telegrams the
#            shared iMessage bot ON STATE CHANGE only (to avoid spam).
#
# Cron:
#   */15 * * * * .../check_freshness.sh 2>&1 | /usr/bin/logger -t liquidhub-freshness

set -u

DB_URL="${DB_URL:-postgresql://liquidhub:liquidhub@localhost:5432/liquidhub}"
OHLCV_5M_MAX="${OHLCV_5M_MAX:-600}"
DERIV_MAX="${DERIV_MAX:-300}"
PAIR_OHLCV_MAX="${PAIR_OHLCV_MAX:-1200}"

STATE_FILE="${STATE_FILE:-/tmp/liquidhub-freshness.state}"
TELEGRAM_TOKEN_FILE="${TELEGRAM_TOKEN_FILE:-/home/adas/MotherlodeApp/iMessage/BotToken}"
IMESSAGE_UNIT="${IMESSAGE_UNIT:-/etc/systemd/system/imessage.service}"

PSQL="psql -tAq -F| -v ON_ERROR_STOP=1 $DB_URL"

issues=()

check() {
    local label="$1" max="$2" sql="$3"
    local age
    age=$($PSQL -c "$sql") || { issues+=("[FAIL] $label: query error"); return; }
    if [ -z "$age" ]; then
        issues+=("[STALE] $label: no rows at all")
        return
    fi
    age=${age%.*}
    if [ "$age" -gt "$max" ]; then
        issues+=("[STALE] $label: age=${age}s (limit ${max}s)")
    fi
}

check "ohlcv 5m" "$OHLCV_5M_MAX" \
    "SELECT EXTRACT(EPOCH FROM (now() - max(timestamp)))::bigint FROM ohlcv_data WHERE timeframe='5m';"
check "open_interest" "$DERIV_MAX" \
    "SELECT EXTRACT(EPOCH FROM (now() - max(timestamp)))::bigint FROM open_interest_snapshots;"
check "funding_rate" "$DERIV_MAX" \
    "SELECT EXTRACT(EPOCH FROM (now() - max(timestamp)))::bigint FROM funding_rate_snapshots;"

stale_pairs=$($PSQL <<SQL
SELECT mp.symbol || ' age=' || EXTRACT(EPOCH FROM (now() - max(o.timestamp)))::bigint || 's'
FROM ohlcv_data o JOIN market_pairs mp ON mp.id = o.pair_id
WHERE mp.active AND o.timeframe = '5m'
GROUP BY mp.symbol
HAVING EXTRACT(EPOCH FROM (now() - max(o.timestamp))) > $PAIR_OHLCV_MAX
ORDER BY 1;
SQL
)
if [ -n "$stale_pairs" ]; then
    while IFS= read -r p; do
        issues+=("[STALE] per-pair 5m: $p")
    done <<< "$stale_pairs"
fi

# State signature for change detection
if [ "${#issues[@]}" -eq 0 ]; then
    cur_state="ok"
    summary="OK"
else
    # hash the issue set so identical-stale-state runs stay quiet
    cur_state=$(printf '%s\n' "${issues[@]}" | sha256sum | cut -c1-16)
    summary=$(printf '%s\n' "${issues[@]}")
fi

prev_state=$(cat "$STATE_FILE" 2>/dev/null || echo "ok")
echo "$cur_state" > "$STATE_FILE"

notify_telegram() {
    local body="$1"
    local token chat_id host
    token=$(grep -oE '[0-9]{6,}:[A-Za-z0-9_-]{30,}' "$TELEGRAM_TOKEN_FILE" 2>/dev/null | head -1)
    chat_id=$(grep -oE 'TELEGRAM_CHAT_ID=[0-9]+' "$IMESSAGE_UNIT" 2>/dev/null | head -1 | cut -d= -f2)
    [ -z "$token" ] || [ -z "$chat_id" ] && return 0
    host=$(hostname -s)
    curl -s -m 10 -X POST "https://api.telegram.org/bot${token}/sendMessage" \
        --data-urlencode "chat_id=${chat_id}" \
        --data-urlencode "parse_mode=Markdown" \
        --data-urlencode "text=*[${host}] liquidhub-freshness*
${body}" >/dev/null || true
}

if [ "$cur_state" != "$prev_state" ]; then
    if [ "$cur_state" = "ok" ]; then
        notify_telegram "✅ recovered — all collectors fresh"
    else
        notify_telegram "⚠️ ${#issues[@]} issue(s):
\`\`\`
${summary}
\`\`\`"
    fi
fi

if [ "${#issues[@]}" -eq 0 ]; then
    exit 0
fi

printf '%s\n' "${issues[@]}"
echo "---"
echo "${#issues[@]} issue(s) at $(date -Is) (state=$cur_state, prev=$prev_state)"
exit 1
