#!/usr/bin/env bash
# Restore drill: seed -> dump+encrypt -> mutate -> decrypt+restore -> verify.
#
# Proves the backup pipeline produces a restorable artifact. Requires a
# scratch PostgreSQL (NOT a production database!) reachable via
# PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE, plus pg_dump/pg_restore and the
# repo venv (for the crypto helpers).
#
# Example:
#   docker run -d --name agentix-pg -e POSTGRES_PASSWORD=test -p 55432:5432 postgres:16
#   export PGHOST=localhost PGPORT=55432 PGUSER=postgres PGPASSWORD=test PGDATABASE=postgres
#   export BACKUP_ENCRYPTION_KEY="$(openssl rand -hex 32)"
#   ./tests/e2e/restore_drill.sh
set -euo pipefail

: "${PGHOST:?set PGHOST}" "${PGPORT:?set PGPORT}" "${PGUSER:?set PGUSER}" \
  "${PGPASSWORD:?set PGPASSWORD}" "${PGDATABASE:?set PGDATABASE}" \
  "${BACKUP_ENCRYPTION_KEY:?set BACKUP_ENCRYPTION_KEY}"

PY=.venv/bin/python
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export PGPASSWORD

echo "==> 1. seed"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -q <<'SQL'
DROP TABLE IF EXISTS drill_marks;
CREATE TABLE drill_marks (id serial PRIMARY KEY, note text);
INSERT INTO drill_marks (note) VALUES ('before-backup');
SQL

echo "==> 2. dump (custom format) + encrypt"
pg_dump -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -Fc -f "$TMP/drill.dump"
$PY - "$TMP/drill.dump" "$TMP/drill.dump.enc" <<'PY'
import sys
from app.backup_crypto import encrypt_file, load_encryption_key
info = encrypt_file(sys.argv[1], sys.argv[2], load_encryption_key())
print(f"    sha256_plain={info['sha256_plain'][:16]}… size={info['size_plain']}B")
PY

echo "==> 3. mutate (drop + diverge)"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -q <<'SQL'
DROP TABLE drill_marks;
CREATE TABLE drill_marks (id serial PRIMARY KEY, note text);
INSERT INTO drill_marks (note) VALUES ('after-mutation');
SQL

echo "==> 4. decrypt + verify checksum + restore"
$PY - "$TMP/drill.dump.enc" "$TMP/drill.dec.dump" <<'PY'
import sys
from app.backup_crypto import decrypt_file, load_encryption_key
info = decrypt_file(sys.argv[1], sys.argv[2], load_encryption_key())
print(f"    decrypted ok, sha256_plain={info['sha256_plain'][:16]}…")
PY
pg_restore -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    --clean --if-exists --no-owner --no-privileges "$TMP/drill.dec.dump"

echo "==> 5. verify"
NOTE=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -tAc \
    "SELECT note FROM drill_marks ORDER BY id")
if [ "$NOTE" = "before-backup" ]; then
    echo "PASS: restore returned the pre-backup state ('$NOTE')"
else
    echo "FAIL: unexpected content after restore: '$NOTE'"
    exit 1
fi
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -q -c "DROP TABLE drill_marks"
echo "DONE"
