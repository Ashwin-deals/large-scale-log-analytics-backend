#!/usr/bin/env bash
# Redeploy MorphGuard on the GCP VM (status doc G1, plus the G2 stream).
# Run ON the VM, from anywhere:
#
#   bash vm_deploy.sh                 # deploy
#   DRY_RUN=1 bash vm_deploy.sh       # print every step, change nothing
#
# Idempotent: safe to run again after a failure. It never writes .env (secrets)
# -- it checks it and stops with what to fix. Every path is overridable:
#   APP_ROOT, BACKEND_DIR, FRONTEND_DIR, WEB_ROOT, DOMAIN, SERVICE, BRANCH,
#   WITH_STREAM=1 (also start Kafka + the stream processor), SKIP_DATA=1.
set -euo pipefail

APP_ROOT=${APP_ROOT:-$HOME}
BACKEND_DIR=${BACKEND_DIR:-$APP_ROOT/large-scale-log-analytics-backend}
FRONTEND_DIR=${FRONTEND_DIR:-$APP_ROOT/large-scale-log-analytics-frontend}
WEB_ROOT=${WEB_ROOT:-/var/www/morphguard}
DOMAIN=${DOMAIN:-morphguard.duckdns.org}
SERVICE=${SERVICE:-morphguard}
BRANCH=${BRANCH:-main}
WITH_STREAM=${WITH_STREAM:-0}
SKIP_DATA=${SKIP_DATA:-0}
DRY_RUN=${DRY_RUN:-0}
SOURCES=(openstack apache cloudtrail django)

say()  { printf '\n==> %s\n' "$*"; }
die()  { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] %s\n' "$*"; else "$@"; fi; }

say "Checks"
[ -d "$BACKEND_DIR/.git" ]  || die "backend checkout not found at $BACKEND_DIR (set BACKEND_DIR)"
[ -d "$FRONTEND_DIR/.git" ] || die "frontend checkout not found at $FRONTEND_DIR (set FRONTEND_DIR)"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || die "Python 3.11+ needed (scikit-learn 1.8.0 and the saved models); found $(python3 --version 2>&1)"
command -v npm >/dev/null || die "npm not installed"
ENV_FILE=${ENV_FILE:-$BACKEND_DIR/.env}
[ -f "$ENV_FILE" ] || die "$ENV_FILE missing: copy .env.example and set MONGO_URI and JWT_SECRET"
grep -q '^MONGO_URI=mongodb+srv://' "$ENV_FILE" || echo "warn: MONGO_URI does not look like an Atlas URI"
grep -Eq '^JWT_SECRET=change-this-secret$|^JWT_SECRET=$' "$ENV_FILE" && die "JWT_SECRET is the default: set a new random secret (it was exposed; rotate it)"
grep -Eq "^CORS_ORIGIN=.*https://$DOMAIN" "$ENV_FILE" \
  || die "CORS_ORIGIN in .env must include https://$DOMAIN (comma-separated list)"
for repo in "$BACKEND_DIR" "$FRONTEND_DIR"; do
  [ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ] \
    || die "$repo has local changes to tracked files; commit or stash them first"
done
echo "ok"

say "1/6 Code: fast-forward both repos to origin/$BRANCH"
for repo in "$BACKEND_DIR" "$FRONTEND_DIR"; do
  run git -C "$repo" fetch origin
  run git -C "$repo" checkout "$BRANCH"
  run git -C "$repo" merge --ff-only "origin/$BRANCH"
done

say "2/6 Backend dependencies"
cd "$BACKEND_DIR"
[ -d venv ] || run python3 -m venv venv
run venv/bin/pip install -q -r requirements.txt

if [ "$SKIP_DATA" != 1 ]; then
  say "3/6 Data from Hugging Face (raw logs, HDFS derived files, every vocabulary)"
  run venv/bin/python scripts/download_dataset.py --source all --with-derived

  say "4/6 Rebuild each source's features, labels and block metadata"
  # The pipeline retrains deterministically (seed 42), reproducing the committed
  # models byte for byte; restoring them afterwards keeps the checkout clean so
  # the next fast-forward cannot conflict. Only the ignored data files remain.
  for source in "${SOURCES[@]}"; do
    run venv/bin/python scripts/run_source_pipeline.py --source "$source"
  done
  run git checkout -- data/models data/evaluation data/storage
  say "   Check every deployed model against its vocabulary"
  run venv/bin/python - <<'PY'
import json, sys, warnings
from pathlib import Path
import joblib
sys.path.insert(0, ".")
from detection.predict import check_encoder_compatibility
from source_context import SOURCES, get_context
bad = []
for name in SOURCES:
    ctx = get_context(name)
    current = json.loads(ctx.current_version_path.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            check_encoder_compatibility(joblib.load(current["model_path"]), vocab_path=ctx.vocab_path)
            print(f"  {name}: V{current['version']} ok")
        except Exception as exc:
            bad.append(name); print(f"  {name}: {exc}")
sys.exit(1 if bad else 0)
PY
else
  say "3-4/6 Data skipped (SKIP_DATA=1)"
fi

say "5/6 Frontend build -> $WEB_ROOT"
cd "$FRONTEND_DIR"
run npm ci --silent
run env VITE_API_BASE="https://$DOMAIN" npm run build
run sudo mkdir -p "$WEB_ROOT"
run sudo rsync -a --delete dist/ "$WEB_ROOT/"

say "6/6 Restart the API ($SERVICE) and check it"
run sudo systemctl restart "$SERVICE"
if [ "$DRY_RUN" != 1 ]; then
  for _ in $(seq 1 20); do
    curl -fsS "https://$DOMAIN/api/health" >/dev/null 2>&1 && break
    curl -fsS "http://127.0.0.1/api/health" >/dev/null 2>&1 && break
    sleep 2
  done
  curl -fsS "https://$DOMAIN/api/health" 2>/dev/null \
    || curl -fsS "http://127.0.0.1/api/health" \
    || die "API not healthy: sudo journalctl -u $SERVICE -n 80"
  echo
fi

if [ "$WITH_STREAM" = 1 ]; then
  say "Stream (G2): Kafka in Docker + the HDFS stream processor service"
  cd "$BACKEND_DIR"
  run sudo docker compose up -d
  unit=$(sed -e "s#__USER__#$(id -un)#g" -e "s#__BACKEND__#$BACKEND_DIR#g" deploy/morphguard-stream@.service)
  if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] install unit:\n%s\n' "$unit"; else
    printf '%s\n' "$unit" | sudo tee /etc/systemd/system/morphguard-stream@.service >/dev/null; fi
  run sudo systemctl daemon-reload
  run sudo systemctl enable --now morphguard-stream@hdfs
fi

say "Deployed $(git -C "$BACKEND_DIR" log --oneline -1) / $(git -C "$FRONTEND_DIR" log --oneline -1)"
echo "Open https://$DOMAIN and run docs/review/walkthrough_checklist.md against it."
