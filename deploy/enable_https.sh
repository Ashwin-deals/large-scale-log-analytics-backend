#!/usr/bin/env bash
# HTTPS for morphguard.duckdns.org (status doc G3). Run ON the VM after the
# firewall allows port 443 (see deploy/README.md, step 1). Uses certbot's nginx
# plugin, which edits the existing nginx site, adds the certificate and a
# redirect from http, and installs a renewal timer.
#
#   EMAIL=you@example.com bash enable_https.sh
#   DRY_RUN=1 ... bash enable_https.sh      # certbot --dry-run against staging
set -euo pipefail
DOMAIN=${DOMAIN:-morphguard.duckdns.org}
EMAIL=${EMAIL:?set EMAIL for Let\'s Encrypt expiry notices}
DRY_RUN=${DRY_RUN:-0}

command -v nginx >/dev/null || { echo "nginx not installed" >&2; exit 1; }
sudo nginx -t
# The name must point at this VM before Let's Encrypt can validate it.
resolved=$(getent hosts "$DOMAIN" | awk '{print $1}' | head -1)
mine=$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip || true)
echo "$DOMAIN -> $resolved; this VM's external IP: ${mine:-unknown}"
[ -n "$mine" ] && [ "$resolved" != "$mine" ] && { echo "DNS does not point here yet; fix DuckDNS first" >&2; exit 1; }

command -v certbot >/dev/null || sudo apt-get install -y certbot python3-certbot-nginx
if [ "$DRY_RUN" = 1 ]; then
  sudo certbot certonly --nginx -d "$DOMAIN" --dry-run --agree-tos -m "$EMAIL" -n
else
  sudo certbot --nginx -d "$DOMAIN" --redirect --agree-tos -m "$EMAIL" -n
  sudo systemctl reload nginx
  curl -fsS "https://$DOMAIN/api/health" && echo && echo "HTTPS ok"
fi
