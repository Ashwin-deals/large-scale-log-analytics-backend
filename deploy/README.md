# Deploying MorphGuard to the GCP VM (status doc G1–G4)

| Setting | Value |
|---|---|
| VM | `morphguard-vm` in `asia-south1-a` |
| Project | `log-analytics-512705` |
| Address | static IP `34.14.128.116`, `morphguard.duckdns.org` |
| Serving | nginx serves the built React app and proxies `/api` to gunicorn (systemd service `morphguard`) |
| Accounts | MongoDB Atlas |

Each step says where to run it. The kit is made to be re-run: every script checks first and is safe to run again.

## 0. Before anything (A4, on your laptop and huggingface.co)

The Atlas password, a Hugging Face token and the VM's JWT secret were exposed in a chat. Rotate all three first.

1. **Atlas:** change the database user's password. Put the new URI in `.env` on your laptop and on the VM, as `MONGO_URI`.
2. **JWT secret:** generate a new one with `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`. Set it as `JWT_SECRET` in the VM's `.env`. Everyone has to sign in again afterwards.
3. **Hugging Face:** delete the old token at huggingface.co/settings/tokens.

`vm_deploy.sh` refuses to run while `JWT_SECRET` is still the default.

## 1. Firewall and VM size (laptop, `gcloud`)

```bash
gcloud config set project log-analytics-512705
gcloud compute instances describe morphguard-vm --zone asia-south1-a --format='value(machineType.basename(),status)'
gcloud compute firewall-rules list --format='table(name,sourceRanges.list(),allowed[].map().firewall_rule().list())'
gcloud compute firewall-rules create allow-https --allow=tcp:443 --target-tags=https-server --description="HTTPS for morphguard"   # if 443 is missing
gcloud compute instances add-tags morphguard-vm --zone asia-south1-a --tags=https-server
```

**VM size (G4):** the API alone fits in `e2-small`. Kafka, the API and the stream processor together need about 8 GB, which is `e2-standard-2`. To resize, stop the VM, then run `gcloud compute instances set-machine-type morphguard-vm --zone asia-south1-a --machine-type e2-standard-2`, then start it. The static IP stays.

## 2. Code, data and app (on the VM)

```bash
gcloud compute ssh morphguard-vm --zone asia-south1-a
# on the VM: the kit arrives with the first pull; run it from the checkout
cd ~/large-scale-log-analytics-backend && git fetch origin && git checkout main && git merge --ff-only origin/main
nano .env    # MONGO_URI (new), JWT_SECRET (new), CORS_ORIGIN=https://morphguard.duckdns.org
DRY_RUN=1 bash deploy/vm_deploy.sh        # read the plan
bash deploy/vm_deploy.sh                  # do it (about 10 minutes, most of it the data download)
```

**If the checkouts or web root are elsewhere,** set `APP_ROOT`, `BACKEND_DIR`, `FRONTEND_DIR` or `WEB_ROOT`. Find the current web root with `grep -R root /etc/nginx/sites-enabled/`.

**Rebuilt data:** the script rebuilds every source's data on the VM. It then restores the committed models, so `git status` stays clean.

## 3. HTTPS (G3, on the VM)

```bash
EMAIL=you@example.com DRY_RUN=1 bash deploy/enable_https.sh   # staging check
EMAIL=you@example.com bash deploy/enable_https.sh              # real certificate + http->https redirect
```

**Login over plain HTTP:** until HTTPS works, the frontend built in step 2 calls `https://morphguard.duckdns.org`. Run step 3 straight after step 2. Alternatively, build once with `DOMAIN=34.14.128.116` and an `http://` CORS entry.

## 4. Live stream on the VM (G2, optional)

```bash
WITH_STREAM=1 SKIP_DATA=1 bash deploy/vm_deploy.sh     # Kafka in Docker + morphguard-stream@hdfs
venv/bin/python -m streaming.producer --source hdfs --limit 200000 --rate 2000   # feed it for the demo
```

**Docker:** install it first with `sudo apt-get install -y docker.io docker-compose-v2`, then `sudo usermod -aG docker $USER`, then log out and back in.

**Other sources:** to stream another source, run `sudo systemctl enable --now morphguard-stream@django`.

## 5. Budget and after the review (G4, laptop / console)

- **Budget alert:** Billing → Budgets & alerts → a budget on the credit account, with alerts at 50%, 90% and 100%.
- **After the review:** stop the VM with `gcloud compute instances stop morphguard-vm --zone asia-south1-a`. The disk and static IP still bill a little, so release the IP once the project is finished.

## Checks after deploying

- `https://morphguard.duckdns.org` shows the padlock, and login works.
- `curl -s https://morphguard.duckdns.org/api/health` returns `{"status":"ok"}`.
- `docs/review/walkthrough_checklist.md` passes against the live site, for every source.
