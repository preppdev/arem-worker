# Deploying a new AREM worker box

Bringing up a fresh Ubuntu host as a production worker is a single command
plus a one-time credential drop:

```bash
curl -sSL https://raw.githubusercontent.com/preppdev/arem-worker/main/provision.sh | bash
```

This is `provision.sh`. It's idempotent — re-run any time, it skips steps
that are already done.

## What `provision.sh` does (10 steps)

1. **Preflight** — confirms Ubuntu, sudo available, NVIDIA GPU present.
2. **System packages** — `apt install build-essential git curl rclone jq
   dkms python3.11 ubuntu-drivers-common gnome-terminal`.
3. **NVIDIA driver** — installs the 580-open cohort + the matching
   per-kernel module package for the current kernel + the HWE metapackage.
   `modprobe nvidia`.
4. **Repo clone** — `git clone preppdev/arem-worker` → `$HOME/arem-worker`.
5. **Python env** — miniconda3 in `$HOME/miniconda3`, creates the
   `arem-photo-ai` env from `requirements.txt` (torch+cu124, rawpy, etc).
6. **Credentials gate** — checks for `/etc/arem-worker.env` and
   `~/.config/rclone/rclone.conf`. Stops with copy-paste instructions if
   either is missing; re-run after staging them.
7. **systemd units + sudoers** — installs
   `arem-worker-local.service`, `arem-host-heartbeat.service`,
   `arem-host-heartbeat.timer`, and `/etc/sudoers.d/arem-worker`. Enables
   + starts both services.
8. **TUI autostart** — installs `~/.config/autostart/arem-fleet-tui.desktop`
   so the local status TUI launches on GNOME login.
9. **Model checkpoints** — `scripts/sync_checkpoints.sh` pulls the three
   inference checkpoints + the room classifier from R2.
10. **Verify** — prints service status and a reminder to check
    `/workers` on the dashboard within 60s.

## The one manual step: credentials

Two files can't be in the public repo:

- `/etc/arem-worker.env` — production env (DATABASE_URL, WORKER_TOKEN,
  EXTERNAL_API_TOKEN, R2 keys, Dropbox refresh token, Resend key, ...)
- `~/.config/rclone/rclone.conf` — Dropbox + R2 remote credentials

`provision.sh` stops at step 6 if these are missing and prints two
viable acquisition paths:

### (A) scp from an existing worker box

From your dev laptop (which can reach both):
```bash
scp jordan@<existing-worker>:/etc/arem-worker.env /tmp/arem-worker.env
scp jordan@<existing-worker>:~/.config/rclone/rclone.conf /tmp/rclone.conf
scp /tmp/arem-worker.env /tmp/rclone.conf jordan@<new-worker>:/tmp/
```

### (B) `vercel env pull` for `arem-worker.env` only

On your dev laptop, in the `arem-editing` dashboard repo:
```bash
vercel env pull /tmp/arem.env --environment=production --yes
scp /tmp/arem.env jordan@<new-worker>:/tmp/arem-worker.env
```
(rclone.conf still has to come from path A — Vercel env doesn't include
the rclone tokens.)

Then on the new box, install both files and re-run `provision.sh`:
```bash
sudo install -m 0640 -o $(whoami) -g $(whoami) /tmp/arem-worker.env /etc/arem-worker.env
mkdir -p ~/.config/rclone
install -m 0600 /tmp/rclone.conf ~/.config/rclone/rclone.conf
bash ~/arem-worker/provision.sh   # steps 1-5 are no-ops on re-run
```

## Ongoing maintenance: `update.sh`

When the repo gets new commits (heartbeat hardening, new model
checkpoints, etc.), pull and re-sync on each box:

```bash
bash ~/arem-worker/update.sh
```

What it does:
- `git pull` the repo
- Re-installs any `*.service` / `*.timer` / sudoers files that changed
  (uses `cmp -s` to detect actual changes; touches nothing if HEAD == HEAD)
- Re-installs the TUI autostart entry if it changed
- `pip install -r requirements.txt` inside the conda env
- `systemctl restart arem-worker-local` so the new code is picked up

If you've also bumped model checkpoints in R2, follow with
`bash ~/arem-worker/scripts/sync_checkpoints.sh`.

## Verifying the deploy worked

Within 60s of `provision.sh` completing, the new box should show up on
`https://arem-editing-dashboard.vercel.app/workers` and
`/fleet`. Look for two new entries:
- `host-<hostname-short>` (heartbeat row)
- `local-3090-<hostname-short>` (worker row; status `alive` until it
  claims a job)

The local TUI auto-launches at GNOME login. To see it now without
relogging:
```bash
gnome-terminal --maximize --hide-menubar -- bash -c \
  'watch -n 2 -t -c python3 ~/arem-worker/arem-fleet-tui.py'
```

## Known recovery scenarios

- **Kernel updated, NVIDIA module not loaded after reboot**
  `apt list --installed 2>/dev/null | grep linux-modules-nvidia` and confirm
  there's a package for the running kernel (`uname -r`). If not, re-run
  `provision.sh` — step 3 installs the matching version.

- **`/tmp/vercel-prod.env` lost on reboot, worker stuck activating**
  Already handled: the `*.service` files use
  `EnvironmentFile=-/etc/arem-worker.env`, which survives reboot.
  `provision.sh` installs this file at step 6.

- **Box can't reach the dashboard**
  Check the heartbeat journal: `journalctl -u arem-host-heartbeat -n 30`.
  TUI's `DASHBOARD` indicator turns amber when unreachable; worker still
  runs but won't claim new jobs (it polls `/api/jobs/claim`).

- **Want to remove a worker from the fleet**
  ```bash
  sudo systemctl disable --now arem-worker-local arem-host-heartbeat.timer
  ```
  The dashboard's `/workers` view will mark it `stale` within a few
  minutes, `dead` within 15.
