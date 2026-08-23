# Renderer-A — Channelry video render worker

Standalone render worker for **Channelry** faceless videos, running as
**disposable GitHub Actions compute** (ephemeral, dynamically-addressed
runners). It polls the Cloudflare queue through the existing `render-workers`
API, downloads the per-beat assets the job carries, chops character
backgrounds out with RemBG, composes each scene (background + positioned
characters), synthesizes the dialogue with edge-tts, renders with FFmpeg,
uploads `final.mp4` to R2, and reports completion — so the finished video
appears in the user's Runs.

```
GitHub runner ─ HTTPS→ Cloudflare API (heartbeat/claim/progress/complete)
            └── HTTPS→ R2 (download assets, upload final.mp4)
```

**Everything is outbound.** The runner is ephemeral and dynamically
addressed — it never depends on a stable/allowlisted IP, and Cloudflare never
calls it. This is what lets a freshly-provisioned GitHub-hosted runner pick
up a job and be destroyed right after.

> Renderer-B is reserved for future heavy/independent workloads (3D, dense AI)
> and is intentionally not touched yet. Renderer-A contains the complete
> video pipeline including character background removal.

---

## Queue contract (already implemented by the backend — we reuse it)

| Endpoint (backend) | Purpose |
|---|---|
| `POST /api/render-workers/heartbeat` | register before claim; refresh `last_seen_at` |
| `POST /api/render-workers/claim` | atomically claim one `queued` job (or none) |
| `GET  /api/render-workers/jobs/{id}/status` | cancellation / status check |
| `PATCH /api/render-workers/jobs/{id}` | `status` = `rendering`/`done`/`failed`, `progress`, `video_url` |

Auth is `Authorization: Bearer <CHANNELRY_RENDER_WORKER_TOKEN>` — the
existing server-to-server `WORKER_TOKEN` on the channelry-admin Worker (the
plan’s “RENDERER_TOKEN”). It grants queue/report only, never admin.

The renderer handles the `character_video` + `auto_cast` job kind (the flow
the new editor produces), plus the classic single-character `still` / `walk`
methods (`settings.method`, `settings.image_url`). Other job kinds are
reported `failed` with a clear “unsupported render method” reason rather than
hanging.

### Beat shape handled (matches the backend’s `body.beats` contract)

```jsonc
{
  "characters": [
    { "image_url": "…R2…", "x": 0.5, "y": 0.92, "scale": 0.6, "voice": "…", "dialogue": "…" }
  ],
  "background_url": "…R2…",
  "dialogue": "…",
  "voice": "en-GB-SoniaNeural"
}
```

The full character image is downloaded and background-removed **here at render
time** — the editor deliberately shows the untouched image (plan §6).

---

## Run locally (Docker)

```bash
cp .env.example .env   # fill in real values
docker build -t renderer-a .
# long-running worker:
docker run --env-file .env renderer-a
# bounded batch (one job) then exit:
docker run --env-file .env renderer-a --once
```

## Run locally without Docker (quick smoke test)

```bash
pip install -r requirements.txt
CHANNELRY_FORCE_NO_REMBG=1 python -c \
  "import os,sys;sys.path.insert(0,'.');import tests.smoke as s;s.main()"
```

`tests/smoke.py` proves the whole compose→voice→ffmpeg→concat pipeline against
local files (no Cloudflare/R2), so you can validate rendering before wiring the
runner to the real queue.

---

## Deploy targets

- **GitHub Actions** — `.github/workflows/render.yml` (ephemeral runner;
  heartbeat + claim + render, then finish so the VM is reaped).
- **VPS / GCP / AWS / dev** — same Docker image, `--max-jobs 0` (forever).

The Docker image is the same everywhere — you only change the compute
provider (plan §22). Background removal (`renderer/background_removal/`) is an
isolated component you can swap for another provider without touching the
editor or queue (plan §21).

---

## Secrets (names only — never commit values)

For GitHub Actions you need **7 secrets** (the token + 5 R2 values + the API URL);
the worker id/name are auto-generated per run and need not be set:

`CHANNELRY_CLOUD_API_URL`, `CHANNELRY_RENDER_WORKER_TOKEN`,
`CHANNELRY_R2_ENDPOINT`, `CHANNELRY_R2_BUCKET`,
`CHANNELRY_R2_ACCESS_KEY_ID`, `CHANNELRY_R2_SECRET_ACCESS_KEY`,
`CHANNELRY_R2_PUBLIC_BASE_URL`.

Optional/local: `CHANNELRY_RENDER_WORKER_ID`, `CHANNELRY_RENDER_WORKER_NAME`
(defaults to github-runner; a fresh value per job is expected on ephemeral
runners). Keep everything in GitHub Actions Secrets / Wrangler secrets /
Docker `-e` flags — never in this repo, the Dockerfile, logs, or a scene
config. `CHANNELRY_RENDER_WORKER_TOKEN` is the existing `WORKER_TOKEN` secret
on the channelry-admin Worker (the renderer's credential; server-to-server
only).

## Security notes

- No secrets in source; the render token is server-to-server only.
- The final video + assets are transient on the runner (ephemeral disk).
- Everything of record lives in Cloudflare / R2 (jobs, users, credits, videos).