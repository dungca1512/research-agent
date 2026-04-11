# Bug Report

## Status
RESOLVED

## Bug Title
Docker volume bind-mount fails: `/data` host directory does not exist on macOS

## Bug Description
`docker compose up -d` fails during container creation because the `research_data` volume is configured as a bind mount to `/data` on the host. This path was designed for the Oracle Cloud VM (where `/data` is a mounted block volume), but it does not exist on a local macOS machine.

## Steps to Reproduce
1. Clone repo on macOS
2. Run `docker compose up -d`

## Actual Result
```
Error response from daemon: failed to populate volume: error while mounting volume
'/var/lib/docker/volumes/research-agent_research_data/_data':
failed to mount local volume: mount /data:/var/lib/docker/volumes/...
flags: 0x1000: no such file or directory
```

## Expected Result
All 7 services start successfully.

## Context
- **Error Message**: `mount /data: no such file or directory`
- **Environment**: macOS, Docker Desktop
- **When introduced**: Observation #1187 — added bind-mount to `/data` for Oracle Cloud deployment

---

## Root Cause Analysis

`docker-compose.yml:150-156` defines `research_data` as a bind mount:

```
volumes:
  research_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data      ← host path must exist before Docker can mount it
```

On the Oracle Cloud VM, `/data` is a block volume mounted by `scripts/setup-server.sh`.
On macOS, `/data` is never created → Docker errors out immediately.

```
docker compose up
       │
       ▼
create volume research_data
       │
       ▼
bind-mount /data → /var/lib/docker/volumes/.../   ← FAILS: /data missing
```

## Proposed Fixes

### Fix Option 1 — Recommended: Use `./data` (project-relative bind mount)
Change `device: /data` → `device: ./data` and ensure the directory exists.
Docker Compose resolves relative paths from the compose file location, so `./data`
maps to `/Users/dungca/research-agent/data` locally and can be overridden on the
server via an `.env` file or a server-specific compose override.

**docker-compose.yml change:**
```yaml
volumes:
  research_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./data     # relative to project root
```

The `data/` directory must exist before compose runs. Add it to the repo
(with a `.gitkeep`) or create it with `mkdir -p data`.

### Fix Option 2 — Alternative: Use a plain named volume (no bind mount)
Remove `driver_opts` entirely. Docker manages the volume internally.
Data persists across restarts but is not accessible directly on the host filesystem.

```yaml
volumes:
  research_data:       # no driver_opts — Docker-managed named volume
```

Trade-off: simpler, but data lives inside Docker's storage area (`/var/lib/docker/volumes/`)
and is harder to inspect or back up directly.

### Fix Option 3 — Alternative: docker-compose.override.yml for server
Keep `device: /data` in docker-compose.yml for the Oracle Cloud server,
and create a local `docker-compose.override.yml` that overrides the volume
to use `./data`. Override files are gitignored and applied automatically.

Trade-off: adds a file per environment; easy to forget.

**Recommendation**: Fix Option 1 — minimal change, works on both local and server
(server already has `/data` but `./data` resolves the same way once the project
is cloned to `/home/ubuntu/research-agent`; alternatively the server path can be
set via `DATA_PATH` env var with an override file).

## Verification Plan
1. Apply fix
2. Run `docker compose up -d`
3. Confirm all 7 services show `Up` in `docker compose ps`
4. Check `docker volume inspect research-agent_research_data` shows correct mount path
