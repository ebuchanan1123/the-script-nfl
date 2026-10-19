# The Script Frontend

React, TypeScript, and Vite dashboard for exploring NFL player forecasts, weekly/ROS rankings,
player trends, and model evaluation metrics.

## Local development

From the repository root, install the Python dependencies and frontend packages, then run the API and
Vite development server in separate terminals:

```bash
uv sync --extra data --extra baseline --extra neural --extra api --extra dev
npm --prefix frontend ci
uv run the-script-api
npm --prefix frontend run dev -- --host 127.0.0.1
```

The frontend expects FastAPI at `http://127.0.0.1:8000/api` by default. Set `VITE_API_URL` to
override the API base URL.

## Static forecast mode

Static mode reads a versioned JSON release instead of calling FastAPI for player search, forecasts,
rankings, injuries, and metrics. Enable it at build time with:

```bash
VITE_DATA_MODE=static npm run build
```

The data manifest is read from `/data/manifest.json` by default. Override its base directory using
`VITE_DATA_URL`. Generate releases from the repository root with
`uv run python -m the_script.static_export`. Releases contain model inputs' derived forecasts and
metadata, not source Parquet files or model binaries. The manifest is replaced only after a complete
release is generated, so a failed export does not expose a partial release.

Do not publish a `--limit` smoke-test release as a production season: those releases are marked
`preview_only` and contain only a subset of players. The exporter also refuses unverified stale
sources unless `--allow-stale` is explicitly passed for local preview use.

## Build checks

```bash
npm --prefix frontend run build
```