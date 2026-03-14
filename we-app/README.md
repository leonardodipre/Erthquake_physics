# L'Aquila Fault Explorer

This folder contains the full scientific web app for interactive exploration of the L'Aquila fault system and GNSS deformation data.

## What is inside

- `scripts/export_web_data.py`: preprocesses the repository assets into browser-friendly JSON/GeoJSON under `public/data`
- `public/data`: generated scientific assets used directly by the frontend
- `src`: React + Vite + TypeScript application with:
  - a 2D overview page for the fault trace and GNSS stations
  - a 3D fault-model page for sub-fault patches and exported model fields

## Data notes

- The fault trace is exported from `dataset_scremato/faglia_aquila.geojson`
- Station time series are exported from the repository CSV files in `dataset_scremato/modified`, `dataset_scremato/accepted`, and `dataset_scremato/acc_test`
- Real station coordinates are loaded from `dataset_scremato/stations_ITA_laquila_150km.json`
- The 3D patch mesh is reconstructed from `green_out/fault_mesh.npz`
- Model snapshots are exported for each discovered checkpoint under `checkpoints/*/*.pt` when a matching evaluation CSV exists
- If a station is missing from the local metadata JSON, the exporter falls back to a deterministic mock coordinate so the schema stays stable

## Run locally

1. Install frontend dependencies:

```bash
npm install
```

2. Rebuild the exported web assets if the scientific inputs change:

```bash
npm run export:data
```

3. Start the Vite development server:

```bash
npm run dev
```

4. Open the local URL shown by Vite, usually `http://localhost:5173`.

## Build

```bash
npm run build
```

The production build is written to `dist/`.

## Exported asset contract

- `public/data/fault.geojson`
- `public/data/fault_patches.json`
- `public/data/station_metadata.json`
- `public/data/stations.json`
- `public/data/timeseries/{station_id}.json`
- `public/data/model_snapshots/index.json`
- `public/data/model_snapshots/{date}.json`
- `public/data/metrics/evaluation.json`
- `public/data/metrics/training_history.json`
- `public/data/metrics/station_sensitivity.json`

The frontend lazy-loads station series and snapshot files to keep the initial page responsive.

## Model selection

The 3D page reads `public/data/models/index.json` and lets you switch between the exported checkpoint families. Each model has its own snapshot index under `public/data/model_snapshots/<model_key>/index.json`.
