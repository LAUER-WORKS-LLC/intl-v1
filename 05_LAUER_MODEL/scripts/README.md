# LOESS Analysis Scripts

This folder contains the scripts for the LOESS-based stock analysis pipeline.

## Scripts

### 01_LOWESS_general.py (Level 1)
Performs year-by-year LOESS smoothing and identifies key points (inflection points and extrema).

**Input:**
- Ticker symbol (user input)
- OHLCV data from `data/<TICKER>/raw/` (or downloads if not present)

**Output:**
- Visualizations: `output/<TICKER>/level_1/`
- Processed data: `data/<TICKER>/processed/<TICKER>_key_points_analysis.csv`

**Usage:**
```bash
python scripts/01_LOWESS_general.py
```

### 02_LOW_INT_chunk.py (Level 2)
Creates chunks between key points, applies LOESS to each chunk, and performs spline interpolation.

**Input:**
- Ticker symbol (user input)
- OHLCV data from `data/<TICKER>/raw/`
- Key points from `data/<TICKER>/processed/<TICKER>_key_points_analysis.csv` (requires Level 1 to run first)

**Output:**
- Individual chunk plots: `output/<TICKER>/level_2/chunks/`
- Chunk feature JSON files: `output/<TICKER>/level_2/chunks/<TICKER>_kp_chunk_<NNN>_features.json`
- Combined visualization: `output/<TICKER>/level_2/<TICKER>_key_point_chunks_combined.png`

**Usage:**
```bash
python scripts/02_LOW_INT_chunk.py
```

## Folder Structure

```
05_LAUER_MODEL/
├── scripts/
│   ├── 01_LOWESS_general.py
│   ├── 02_LOW_INT_chunk.py
│   └── README.md
├── data/
│   └── <TICKER>/
│       ├── raw/
│       │   └── <ticker>_ohlcv_<start_year>_2025.csv
│       └── processed/
│           └── <TICKER>_key_points_analysis.csv
└── output/
    └── <TICKER>/
        ├── level_1/
        │   ├── <TICKER>_<YEAR>_loess_analysis.png
        │   └── <TICKER>_all_years_combined_loess.png
        └── level_2/
            ├── chunks/
            │   ├── <TICKER>_kp_chunk_<NNN>.png
            │   └── <TICKER>_kp_chunk_<NNN>_features.json
            └── <TICKER>_key_point_chunks_combined.png
```

## Workflow

1. Run `01_LOWESS_general.py` first to generate key points
2. Run `02_LOW_INT_chunk.py` to analyze chunks between key points and extract features

## Data Migration

If you have existing data files in the old structure (`data/*.csv`), you can migrate them by:

1. Creating the ticker-specific folder: `data/<TICKER>/raw/`
2. Moving the CSV file: `data/<TICKER>/raw/<ticker>_ohlcv_<start_year>_2025.csv`

Similarly, if you have existing output files, you can organize them into:
- `output/<TICKER>/level_1/` for Level 1 outputs
- `output/<TICKER>/level_2/` for Level 2 outputs

