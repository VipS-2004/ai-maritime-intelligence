# Satellite Maritime Intelligence System

Streamlit application that takes a satellite image, runs a custom-trained YOLOv8 detector (25 vessel classes), then produces maritime analysis, spatial overlays, a risk assessment, and an optional Gemini intelligence write-up.

## Live demo

https://satellite-maritime-intelligence-hxctcpaxdl4ovhwmqnaf5e.streamlit.app/

## Overview

Upload a JPEG or PNG satellite image. The app loads `best.pt`, runs detection, classifies vessels as military / civilian / unknown using `configs/ship_classes.py`, then builds congestion, density, clustering, and risk outputs. Spatial intelligence draws a density heatmap and a 4x4 zone overlay. Gemini is optional and only runs when you request an AI assessment (and when `GEMINI_API_KEY` is set).

The same analysis modules are also used by `main.py` for command-line runs.

## Features

- Detect and classify 25 vessel types with YOLOv8
- Per-class counts plus military / civilian / unknown totals
- Congestion (Low / Medium / High) and density (Sparse / Moderate / Dense) from ship count
- Pairwise clustering check (centers closer than 50 px)
- Risk from the military-to-total ratio (no military = Low; under 30% = Medium; 30% or more = High)
- Alert string from military presence and congestion
- Gaussian density heatmap and 4x4 zone hotspot overlay
- Streamlit report UI (perception, composition, spatial, risk, optional AI)
- Optional Gemini assessment (`gemini-3.5-flash`) from the structured analysis JSON

Not implemented: AIS, lat/long georeferencing, live satellite feeds, or multi-frame tracking.

## Model performance

Validation metrics for the custom YOLOv8 weights (`best.pt`):

| Metric     | Value |
| ---------- | ----- |
| Precision  | 0.687 |
| Recall     | 0.666 |
| mAP@50     | 0.676 |
| mAP@50-95  | 0.566 |

## Intelligence pipeline

```text
Satellite image
  -> YOLOv8 detection (best.pt)
  -> vessel classification (25 classes)
  -> maritime analysis (counts, congestion, density, clustering, risk, alert)
  -> spatial intelligence (heatmap + 4x4 zones)
  -> risk presentation
  -> optional Gemini analyst
  -> intelligence report
```

## Architecture

The application is organized into separate modules for detection, analysis, visualization, and optional AI interpretation.

```text
                         Satellite Image
                               |
                               v
                       +---------------+
                       |    app.py     |
                       |  Streamlit UI |
                       +-------+-------+
                               |
                               v
                       +---------------+
                       |   YOLOv8      |
                       |   Detector    |
                       +-------+-------+
                               |
                               v
                       Vessel Detections
                               |
                +--------------+--------------+
                |                             |
                v                             v
        Maritime Analysis             Spatial Analysis
        analysis/                     visualization/
                |                             |
                |                    Heatmap + Zones
                |                             |
                +--------------+--------------+
                               |
                               v
                       Risk Assessment
                               |
                               v
                    Optional Gemini Analyst
                               |
                               v
                    Intelligence Report
```

## Project structure

```text
app.py                      Streamlit UI
main.py                     CLI pipeline
best.pt                     trained YOLOv8 weights (Git LFS)
requirements.txt
assets/
  detection.png             example detection overlay
  Heatmap.png               example density heatmap
  Zone-map.png              example 4x4 zone overlay
ai/agent.py
analysis/maritime_analysis.py
configs/ship_classes.py
models/detector.py
visualization/visualizer.py
```

## Installation

Python 3.10+ recommended. `best.pt` is stored with Git LFS; pull LFS objects before running.

```bash
git lfs install
git clone <repository-url>
cd "Ship Detection"
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Confirm `best.pt` is in the project root.

## Run the Streamlit app

From the project root:

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501). Upload a satellite image, then run analysis. The AI section is separate and can be skipped.

## Optional Gemini setup

Used only for the AI assessment in `ai/agent.py` (model `gemini-3.5-flash`).

1. Create a Gemini API key.
2. Set `GEMINI_API_KEY` in the environment (or as a Streamlit Cloud secret with the same name).

Without the key, detection, maritime analysis, spatial overlays, and risk still work. The app will report that the AI assessment could not be generated. Quota errors are handled without retrying exhausted keys.

## Vessel classes

Defined in `configs/ship_classes.py` (25 classes).

Military: Destroyer, Cruiser, Frigate, Submarine, Warship, Commander, Aircraft Carrier.

Civilian (in this project’s grouping): Cargo, Tanker, Fishing, Passenger, Yacht, Barge, Container Ship, Bulk Carrier, Oil Tanker, Tug, Auxiliary Ships, Patrol, Landing, Hovercraft, Sailboat, Carrier, Boat, Other.

A detection whose class name is in neither list is counted as unknown.

## Example outputs

Example overlays from `assets/`:

Detection:

![Detection overlay](assets/detection.png)

Density heatmap:

![Density heatmap](assets/Heatmap.png)

Zone overlay:

![Zone overlay](assets/Zone-map.png)

## Command-line usage

`main.py` runs the same detector, analysis, heatmap, and zone grid, then prints results. It also tries Gemini once (optional in practice: it fails cleanly without a key). Matplotlib shows the YOLO plot window.

```bash
python main.py --image path/to/satellite.jpg --weights best.pt --grid 4
```

| Argument   | Default   | Description                          |
| ---------- | --------- | ------------------------------------ |
| `--image`  | required  | Input satellite image                |
| `--weights`| `best.pt` | YOLO weights                         |
| `--grid`   | `4`       | Zone grid size (Streamlit uses 4)    |

## Dependencies

From `requirements.txt`:

- ultralytics >= 8.0.0
- opencv-python-headless == 4.10.0.84
- numpy >= 1.23.0
- matplotlib >= 3.3.0
- scipy >= 1.4.1
- PyYAML >= 5.3.1
- google-genai >= 1.0.0
- streamlit >= 1.30.0
- pandas >= 1.5.0

## Future work

Possible follow-ups, none of which are in the current app:

- Georeferenced footprints (lat/long) when image metadata exists
- AIS or other identity sources
- Stronger calibration of congestion / clustering thresholds
- Export of the report (PDF or JSON)

## License

This repository does not include a license file.
