<p align="center">
  <img width="460" alt="SitePulse AI logo" src="https://github.com/user-attachments/assets/24bdf8c0-27bf-49de-8d4e-5f976de11c6e" />
</p>

<h1 align="center">SitePulse AI</h1>

<p align="center">
  Operational intelligence for heavy equipment.
</p>

<p align="center">
  Motion-aware tracking, articulated equipment analysis, dwell-time monitoring, and live operational visibility.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Ultralytics-YOLO-111111?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Apache%20Kafka-Streaming-231F20?style=for-the-badge&logo=apachekafka&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Deployment-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>

---

## Product Overview

**SitePulse AI** is a real-time equipment monitoring system designed for construction operations. It processes site video, detects and tracks equipment, classifies machine state and activity, preserves machine identity across tracker changes, measures utilization, and streams structured telemetry to downstream consumers.

The system is built as a practical microservices-style prototype for:
- **equipment utilization monitoring**
- **activity classification**
- **idle / dwell time measurement**
- **streaming operational telemetry through Kafka**
- **live operator-facing visualization in Streamlit**

The current implementation focuses on **excavators** and **dump trucks**, with special treatment for articulated motion so an excavator can still be classified as **ACTIVE** when the tracks are stationary but the arm and bucket are working.

---

## Why This Matters

On a real construction site, the most expensive operational blind spot is not just whether a machine exists in frame. It is whether the machine is actually working, what part of its cycle it is in, and how long it has been idle.

SitePulse AI turns raw footage into operational signals:
- **ACTIVE vs INACTIVE**
- **DIGGING / SWINGING_LOADING / DUMPING / WAITING**
- **working time**
- **idle time**
- **dwell time**
- **utilization percentage**
- **persistent equipment identity**

---
## Preview

### End-to-End Demo

<p align="center">
    <img src="https://github.com/user-attachments/assets/9d9599e2-b81a-492d-bf9a-80c2a1aa41ba" alt="SitePulse AI end-to-end demo" width="100%">
  </a>
</p>


### Streamlit Application

<p align="center">
    <img src="https://github.com/user-attachments/assets/62daced1-1fb3-4a2c-b55f-20d72d9f10f5" alt="SitePulse AI Streamlit app preview" width="100%">
  </a>
</p>



---


## Core Capabilities

| Capability | What SitePulse AI Does |
|---|---|
| Equipment Detection | Detects excavators and dump trucks from video frames |
| Persistent Identity | Uses Re-ID logic above tracker IDs so dwell time does not reset when tracker IDs change |
| Utilization Tracking | Classifies each machine as ACTIVE or INACTIVE |
| Articulated Motion Handling | Detects ACTIVE excavator behavior even when only the arm/bucket is moving |
| Activity Classification | Classifies DIGGING, SWINGING_LOADING, DUMPING, WAITING |
| Dwell Analytics | Tracks current dwell and maximum dwell per asset |
| Telemetry Streaming | Emits structured events to Kafka and JSONL |
| Live UI | Displays processed video, machine cards, live status, and utilization metrics |

---

## System Architecture

```text
Input Video
    │
    ▼
CV Microservice
(sitepulse_ai.cli / sitepulse_ai.pipeline)
    │
    ├── Machine detection
    ├── Excavator arm/bucket segmentation
    ├── Optical-flow motion analysis
    ├── Re-ID and asset continuity
    ├── State + activity fusion
    ├── Utilization + dwell analytics
    │
    ├── Processed video
    ├── JSONL event stream
    ├── Summary JSON
    └── Kafka topic
              │
              ▼
       Streamlit UI
       (apps/streamlit_app.py)
```

### Services
- **Backend CV microservice**: runs the perception, fusion, utilization, and event generation pipeline
- **Redpanda**: Kafka-compatible broker for streaming structured machine events
- **Streamlit UI**: visualizes processed output and live machine analytics

---
## End-to-End Processing Flow

1. **Read video frame**
2. **Detect machines** using the equipment detector
3. **Track detections** and reconcile identities through the Re-ID layer
4. **Crop excavator detections** and run the arm/bucket segmentation model
5. **Compute optical flow** over time
6. **Measure regional motion**
   - excavator body
   - excavator arm
   - excavator bucket
   - dump truck body
   - dump truck bed proxy
7. **Classify ACTIVE vs INACTIVE**
8. **Run activity FSM**
9. **Update dwell and utilization**
10. **Render overlays**
11. **Emit structured events**
12. **Write processed video, JSONL, and summary outputs**
13. **Display live results in Streamlit**

---

## Technical Write-up

## 1) Machine Detection

The first model detects the site equipment classes used by the prototype:
- **excavator**
- **dump truck**

This stage gives the system the top-level machine regions that drive tracking, Re-ID, motion analysis, and telemetry.

### Why a dedicated detector is needed
The detector provides:
- the machine bounding boxes used for motion regions
- the base tracker IDs
- the machine class labels used for routing logic
- the appearance crops used by Re-ID

Without a strong detector, everything downstream becomes unstable: tracking, identity continuity, activity classification, and dwell time all degrade.

---

## 2) Articulated Equipment Challenge

This is the most important technical problem in the prototype.

A standard motion-based utilization system would fail on excavators because:
- the **tracks can remain stationary**
- the **chassis can remain almost stationary**
- only the **arm and bucket** may move

If the system only looked at whole-box translation, it would incorrectly classify a working excavator as **INACTIVE**.

### How SitePulse AI solves it
The system handles articulated motion with **region-based motion analysis** plus **instance segmentation** of the moving excavator parts.

For each excavator:
- the machine detector produces the excavator bounding box
- a second model segments **ARM** and **BUCKET**
- optical flow is measured separately for:
  - body region
  - arm region
  - bucket region

This means the excavator can be classified as **ACTIVE** when:
- the arm magnitude is high
- the bucket magnitude is high
- part-active ratio is elevated

even if the body region is mostly static.

### Why this is the right design choice
This design is stronger than pure whole-box motion because it:
- isolates articulated motion
- reduces false idle classification
- aligns with real excavator mechanics
- supports activity classification downstream

In practice, this is what allows the pipeline to detect:
- digging while the chassis is still
- swing phases between dig zone and truck
- dumping near the truck side

---

## 3) Motion Analysis

The pipeline computes dense optical flow between consecutive frames and summarizes motion in semantically meaningful regions.

### Excavator regions
- **body**
- **arm**
- **bucket**

### Dump truck regions
- **body**
- **upper bed region proxy**

For each region, the system derives:
- average magnitude
- x/y directional flow
- active motion ratio
- center speed

These signals are fused into machine state and activity decisions.

---

## 4) ACTIVE vs INACTIVE Utilization Logic

The system does not rely on a single threshold spike.

Instead, it uses:
- activation thresholds
- stricter idle thresholds
- consecutive-frame counters
- hysteresis behavior

This prevents:
- flickering state changes
- false ACTIVE from small motion noise
- false WAITING during brief pauses inside a work cycle

### Excavator ACTIVE logic
An excavator can become **ACTIVE** if any of the following are sustained:
- bucket motion
- arm motion
- part-active ratio
- body motion
- whole-box center movement

### Dump truck ACTIVE logic
A dump truck becomes **ACTIVE** from:
- meaningful center translation
- meaningful body motion
- elevated dump-bed-region motion

---

## 5) Activity Classification

Activity classification is not done with a single hard-coded frame rule.  
It is handled with a **phase-aware finite state machine**.

### Excavator activities
- **WAITING**
- **DIGGING**
- **SWINGING_LOADING**
- **DUMPING**

### Why an FSM was used
Raw threshold logic is too fragile for cyclic machine behavior. Excavators follow a repeated cycle:
- digging
- swinging/loading
- dumping
- swinging back
- digging again

The FSM enforces legal transitions and blocks jumps such as:
- WAITING → DUMPING
- DIGGING → DUMPING without the swing phase

### Evidence accumulation
Each activity label must earn enough evidence over consecutive frames before it is committed.  
This prevents one noisy frame from flipping the label.

### Hysteresis exit
Once a label is active, it is not dropped immediately on a minor score dip.  
This prevents flicker during short pauses or partial occlusion.

---

## 6) How Activities Are Distinguished

### DIGGING
DIGGING is driven primarily by excavator arm motion consistent with a downward digging stroke, combined with bucket position in the lower dig zone.

Signals used:
- arm moving downward
- vertical motion dominance
- bucket low in the excavator box
- anti-swing suppression when horizontal motion dominates

### SWINGING_LOADING
SWINGING_LOADING captures the lateral transport phase of the excavator cycle.

Signals used:
- horizontal arm motion
- horizontal bucket motion
- anti-dig suppression when the bucket is still clearly in the dig zone

### DUMPING
DUMPING is not just “bucket high.”  
It is modeled as:
- release-phase excavator motion
- bucket in the upper zone
- bucket aligned with the nearest dump truck
- spatial agreement with truck center / truck core zone

### WAITING
WAITING is intentionally strict.  
The excavator is only considered waiting when:
- arm motion is quiet
- bucket motion is quiet
- body motion is quiet
- center motion is quiet

---

## 7) Re-ID and Dwell Time Preservation


### The problem
Tracker IDs are not stable enough to be treated as long-term machine identity.  
If the tracker changes ID after brief occlusion, crossing, or re-entry:
- the machine would look like a new machine
- dwell time would reset
- idle analytics would become unreliable

### The solution
SitePulse AI introduces a persistent **asset identity layer** above tracker IDs.

For each new detection, the system tries to reconcile it with a recently missing asset using:
- appearance histogram similarity
- center-distance similarity
- bounding-box IoU similarity
- class consistency
- maximum missing-frame window

If the match is good enough, the old asset is revived and its history is preserved.

---

## 8) Streaming and Data Contract

The backend emits structured telemetry for each machine as:
- **JSONL events**
- **Kafka messages**

Each payload contains:
- timestamp
- frame index
- asset ID
- tracker ID
- equipment type
- current state
- current activity
- tracked / active / idle time
- dwell time
- utilization percentage
- motion features

---
## Quick Start

## 1) Local Python Run

### Backend
```bash
python -m sitepulse_ai.cli   --detector-model /absolute/path/to/detect_best.pt   --parts-model /absolute/path/to/instance_best.pt   --source /absolute/path/to/input.mp4   --output-video /absolute/path/to/outputs/processed.mp4   --events-jsonl /absolute/path/to/outputs/events.jsonl   --summary-json /absolute/path/to/outputs/summary.json   --device mps   --detector-imgsz 960   --parts-imgsz 640   --detector-conf 0.15   --detector-iou 0.50   --parts-conf 0.15   --reid-max-missing-frames 45   --reid-min-similarity 0.58   --reid-max-center-dist-ratio 0.18   --verbose
```

### Streamlit UI
```bash
streamlit run apps/streamlit_app.py
```

---

## 2) Docker Compose Run

```bash
cp .env.example .env
mkdir -p models data outputs
docker compose up --build
```
---
## Docker Services

### `redpanda`
Kafka-compatible broker used for local event streaming.

### `backend`
Runs the CV microservice and writes:
- processed video
- JSONL events
- summary JSON
- Kafka messages

### `ui`
Runs the Streamlit front end.

---

## Important Note

This project was developed in under two days due to the submission deadline.  
Within that limited timeframe, the focus was placed on delivering a working end-to-end proof of concept that demonstrates the viability of the full pipeline, including equipment detection, articulated-motion analysis, activity classification, utilization tracking, dwell-time monitoring, streaming output, and a live operational UI.

For the articulated-motion component, only around **150 images** were manually annotated for the excavator arm and bucket segmentation task. As a result, the current articulated-motion model should be viewed as an initial proof of concept rather than the final ceiling of performance.

The current results validate that the system design works and that the pipeline is structurally aligned with a production-oriented deployment model. With a larger and more diverse annotated dataset, longer training cycles, and broader scene coverage, the models would be expected to deliver significantly stronger accuracy, robustness, and generalization across different camera angles, lighting conditions, machine types, and site conditions.

In other words, this repository demonstrates that the proposed architecture and workflow are viable at a production-oriented level, while also making clear that model performance can be materially improved through dataset scale, annotation depth, and additional training time.

---
