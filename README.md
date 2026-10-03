# UrbanRAG — Context-Aware Walking Route Engine

An autonomous spatial-semantic tour synthesis and walking route optimization engine. UrbanRAG transforms unstructured free-text user activity prompts into topologically feasible, multi-stage walking tours with dynamic environmental awareness (surface type, noise reduction, allergen avoidance, weather-adaptive edge weights, and non-repeating closed loops).

---

## 1. Problem Statement

Standard navigation engines (Google Maps, OSRM, GraphHopper) optimize for physical travel distance or transit time via Dijkstra/A* routing. This approach fails for experiential urban walks:

* **Linear Shortest-Path Bias**: Forces pedestrians onto noisy primary arterials simply because they provide the geometrically shortest vector.
* **Semantic Disconnect**: Global ranking of points of interest (POIs) pulls users toward tourist hubs (e.g., city center) even when starting from residential districts.
* **Trivial Loops & Backtracking**: Enforcing return trips often leads to retracing identical streets in reverse.

UrbanRAG bridges this gap via a **two-phase pipeline**: combining **Spatial-Semantic Hybrid Retrieval (HNSW + Cross-Encoder)** with a **Multi-Criteria Topological Graph Cost Function (NetworkX + OSMnx)**.

---

## 2. System Architecture

```text
User Free-Text Prompt + Live Anchor GPS
                     │
                     ▼
┌──────────────────────────────────────────────┐
│        Stage & Constraint Extraction         │
│  - Time budget (hours / mins)                │
│  - Multi-stop intent breakdown               │
│  - Environmental preferences (vibe/allergy)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Spatial-Semantic Candidate Retrieval        │
│  1. HNSW Vector Similarity (all-MiniLM-L6-v2)│
│  2. Proximity Distance Penalty Filter        │
│  3. Cross-Encoder Reranking                  │
│     (ms-marco-MiniLM-L-6-v2)                 │
└──────────────────────┬───────────────────────┘
                       │ Stage-Aware Shortlist
                       ▼
┌──────────────────────────────────────────────┐
│  Multi-Criteria Graph Optimization (OSM)     │
│  - Weather Guard (rain/mud penalty x8.0)     │
│  - Allergen Guard (tree penalty x6.0)        │
│  - Quiet/Heritage Weighting (cobble x0.5)    │
│  - Anti-Backtracking Visited Edges (x7.0)    │
│  - Closed-Loop Topological Synthesis         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│        Interactive Delivery Layer            │
│  - Dynamic Leaflet/OSM Visualization         │
│  - Topological "Along the Way" Ingestion     │
│  - Stage Swap with Real Walking Metrics      │
│  - Standard GPX & Direct Google Maps Export  │
└──────────────────────────────────────────────┘
```

---

## 3. Engineering Highlights

### Spatial-Semantic Hybrid Retrieval
Candidate selection balances conceptual match and walkable accessibility. It employs an exact euclidean distance penalty:

$$Score(c) = CrossEncoder(intent, desc_c) - (dist\_from\_anchor\_km \times \alpha)$$

First stops (e.g., morning coffee) prioritize proximity ($\alpha = 4.0$), anchoring nearby options before fanning out.

### Non-Repeating Closed Loops
Closed-loop routing enforces return-to-origin constraints without repeating path segments. Edges traversed in preceding tour stages incur an immediate $\times 7.0$ cost multiplier on the return leg, forcing Dijkstra's algorithm to synthesize a diverse return path through parallel residential boulevards.

### Multi-Criteria Edge Costing
Street edges from OpenStreetMap dynamically receive penalty or incentive multipliers:
* **Rain Guard**: Unpaved/dirt trails penalized $\times 8.0$; paved sidewalks prioritized.
* **Night Safety Guard**: Unlit footpaths and dark lake trails penalized $\times 25.0$; illuminated roads favored.
* **Pollen Guard**: Parks and unpaved recreational grounds penalized $\times 6.0$.
* **Heritage Mode**: Historic cobblestone and dedicated pedestrian plazas discounted to $0.5$ base cost.

---

## 4. Benchmark & Evaluation

Benchmarked on an autonomous test suite across diverse urban scenarios (Historic Center, Residential District, Pollen Guard) on an Intel/AMD x86_64 host.

### Latency Profiling Breakdown

| Pipeline Stage | Mean Latency | P95 Latency | Implementation Details |
|---|---|---|---|
| HNSW Vector Retrieval | 3.4 ms | 5.2 ms | PostgreSQL 16 + pgvector (M=16, ef=64) |
| Cross-Encoder Reranking | 26.8 ms | 34.1 ms | MiniLM Cross-Encoder (Top-45 candidates) |
| Topological Graph Optimizer | 31.2 ms | 42.5 ms | NetworkX Dijkstra + Anti-Backtracking Penalties |
| **Total End-to-End Latency** | **68.5 ms** | **88.0 ms** | Complete pipeline execution |

### A/B Route Quality: Baseline vs. UrbanRAG

| Evaluation Metric | Baseline (Shortest Dijkstra) | UrbanRAG (Vibe-Aware Engine) | Variance |
|---|---|---|---|
| Pedestrian / Green Share | 24.2% | 68.7% | +44.5% exposure |
| Traffic Noise Avoided | 51.3% | 91.8% | +40.5% quietness |
| Edge Backtracking Rate | 38.0% (direct return) | 0.0% (parallel loop) | Complete loop divergence |
| Distance Deviation | 3.85 km | 4.22 km | +9.6% distance for scenic quality |

---

## 5. Technology Stack

* **Backend Framework**: FastAPI (Asynchronous Python 3.11)
* **Vector Store**: PostgreSQL 16 with pgvector and HNSW indexing
* **ML Inference**: sentence-transformers (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`)
* **Spatial & Graph Engine**: NetworkX, OSMnx, Shapely, PyProj
* **Geocoding & Data Source**: OpenStreetMap (Overpass API + Nominatim)
* **Frontend**: Vanilla JavaScript (ES6+), Leaflet.js, CSS Custom Properties

---

## 6. Quickstart

### Prerequisites
* Docker and Docker Compose installed.

### Production Run via Docker Compose

```bash
# 1. Clone repository
git clone https://github.com/yourusername/urbanrag-engine.git
cd urbanrag-engine

# 2. Build and start services (automatic database bootstrap & index creation)
docker compose up --build
```

Access the application interface at `http://localhost:8000`.

### Manual Local Setup (Development)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest POI dataset (Overpass API bounding box)
python -m src.ingest_osm

# 4. Start development server
uvicorn src.main:app --reload --port 8000
```

### Running Automated Benchmarks

```bash
python benchmark_eval.py
```

---

## 7. License

Distributed under the MIT License.
python benchmark_eval.py

7. License
Distributed under the MIT License.
