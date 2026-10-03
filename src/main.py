from contextlib import asynccontextmanager
from functools import lru_cache
import os
import re
from typing import List, Optional, Tuple
import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from pgvector.psycopg2 import register_vector
import psycopg2
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer
from src.map_builder import generate_tour_map_html
from src.nlp_parser import NaturalLanguageTourParser
from src.router import UrbanRouter, haversine_distance
from src.schemas import (
    AssistantAction,
    CandidatePOI,
    FreeTextTourRequest,
    OnTheWayPOI,
    TourPlanRequest,
    TourPlanResponse,
    TourStagePreference,
    TourStop,
    VibeAuditMetrics,
)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "geovector"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgrespassword"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

services = {}
nl_parser = NaturalLanguageTourParser()


@lru_cache(maxsize=512)
def get_cached_embedding(intent_text: str) -> np.ndarray:
    bi_model = services.get("bi_encoder")
    if bi_model is None:
        raise RuntimeError("Bi-encoder service is not initialized")
    return bi_model.encode(intent_text, normalize_embeddings=True)


def evaluate_opening_status(
    opening_hours_raw: Optional[str],
    arrival_hour: float,
    dwell_time_min: int,
    category: str = "place",
) -> Tuple[str, str]:
    """Evaluates whether a POI is open, closing soon, or closed at arrival_hour.

    Applies realistic urban fallbacks for commercial spots during night hours.
    """
    arrival_mod = arrival_hour % 24.0
    commercial_categories = {
        "cafe",
        "restaurant",
        "fast_food",
        "bar",
        "bakery",
        "shop",
        "pastry",
        "library",
        "museum",
    }
    always_open_categories = {
        "park",
        "garden",
        "square",
        "pedestrian",
        "viewpoint",
        "water",
        "beach",
        "monument",
        "artwork",
        "attraction",
    }

    if category in always_open_categories:
        return ("open", "Open 24/7 (Public Area)")

    # 1. Якщо це комерційний заклад без явного тегу 24/7 уночі (22:00 - 08:00) — зачинений
    if category in commercial_categories:
        if not opening_hours_raw or opening_hours_raw != "24/7":
            if arrival_mod >= 22.0 or arrival_mod < 8.0:
                return (
                    "closed",
                    "Closed at night (Typical hours: 08:00 - 22:00)",
                )

    if not opening_hours_raw or opening_hours_raw == "24/7":
        return (
            "open",
            "Open 24/7" if opening_hours_raw == "24/7" else "Hours not listed",
        )

    match = re.search(
        r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", opening_hours_raw
    )
    if not match:
        return ("unknown", opening_hours_raw)

    open_h = int(match.group(1)) + int(match.group(2)) / 60.0
    close_h = int(match.group(3)) + int(match.group(4)) / 60.0

    is_overnight = close_h < open_h

    if is_overnight:
        is_open = (arrival_mod >= open_h) or (arrival_mod < close_h)
    else:
        is_open = open_h <= arrival_mod < close_h

    if not is_open:
        return (
            "closed",
            f"Closed (Operating Hours: {match.group(1)}:{match.group(2)} - {match.group(3)}:{match.group(4)})",
        )

    if is_overnight:
        time_left_min = (
            ((close_h + 24.0) - arrival_mod) * 60.0
            if arrival_mod >= open_h
            else (close_h - arrival_mod) * 60.0
        )
    else:
        time_left_min = (close_h - arrival_mod) * 60.0

    if time_left_min <= (dwell_time_min + 15):
        return (
            "closing_soon",
            f"Closing soon at {match.group(3)}:{match.group(4)} (~{int(time_left_min)}m remaining)",
        )

    return ("open", f"Open until {match.group(3)}:{match.group(4)}")


def generate_gpx_track(
    stops: List[TourStop],
    path_coords: List[List[float]],
    track_name: str = "UrbanRAG Poltava Tour",
) -> str:
    wpt_xml = ""
    for s in stops:
        wpt_xml += f"""  <wpt lat="{s.lat:.6f}" lon="{s.lon:.6f}">
    <name>{s.name}</name>
    <desc>{s.intent} (Stay: {s.dwell_time_min}m) - {s.open_status_note or ''}</desc>
    <sym>Flag, Blue</sym>
  </wpt>\n"""

    trkpt_xml = ""
    for pt in path_coords:
        trkpt_xml += f'      <trkpt lat="{pt[0]:.6f}" lon="{pt[1]:.6f}"></trkpt>\n'

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="UrbanRAG Context Router" xmlns="http://www.topografix.com/GPX/1/1">
  <metadata>
    <name>{track_name}</name>
  </metadata>
{wpt_xml}  <trk>
    <name>{track_name}</name>
    <trkseg>
{trkpt_xml}    </trkseg>
  </trk>
</gpx>"""


def generate_google_maps_url(
    start_lat: float,
    start_lon: float,
    stops: List[TourStop],
    finish_mode: str = "return_to_start",
    finish_coord: Optional[tuple] = None,
) -> str:
    if not stops:
        return f"https://www.google.com/maps/@{start_lat},{start_lon},15z"

    origin = f"{start_lat:.5f},{start_lon:.5f}"

    if finish_mode == "return_to_start":
        destination = origin
        waypoint_coords = [f"{s.lat:.5f},{s.lon:.5f}" for s in stops]
    elif finish_mode == "custom" and finish_coord:
        destination = f"{finish_coord[0]:.5f},{finish_coord[1]:.5f}"
        waypoint_coords = [f"{s.lat:.5f},{s.lon:.5f}" for s in stops]
    else:
        destination = f"{stops[-1].lat:.5f},{stops[-1].lon:.5f}"
        waypoint_coords = [f"{s.lat:.5f},{s.lon:.5f}" for s in stops[:-1]]

    url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}&travelmode=walking"

    if waypoint_coords:
        waypoints_str = urllib.parse.quote("|".join(waypoint_coords))
        url += f"&waypoints={waypoints_str}"

    return url


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing UrbanRAG Engine: loading embedding and ranking models...")
    services["bi_encoder"] = SentenceTransformer("all-MiniLM-L6-v2")
    services["cross_encoder"] = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    services["router"] = UrbanRouter()
    print("Models and topological router ready.")
    yield
    services.clear()


app = FastAPI(
    title="UrbanRAG — Context-Aware Tour Planner",
    description="Multi-Criteria Environmental Walk Planner with Vector Spatial Retrieval",
    version="5.2.0",
    lifespan=lifespan,
)


@app.get("/", response_class=FileResponse)
def serve_home():
    return FileResponse("src/templates/index.html")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "urbanrag-engine"}


@app.get("/api/v1/geocode")
def geocode_location(query: str):
    if not query or len(query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query too short")

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{query}, Poltava", "format": "json", "limit": 1}
    headers = {"User-Agent": "UrbanRAG-Autonomous-City-Engine/2.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200 and resp.json():
            first = resp.json()[0]
            return {
                "display_name": first.get("display_name"),
                "lat": float(first.get("lat")),
                "lon": float(first.get("lon")),
            }
        else:
            params["q"] = query
            resp2 = requests.get(url, params=params, headers=headers, timeout=5)
            if resp2.status_code == 200 and resp2.json():
                first = resp2.json()[0]
                return {
                    "display_name": first.get("display_name"),
                    "lat": float(first.get("lat")),
                    "lon": float(first.get("lon")),
                }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Geocoding service error: {e}"
        )

    raise HTTPException(status_code=404, detail="Location not found")


@app.post("/api/v1/plan-tour", response_model=TourPlanResponse)
def plan_tour(payload: TourPlanRequest):
    bi_model = services.get("bi_encoder")
    cross_model = services.get("cross_encoder")
    router: UrbanRouter = services.get("router")

    if not bi_model or not cross_model or not router:
        raise HTTPException(status_code=500, detail="Services not initialized")

    max_minutes = payload.total_budget_hours * 60.0
    log_lines = []
    log_lines.append("=== 1. CONSTRAINTS & ROUTE PROFILE ===")
    log_lines.append(f"Starting GPS: ({payload.lat:.4f}, {payload.lon:.4f})")
    log_lines.append(f"Finish Mode: {payload.finish_mode.upper()}")
    log_lines.append(
        f"Time Budget: {payload.total_budget_hours}h ({max_minutes:.0f}m)"
    )
    log_lines.append(f"Vibe / Environment: '{payload.route_vibe_preference}'")
    log_lines.append(
        f"Weather: {payload.weather_condition.upper()} | Start Hour:"
        f" {payload.start_time_hour:02d}:00"
    )
    if payload.excluded_place_names:
        log_lines.append(f"Excluded Places: {payload.excluded_place_names}")
    if payload.force_trim_dwell:
        log_lines.append("[ACTION] Fast Trim Dwell Time active.")
    if payload.expand_loop:
        log_lines.append("[ACTION] Expand Loop active.")

    stages = list(payload.stages)

    if payload.expand_loop and len(stages) < 4:
        stages.insert(
            len(stages) - 1,
            TourStagePreference(
                intent=(
                    "scenic architectural pedestrian promenade or river valley"
                    " panorama"
                ),
                expected_stay_min=15,
            ),
        )
        log_lines.append("  [+] Added scenic loop transit waypoint.")

    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    cur.execute("""
        SELECT name, category, lat, lon, description, opening_hours
        FROM places
        WHERE category IN ('attraction', 'museum', 'monument', 'viewpoint', 'square');
    """)
    on_the_way_candidates = [
        {
            "name": r[0],
            "category": r[1],
            "lat": r[2],
            "lon": r[3],
            "description": r[4],
            "opening_hours": r[5],
        }
        for r in cur.fetchall()
    ]

    stage_candidates = []
    dwell_times = []
    stage_alternatives = []

    vibe_lower = payload.route_vibe_preference.lower()
    avoid_trees = any(
        w in vibe_lower
        for w in [
            "allergy",
            "pollen",
            "avoid trees",
            "no parks",
            "avoid parks",
        ]
    )

    max_walk_radius_km = max(1.8, (payload.total_budget_hours * 4.5) / 2.0)
    log_lines.append(
        f"Spatial Search Filter: Max search radius ~{max_walk_radius_km:.1f} km"
        " from anchor\n"
    )

    log_lines.append("=== 2. SPATIAL-SEMANTIC RETRIEVAL & RERANKING ===")

    used_tour_places = set(payload.excluded_place_names or [])
    for s in stages:
        if s.forced_place_name:
            used_tour_places.add(s.forced_place_name)

    prev_anchor_lat = payload.lat
    prev_anchor_lon = payload.lon

    for idx, stage in enumerate(stages, 1):
        dwell = stage.expected_stay_min
        if payload.force_trim_dwell:
            dwell = max(10, int(dwell * 0.65))

        dwell_times.append(dwell)
        log_lines.append(
            f"--- Stage {idx}: '{stage.intent}' (Stay: {dwell}m) ---"
        )

        if (
            stage.forced_place_name
            and stage.forced_place_name not in payload.excluded_place_names
        ):
            cur.execute(
                """
                SELECT name, category, lat, lon, description, opening_hours
                FROM places WHERE name = %s LIMIT 1;
            """,
                (stage.forced_place_name,),
            )
            f_row = cur.fetchone()
            if f_row:
                dist_prev = (
                    haversine_distance(
                        prev_anchor_lat, prev_anchor_lon, f_row[2], f_row[3]
                    )
                    / 1000.0
                )
                forced_cand = [{
                    "name": f_row[0],
                    "category": f_row[1],
                    "lat": f_row[2],
                    "lon": f_row[3],
                    "description": f_row[4],
                    "opening_hours": f_row[5],
                    "intent": stage.intent,
                    "dwell_time": dwell,
                    "score": 10.0,
                    "dist_from_prev_km": round(dist_prev, 2),
                }]
                stage_candidates.append(forced_cand)
                stage_alternatives.append([])
                prev_anchor_lat, prev_anchor_lon = f_row[2], f_row[3]
                used_tour_places.add(f_row[0])
                log_lines.append(
                    f"  [LOCKED STOP] {stage.forced_place_name}"
                    f" ({dist_prev:.2f}km from prev)\n"
                )
                continue

        q_vec = get_cached_embedding(stage.intent)

        cur.execute(
            """
            SELECT name, category, lat, lon, description, opening_hours,
                   (embedding <=> %s) AS cosine_dist
            FROM places
            ORDER BY cosine_dist ASC
            LIMIT 50;
        """,
            (q_vec,),
        )
        rows = cur.fetchall()

        raw_candidates = []
        seen_local_names = set()
        is_first_stage = idx == 1

        for r in rows:
            p_name = r[0]
            if p_name in used_tour_places or p_name in seen_local_names:
                continue

            p_lat, p_lon = r[2], r[3]
            dist_prev_km = (
                haversine_distance(
                    prev_anchor_lat, prev_anchor_lon, p_lat, p_lon
                )
                / 1000.0
            )

            intent_lower = stage.intent.lower()
            if is_first_stage and any(
                w in intent_lower
                for w in [
                    "coffee",
                    "bakery",
                    "snack",
                    "pastry",
                    "breakfast",
                    "cafe",
                ]
            ):
                if dist_prev_km > 1.3:
                    continue

            if dist_prev_km > max_walk_radius_km + 1.2:
                continue

            seen_local_names.add(p_name)
            raw_candidates.append({
                "name": p_name,
                "category": r[1],
                "lat": p_lat,
                "lon": p_lon,
                "description": r[4],
                "opening_hours": r[5],
                "intent": stage.intent,
                "dwell_time": dwell,
                "dist_from_prev_km": round(dist_prev_km, 2),
            })

        if not raw_candidates:
            for r in rows[:15]:
                p_name = r[0]
                if p_name in used_tour_places or p_name in seen_local_names:
                    continue
                dist_prev_km = (
                    haversine_distance(
                        prev_anchor_lat, prev_anchor_lon, r[2], r[3]
                    )
                    / 1000.0
                )
                seen_local_names.add(p_name)
                raw_candidates.append({
                    "name": r[0],
                    "category": r[1],
                    "lat": r[2],
                    "lon": r[3],
                    "description": r[4],
                    "opening_hours": r[5],
                    "intent": stage.intent,
                    "dwell_time": dwell,
                    "dist_from_prev_km": round(dist_prev_km, 2),
                })

        pairs = [[stage.intent, c["description"]] for c in raw_candidates]
        scores = cross_model.predict(pairs)
        for i, s in enumerate(scores):
            raw_score = float(s)
            c = raw_candidates[i]
            dist_penalty = 4.0 if is_first_stage else 1.8
            raw_score -= c["dist_from_prev_km"] * dist_penalty

            if avoid_trees and (
                c["category"] in ["park", "pitch"]
                or any(
                    w in c["name"].lower()
                    for w in [
                        "square",
                        "park",
                        "garden",
                        "сад",
                        "сквер",
                        "парк",
                    ]
                )
            ):
                raw_score -= 20.0
            c["score"] = raw_score

        raw_candidates.sort(key=lambda x: x["score"], reverse=True)

        for rank, c in enumerate(raw_candidates[:4], 1):
            walk_min = int(round((c["dist_from_prev_km"] * 1000.0) / 75.0))
            log_lines.append(
                f"  [{rank}] {c['name']} ({c['category']}) ->"
                f" {c['dist_from_prev_km']}km (~{walk_min}m walk) | Score:"
                f" {c['score']:.2f}"
            )

        top_candidates = raw_candidates[:3]
        stage_candidates.append(top_candidates)

        if top_candidates:
            prev_anchor_lat = top_candidates[0]["lat"]
            prev_anchor_lon = top_candidates[0]["lon"]
            used_tour_places.add(top_candidates[0]["name"])

        stage_alternatives.append([
            CandidatePOI(
                name=c["name"],
                category=c["category"],
                score=round(c["score"], 2),
                lat=c["lat"],
                lon=c["lon"],
                distance_from_prev_km=c["dist_from_prev_km"],
            )
            for c in raw_candidates[:7]
        ])
        log_lines.append(
            "  Shortlisted for optimizer:"
            f" {[c['name'] for c in top_candidates]}\n"
        )

    cur.close()
    conn.close()

    log_lines.append("=== 3. TOPOLOGICAL ROUTING & MULTI-CRITERIA COST ===")
    finish_coord = (
        (payload.finish_lat, payload.finish_lon)
        if (payload.finish_lat and payload.finish_lon)
        else None
    )

    tour = router.optimize_multistage_tour(
        payload.lat,
        payload.lon,
        stage_candidates,
        dwell_times,
        max_minutes,
        finish_mode=payload.finish_mode,
        finish_coord=finish_coord,
        routing_mode=payload.routing_mode,
        vibe_preference=payload.route_vibe_preference,
        target_steps=payload.target_steps,
        weather=payload.weather_condition,
        hour=payload.start_time_hour,
    )

    if not tour:
        raise HTTPException(
            status_code=400, detail="Could not find a feasible route"
        )

    audit = tour["vibe_audit"]
    applied_rules = tour.get("applied_rules", [])

    log_lines.append("Active Cost Modifiers:")
    for rule in applied_rules:
        log_lines.append(f"  * {rule}")

    log_lines.append("\nExposure Metrics:")
    log_lines.append(
        f"  * Pedestrian/Paved Trails: {audit['pedestrian_share_pct']}%"
    )
    log_lines.append(
        f"  * Traffic Noise Avoided: {audit['traffic_avoided_pct']}% of route"
    )
    log_lines.append(f"  * Historical Cobblestone: {audit['cobblestone_pct']}%")

    steps_count = tour["itinerary"]["estimated_steps"]
    dist_km = tour["itinerary"]["total_distance_km"]

    excluded_names = {s["name"] for s in tour["stops"]}
    eligible_on_the_way = [
        p for p in on_the_way_candidates if p["name"] not in excluded_names
    ]
    stops_coords = [(s["lat"], s["lon"]) for s in tour["stops"]]

    on_the_way_list = router.find_pois_on_the_way(
        tour["itinerary"]["path_coordinates"],
        eligible_on_the_way,
        stops_coords=stops_coords,
        max_dist_m=65.0,
    )

    if on_the_way_list:
        log_lines.append("\n=== 4. OPPORTUNISTIC DISCOVERY (WALK-BY) ===")
        for otw in on_the_way_list:
            log_lines.append(
                f"   [{otw['name']}] ({otw['category']}) - ~{otw['distance_to_path_m']}m"
                f" away (after stop #{otw['insert_after_order']})"
            )

    # Динамічна оцінка робочих годин
    arrival_time_clock = float(payload.start_time_hour)
    closed_stops_count = 0
    stops_response = []

    for idx, s in enumerate(tour["stops"]):
        leg_walk_min = (
            tour["leg_durations_min"][idx]
            if "leg_durations_min" in tour
            and idx < len(tour["leg_durations_min"])
            else 10.0
        )
        arrival_time_clock = (arrival_time_clock + (leg_walk_min / 60.0)) % 24.0

        status, note = evaluate_opening_status(
            s.get("opening_hours"),
            arrival_time_clock,
            s["dwell_time"],
            category=s.get("category", "place"),
        )

        if status == "closed":
            closed_stops_count += 1

        stops_response.append(
            TourStop(
                order=idx + 1,
                intent=s["intent"],
                name=s["name"],
                category=s["category"],
                description=s["description"],
                dwell_time_min=s["dwell_time"],
                lat=s["lat"],
                lon=s["lon"],
                opening_hours_raw=s.get("opening_hours"),
                open_status=status,
                open_status_note=note,
                alternative_candidates=(
                    stage_alternatives[idx]
                    if idx < len(stage_alternatives)
                    else []
                ),
            )
        )
        arrival_time_clock = (
            arrival_time_clock + (s["dwell_time"] / 60.0)
        ) % 24.0

    assistant_recs = []
    available_actions = []

    for s_item in stops_response:
        if s_item.open_status == "closed":
            assistant_recs.append(
                f"Notice: <b>{s_item.name}</b> appears to be <b>closed</b> at"
                f" estimated arrival (~{int(arrival_time_clock):02d}:00)."
                " Consider swapping it for an open location."
            )
        elif s_item.open_status == "closing_soon":
            assistant_recs.append(
                f"Heads up: <b>{s_item.name}</b> is closing soon"
                f" ({s_item.open_status_note})."
            )

    if closed_stops_count > 0 and closed_stops_count == len(stops_response):
        assistant_recs.insert(
            0,
            "Schedule Warning: All selected commercial venues appear closed at"
            " this hour. The engine prioritized open public spaces.",
        )

    if tour["total_time"] > max_minutes:
        diff = round(tour["total_time"] - max_minutes, 1)
        if diff <= 16.0:
            assistant_recs.append(
                f"Near-Miss Budget: Exceeds limit by {diff} mins"
                f" ({tour['total_time']:.0f}m / {max_minutes:.0f}m). Click Fast"
                " Trim to fit strictly."
            )
            available_actions.append(
                AssistantAction(
                    action_type="trim_budget",
                    label="Fast Trim to Budget",
                    description="Reduces stay durations to fit strictly.",
                    payload={"force_trim_dwell": True},
                )
            )
        else:
            assistant_recs.append(
                "Distance & Time Conflict: Tour requires"
                f" {tour['walking_time']:.0f} mins of walking ({dist_km} km)."
                " Consider removing a stop."
            )
            available_actions.append(
                AssistantAction(
                    action_type="trim_budget",
                    label="Fast Trim (Reduce Stays)",
                    description=(
                        "Reduces stay durations to bring tour closer to limit."
                    ),
                    payload={"force_trim_dwell": True},
                )
            )
    else:
        leftover = round(max_minutes - tour["total_time"], 1)
        if leftover >= 30.0 and dist_km < 3.0:
            assistant_recs.append(
                f"Pacing Opportunity: You have {leftover:.0f} mins of spare"
                f" time ({steps_count:,} steps). Extend into a loop to reach"
                " 5,000+ steps."
            )
            available_actions.append(
                AssistantAction(
                    action_type="expand_loop",
                    label="Expand Loop (+3,000 steps)",
                    description=(
                        "Adds a scenic loop through historic avenues and"
                        " viewpoints."
                    ),
                    payload={"expand_loop": True},
                )
            )

    if payload.finish_mode == "return_to_start":
        assistant_recs.append(
            "Closed Loop: Route returns to start origin without re-traversing"
            " identical streets."
        )
    if avoid_trees:
        assistant_recs.append(
            "Pollen Guard: Parks and unpaved trails avoided. Open paved streets"
            " prioritized."
        )
    elif "historic" in vibe_lower or "architecture" in vibe_lower:
        assistant_recs.append(
            "Heritage Focus: Cobblestone streets and historic avenues"
            " prioritized."
        )

    if payload.weather_condition.lower() == "rainy":
        assistant_recs.append(
            "Rain Guard: Unpaved paths penalized. Paved pedestrian sidewalks"
            " prioritized."
        )
    if payload.start_time_hour >= 20 or payload.start_time_hour <= 6:
        assistant_recs.append(
            "Night Safety Guard: Unlit park paths and dark waterfronts avoided."
            " Well-lit avenues prioritized."
        )

    log_lines.append(
        f"\nFinal Plan: {' -> '.join([s['name'] for s in tour['stops']])}"
    )
    if payload.finish_mode == "return_to_start":
        log_lines.append("  |--> Returning to Start Anchor")
    log_lines.append(
        f"Walk: {tour['walking_time']}m ({dist_km} km, ~{steps_count:,} steps)"
        f" | Stops: {tour['dwell_time']}m"
    )
    log_lines.append(
        f"Total: {tour['total_time']:.1f}m / {max_minutes:.0f}m (Status:"
        f" {'WITHIN BUDGET' if tour['is_within_budget'] else 'TIME EXCEEDED'})"
    )

    full_log_text = "\n".join(log_lines)

    summary_text = (
        f"Walk: {tour['walking_time']}m ({dist_km}km, ~{steps_count:,} steps) |"
        f" Stops: {tour['dwell_time']}m | Total:"
        f" {round(tour['total_time'], 1)}m / {int(max_minutes)}m"
    )

    gpx_content = generate_gpx_track(
        stops_response, tour["itinerary"]["path_coordinates"]
    )
    gmaps_url = generate_google_maps_url(
        payload.lat,
        payload.lon,
        stops_response,
        finish_mode=payload.finish_mode,
        finish_coord=finish_coord,
    )

    return TourPlanResponse(
        status="success",
        total_time_min=round(tour["total_time"], 1),
        walking_time_min=round(tour["walking_time"], 1),
        stops_time_min=tour["dwell_time"],
        time_budget_min=max_minutes,
        is_within_budget=tour["is_within_budget"],
        total_distance_km=dist_km,
        estimated_steps=steps_count,
        stops=stops_response,
        path_coordinates=tour["itinerary"]["path_coordinates"],
        summary=summary_text,
        vibe_audit=VibeAuditMetrics(**audit),
        on_the_way=[OnTheWayPOI(**p) for p in on_the_way_list],
        assistant_recommendations=assistant_recs,
        available_actions=available_actions,
        reasoning_log=full_log_text,
        gpx_xml=gpx_content,
        google_maps_url=gmaps_url,
    )


@app.post("/api/v1/plan-tour/prompt")
def plan_tour_from_natural_language(payload: FreeTextTourRequest):
    if payload.explicit_stages and len(payload.explicit_stages) > 0:
        total_hours, _ = nl_parser.parse(payload.user_prompt)
        stages_objs = list(payload.explicit_stages)
    else:
        total_hours, stages_data = nl_parser.parse(payload.user_prompt)
        stages_objs = []
        for idx, s in enumerate(stages_data):
            forced = payload.forced_replacements.get(str(idx))
            stages_objs.append(
                TourStagePreference(
                    intent=s["intent"],
                    expected_stay_min=s["expected_stay_min"],
                    forced_place_name=forced,
                )
            )

    tour_request = TourPlanRequest(
        lat=payload.lat,
        lon=payload.lon,
        finish_mode=payload.finish_mode,
        finish_lat=payload.finish_lat,
        finish_lon=payload.finish_lon,
        total_budget_hours=total_hours,
        routing_mode=payload.routing_mode,
        route_vibe_preference=payload.route_vibe_preference,
        target_steps=payload.target_steps,
        weather_condition=payload.weather_condition,
        start_time_hour=payload.start_time_hour,
        expand_loop=payload.expand_loop,
        force_trim_dwell=payload.force_trim_dwell,
        excluded_place_names=payload.excluded_place_names,
        stages=stages_objs,
    )

    tour_response = plan_tour(tour_request)

    stops_data = [s.model_dump() for s in tour_response.stops]
    card_summary = (
        f"<b>Preference:</b> {payload.route_vibe_preference}<br>"
        f"{tour_response.summary}"
    )

    html_content = generate_tour_map_html(
        start_point=[payload.lat, payload.lon],
        stops=stops_data,
        path_coords=tour_response.path_coordinates,
        summary_text=card_summary,
    )

    return JSONResponse(
        content={
            "map_html": html_content,
            "reasoning_log": tour_response.reasoning_log,
            "summary": tour_response.summary,
            "stops": [s.model_dump() for s in tour_response.stops],
            "recommendations": tour_response.assistant_recommendations,
            "available_actions": [
                a.model_dump() for a in tour_response.available_actions
            ],
            "on_the_way": [p.model_dump() for p in tour_response.on_the_way],
            "vibe_audit": (
                tour_response.vibe_audit.model_dump()
                if tour_response.vibe_audit
                else None
            ),
            "total_distance_km": tour_response.total_distance_km,
            "estimated_steps": tour_response.estimated_steps,
            "total_time_min": tour_response.total_time_min,
            "time_budget_min": tour_response.time_budget_min,
            "is_within_budget": tour_response.is_within_budget,
            "gpx_xml": tour_response.gpx_xml,
            "google_maps_url": tour_response.google_maps_url,
        }
    )