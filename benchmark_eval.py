#!/usr/bin/env python3
import time
import statistics
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.router import UrbanRouter, haversine_distance

DB_CONFIG = {
    "dbname": "geovector",
    "user": "postgres",
    "password": "postgrespassword",
    "host": "localhost",
    "port": 5432
}

TEST_SUITE = [
    {
        "name": "Historic Center Walk",
        "start": (49.5891, 34.5513),
        "stages": [
            {"intent": "grab a specialty filter coffee and pastry", "stay": 20},
            {"intent": "scenic historic monument and architectural sight", "stay": 25},
            {"intent": "elevated panoramic viewpoint overlooking the valley", "stay": 20}
        ],
        "vibe": "historic architecture, cobblestone pedestrian streets, avoid cars",
        "budget_min": 120.0
    },
    {
        "name": "Residential District Loop (Almaznyi)",
        "start": (49.5650, 34.5200),
        "stages": [
            {"intent": "artisan bakery or local coffee spot", "stay": 15},
            {"intent": "quiet public green park and walking alley", "stay": 30},
            {"intent": "neighborhood recreation square or sports pitch", "stay": 20}
        ],
        "vibe": "quiet residential alleys, no traffic noise, green trees",
        "budget_min": 100.0
    },
    {
        "name": "Pollen Allergy Strict Walk",
        "start": (49.5800, 34.5400),
        "stages": [
            {"intent": "urban coffee shop with dessert", "stay": 20},
            {"intent": "wide paved city plaza and central promenade", "stay": 30}
        ],
        "vibe": "avoid trees and parks due to severe pollen allergy, stay on open paved streets",
        "budget_min": 90.0
    }
]

def run_benchmarks():
    print("=" * 80)
    print("UrbanRAG Automated Benchmarking & Quality Evaluation Suite")
    print("=" * 80)

    print("\n[1/3] Initializing models and topological graph...")
    t0 = time.perf_counter()
    bi_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    router = UrbanRouter(dist_meters=7500)
    init_time = time.perf_counter() - t0
    print(f"      Engine ready in {init_time:.2f}s")

    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    knn_latencies = []
    cross_latencies = []
    routing_latencies = []
    total_latencies = []

    ab_metrics = {
        "baseline_ped_pct": [],
        "urbanrag_ped_pct": [],
        "baseline_noise_avoided_pct": [],
        "urbanrag_noise_avoided_pct": [],
        "baseline_dist_km": [],
        "urbanrag_dist_km": []
    }

    print("\n[2/3] Executing multi-criteria evaluation queries...")

    for test_idx, tc in enumerate(TEST_SUITE, 1):
        print(f"\n---> Scenario {test_idx}: '{tc['name']}'")
        lat_start, lon_start = tc["start"]
        max_walk_radius_km = max(1.8, (tc["budget_min"] / 60.0 * 4.5) / 2.0)

        t_pipeline_start = time.perf_counter()
        stage_candidates = []
        dwell_times = [s["stay"] for s in tc["stages"]]

        prev_lat, prev_lon = lat_start, lon_start

        for stage_idx, stage in enumerate(tc["stages"]):
            # 1. HNSW KNN Lookup
            t_knn_0 = time.perf_counter()
            q_vec = bi_model.encode(stage["intent"], normalize_embeddings=True)
            cur.execute("""
                SELECT name, category, lat, lon, description,
                       (embedding <=> %s) AS dist
                FROM places
                ORDER BY dist ASC
                LIMIT 45;
            """, (q_vec,))
            rows = cur.fetchall()
            knn_latencies.append((time.perf_counter() - t_knn_0) * 1000.0)

            # Spatial filtering
            candidates = []
            for r in rows:
                p_lat, p_lon = r[2], r[3]
                dist_km = haversine_distance(prev_lat, prev_lon, p_lat, p_lon) / 1000.0
                if dist_km <= max_walk_radius_km + 1.2:
                    candidates.append({
                        "name": r[0], "category": r[1], "lat": p_lat, "lon": p_lon,
                        "description": r[4], "intent": stage["intent"],
                        "dwell_time": stage["stay"], "dist_km": dist_km
                    })

            if not candidates:
                for r in rows[:15]:
                    dist_km = haversine_distance(prev_lat, prev_lon, r[2], r[3]) / 1000.0
                    candidates.append({
                        "name": r[0], "category": r[1], "lat": r[2], "lon": r[3],
                        "description": r[4], "intent": stage["intent"],
                        "dwell_time": stage["stay"], "dist_km": dist_km
                    })

            # 2. Cross-Encoder Reranking
            t_cross_0 = time.perf_counter()
            pairs = [[stage["intent"], c["description"]] for c in candidates]
            scores = cross_model.predict(pairs)
            for i, s in enumerate(scores):
                candidates[i]["score"] = float(s) - (candidates[i]["dist_km"] * 1.8)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            cross_latencies.append((time.perf_counter() - t_cross_0) * 1000.0)

            top_picks = candidates[:3]
            stage_candidates.append(top_picks)
            if top_picks:
                prev_lat, prev_lon = top_picks[0]["lat"], top_picks[0]["lon"]

        # 3. Route Optimization
        t_route_0 = time.perf_counter()
        
        # A: UrbanRAG Multicriteria Tour
        urbanrag_tour = router.optimize_multistage_tour(
            lat_start, lon_start, stage_candidates, dwell_times, tc["budget_min"],
            finish_mode="return_to_start", routing_mode="custom",
            vibe_preference=tc["vibe"]
        )

        # B: Baseline Shortest Path (Classic Dijkstra purely by length)
        baseline_tour = router.optimize_multistage_tour(
            lat_start, lon_start, stage_candidates, dwell_times, tc["budget_min"],
            finish_mode="return_to_start", routing_mode="fastest"
        )
        
        routing_latencies.append((time.perf_counter() - t_route_0) * 1000.0)
        total_latencies.append((time.perf_counter() - t_pipeline_start) * 1000.0)

        # Audit comparison
        u_audit = urbanrag_tour["vibe_audit"]
        b_audit = baseline_tour["vibe_audit"]

        ab_metrics["baseline_ped_pct"].append(b_audit["pedestrian_share_pct"])
        ab_metrics["urbanrag_ped_pct"].append(u_audit["pedestrian_share_pct"])
        ab_metrics["baseline_noise_avoided_pct"].append(b_audit["traffic_avoided_pct"])
        ab_metrics["urbanrag_noise_avoided_pct"].append(u_audit["traffic_avoided_pct"])
        ab_metrics["baseline_dist_km"].append(baseline_tour["itinerary"]["total_distance_km"])
        ab_metrics["urbanrag_dist_km"].append(urbanrag_tour["itinerary"]["total_distance_km"])

        print(f"      Selected Route: {' -> '.join([s['name'] for s in urbanrag_tour['stops']])}")
        print(f"      Pedestrian share:  Baseline = {b_audit['pedestrian_share_pct']}%  vs  UrbanRAG = {u_audit['pedestrian_share_pct']}%")
        print(f"      Traffic avoided:   Baseline = {b_audit['traffic_avoided_pct']}%  vs  UrbanRAG = {u_audit['traffic_avoided_pct']}%")

    cur.close()
    conn.close()

    print("\n[3/3] Generating Consolidated Summary Tables...")
    print("=" * 80)
    print("TABLE 1: LATENCY PROFILING BREAKDOWN (milliseconds)")
    print("=" * 80)
    print(f"HNSW Vector Retrieval (pgvector) : mean = {statistics.mean(knn_latencies):.2f} ms | p95 = {statistics.quantiles(knn_latencies, n=20)[18]:.2f} ms")
    print(f"Cross-Encoder Reranker           : mean = {statistics.mean(cross_latencies):.2f} ms | p95 = {statistics.quantiles(cross_latencies, n=20)[18]:.2f} ms")
    print(f"Topological Graph Optimizer      : mean = {statistics.mean(routing_latencies):.2f} ms | p95 = {statistics.quantiles(routing_latencies, n=20)[18]:.2f} ms")
    print(f"Total End-to-End Latency         : mean = {statistics.mean(total_latencies):.2f} ms | p95 = {statistics.quantiles(total_latencies, n=20)[18]:.2f} ms")

    print("\n" + "=" * 80)
    print("TABLE 2: A/B EVALUATION (Baseline Dijkstra vs UrbanRAG Engine)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Baseline (Shortest)':<20} | {'UrbanRAG (Vibe-Aware)':<20}")
    print("-" * 80)
    print(f"{'Pedestrian / Green Share':<35} | {statistics.mean(ab_metrics['baseline_ped_pct']):.1f}%{'':<15} | {statistics.mean(ab_metrics['urbanrag_ped_pct']):.1f}%")
    print(f"{'Traffic Noise Avoided':<35} | {statistics.mean(ab_metrics['baseline_noise_avoided_pct']):.1f}%{'':<15} | {statistics.mean(ab_metrics['urbanrag_noise_avoided_pct']):.1f}%")
    print(f"{'Average Distance':<35} | {statistics.mean(ab_metrics['baseline_dist_km']):.2f} km{'':<13} | {statistics.mean(ab_metrics['urbanrag_dist_km']):.2f} km")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmarks()