import json
import os
import time
import psycopg2
from pgvector.psycopg2 import register_vector
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "geovector"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgrespassword"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

CACHE_FILE = "data/osm_places_backup.json"
POLTAVA_BBOX = "49.5200,34.4500,49.6500,34.6400"


def query_overpass_city(bbox: str = POLTAVA_BBOX):
    os.makedirs("data", exist_ok=True)
    print(f"[INFO] Collecting POIs across city districts using BBOX [{bbox}]...")

    query = f"""
    [out:json][timeout:45];
    (
      // Cafes, bakeries, and dining spots
      node["amenity"~"^(cafe|restaurant|fast_food|bar|ice_cream|library)$"]({bbox});
      way["amenity"~"^(cafe|restaurant|fast_food|bar|ice_cream|library)$"]({bbox});
      node["shop"~"^(bakery|coffee|pastry)$"]({bbox});
      way["shop"~"^(bakery|coffee|pastry)$"]({bbox});

      // Parks, recreation grounds, waterfronts, sports pitches
      node["leisure"~"^(park|garden|pitch|fitness_station)$"]({bbox});
      way["leisure"~"^(park|garden|pitch|fitness_station)$"]({bbox});
      node["natural"~"^(beach|water)$"]({bbox});
      way["natural"~"^(beach|water)$"]({bbox});
      way["landuse"="recreation_ground"]({bbox});

      // Tourism, viewpoints, cultural heritage
      node["tourism"~"^(viewpoint|museum|attraction)$"]({bbox});
      way["tourism"~"^(viewpoint|museum|attraction)$"]({bbox});

      // Public squares and pedestrian infrastructure
      node["place"="square"]({bbox});
      way["place"="square"]({bbox});
      way["highway"="pedestrian"]({bbox});
    );
    out center tags qt;
    """

    headers = {"User-Agent": "UrbanRAG-FullCity/5.0"}

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"[INFO] Connecting to {endpoint}...")
            resp = requests.post(
                endpoint, data={"data": query}, headers=headers, timeout=50
            )
            if resp.status_code == 200:
                data = resp.json()
                elements = data.get("elements", [])
                if elements:
                    print(
                        f"[INFO] Successfully retrieved {len(elements)} raw elements."
                    )
                    with open(CACHE_FILE, "w", encoding="utf-8") as f:
                        json.dump(elements, f, ensure_ascii=False)
                    return elements
            else:
                print(
                    f"[WARN] Endpoint returned status {resp.status_code}, trying next mirror..."
                )
        except Exception as e:
            print(f"[ERROR] Connection failure for {endpoint}: {e}")
            time.sleep(1)

    if os.path.exists(CACHE_FILE):
        print(
            "[WARN] All remote mirrors unreachable. Loading fallback data from local cache..."
        )
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    raise RuntimeError(
        "Failed to retrieve POI data from both Overpass API endpoints and local backup."
    )


def transform_and_enrich_poi(elements):
    print(f"[INFO] Processing and enriching {len(elements)} raw elements...")
    cleaned_places = []
    seen = set()

    for el in elements:
        tags = el.get("tags", {})

        category = (
            tags.get("amenity")
            or tags.get("shop")
            or tags.get("tourism")
            or tags.get("leisure")
            or tags.get("natural")
            or tags.get("place")
            or tags.get("highway")
            or "place"
        )

        name = tags.get("name:uk") or tags.get("name") or tags.get("name:en")

        if not name or len(name.strip()) < 2:
            if category in ["park", "garden", "recreation_ground"]:
                name = "District Green Zone / Public Park"
            elif category in ["water", "beach"]:
                name = "Local Waterbody / Lakeside"
            elif category == "pedestrian":
                name = "Pedestrian Boulevard / Walkway"
            elif category in ["fitness_station", "pitch"]:
                name = "Outdoor Workout / Sports Ground"
            else:
                continue

        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if not lat or not lon:
            continue

        coord_key = (round(lat, 4), round(lon, 4))
        if coord_key in seen:
            continue
        seen.add(coord_key)

        features = []
        if tags.get("surface"):
            features.append(f"surface: {tags.get('surface')}")
        if tags.get("lit") == "yes":
            features.append("night lighting available")
        if category in ["square", "pedestrian"]:
            features.append(
                "open stone plaza, car-free zone, zero tree pollen exposure"
            )
        if tags.get("outdoor_seating") == "yes":
            features.append("outdoor seating terrace")

        opening_hours = tags.get("opening_hours")
        if opening_hours:
            features.append(f"opening hours: {opening_hours}")

        extra_desc = ""
        if category in [
            "cafe",
            "restaurant",
            "fast_food",
            "bakery",
            "coffee",
            "pastry",
        ]:
            extra_desc = (
                "Coffee, bakery snacks, pastry, desserts, drinks, takeaway food."
            )
        elif category in ["park", "garden", "recreation_ground"]:
            extra_desc = (
                "Public urban green park, quiet nature trees, walking paths, benches."
            )
        elif category in ["viewpoint"]:
            extra_desc = "Scenic elevated viewpoint, wide panorama overlooking city and valley."
        elif category in ["beach", "water"]:
            extra_desc = (
                "Waterfront shoreline, river bank, lake, quiet water walk."
            )
        elif category in ["museum", "attraction"]:
            extra_desc = (
                "Cultural monument, architecture, local attraction, historical site."
            )
        elif category in ["square", "pedestrian"]:
            extra_desc = (
                "Paved pedestrian boulevard, open promenade, architectural facades."
            )

        feature_str = (", ".join(features) + ". ") if features else ""
        description = (
            f"{name} ({category}). {feature_str}{extra_desc} Poltava district."
        )

        cleaned_places.append(
            {
                "name": name,
                "category": category,
                "lat": lat,
                "lon": lon,
                "description": description,
                "opening_hours": opening_hours,
            }
        )

    print(
        f"[INFO] Extracted {len(cleaned_places)} unique valid POIs across all districts."
    )
    return cleaned_places


def load_into_postgres(places):
    print("[INFO] Writing to PostgreSQL and computing vector embeddings...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("DROP TABLE IF EXISTS places;")
    cur.execute(
        """
        CREATE TABLE places (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            category VARCHAR(100),
            lat DOUBLE PRECISION NOT NULL,
            lon DOUBLE PRECISION NOT NULL,
            description TEXT NOT NULL,
            opening_hours VARCHAR(255),
            embedding vector(384)
        );
    """
    )
    conn.commit()
    register_vector(conn)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    for p in tqdm(places, desc="Indexing POIs with sentence-transformers"):
        emb = model.encode(p["description"], normalize_embeddings=True)
        cur.execute(
            """
            INSERT INTO places (name, category, lat, lon, description, opening_hours, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """,
            (
                p["name"],
                p["category"],
                p["lat"],
                p["lon"],
                p["description"],
                p.get("opening_hours"),
                emb,
            ),
        )
    conn.commit()

    print("[INFO] Building HNSW index on vector embeddings...")
    cur.execute(
        """
        CREATE INDEX idx_places_hnsw_embedding 
        ON places USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """
    )
    conn.commit()

    cur.close()
    conn.close()
    print("[INFO] POI database successfully initialized and indexed.")


if __name__ == "__main__":
    elements = query_overpass_city()
    places = transform_and_enrich_poi(elements)
    load_into_postgres(places)