import math
import os
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import osmnx as ox

GRAPH_PATH_TEMPLATE = "data/poltava_walk_{radius}.graphml"


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def distance_point_to_segment(
    p_lat: float,
    p_lon: float,
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
) -> float:
    avg_lat = math.radians((a_lat + b_lat) / 2.0)
    kx = 111320.0 * math.cos(avg_lat)
    ky = 110540.0

    px, py = p_lon * kx, p_lat * ky
    ax, ay = a_lon * kx, a_lat * ky
    bx, by = b_lon * kx, b_lat * ky

    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


class UrbanRouter:

    def __init__(
        self,
        center_point: Tuple[float, float] = (49.5891, 34.5513),
        dist_meters: int = 7500,
    ):
        self.center_point = center_point
        self.dist_meters = dist_meters
        self.graph_path = GRAPH_PATH_TEMPLATE.format(radius=dist_meters)
        self.graph = self._load_or_build_graph()

    def _load_or_build_graph(self) -> nx.MultiDiGraph:
        os.makedirs("data", exist_ok=True)
        if os.path.exists(self.graph_path):
            print(f"Loading cached pedestrian graph from {self.graph_path}...")
            g = ox.load_graphml(self.graph_path)
            if "crs" not in g.graph:
                g.graph["crs"] = "EPSG:4326"
            return g

        old_path = "data/poltava_walk.graphml"
        if self.dist_meters <= 3500 and os.path.exists(old_path):
            print(f"Loading cached pedestrian graph from {old_path}...")
            return ox.load_graphml(old_path)

        print(
            f"Downloading full-city walkable street network ({self.dist_meters}m radius)..."
        )
        try:
            try:
                ox.settings.overpass_url = (
                    "https://overpass-api.de/api/interpreter"
                )
            except AttributeError:
                ox.settings.overpass_endpoint = (
                    "https://overpass-api.de/api/interpreter"
                )
            ox.settings.timeout = 60

            graph = ox.graph_from_point(
                self.center_point, dist=self.dist_meters, network_type="walk"
            )
            graph.graph["crs"] = "EPSG:4326"
            ox.save_graphml(graph, self.graph_path)
            print("Graph successfully saved to cache.")
            return graph
        except Exception as e:
            print(
                f"OSM download fallback: generating synthetic grid ({e})..."
            )
            return self._generate_synthetic_grid()

    def _generate_synthetic_grid(self) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        G.graph["crs"] = "EPSG:4326"
        lat_center, lon_center = self.center_point
        step = 0.002
        nodes = {}
        idx = 1
        for i in range(-12, 13):
            for j in range(-12, 13):
                G.add_node(
                    idx, y=lat_center + i * step, x=lon_center + j * step
                )
                nodes[(i, j)] = idx
                idx += 1

        for (i, j), u in nodes.items():
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (i + di, j + dj)
                if neighbor in nodes:
                    v = nodes[neighbor]
                    dist_m = haversine_distance(
                        G.nodes[u]["y"],
                        G.nodes[u]["x"],
                        G.nodes[v]["y"],
                        G.nodes[v]["x"],
                    )
                    hw = (
                        "pedestrian"
                        if abs(i) <= 2 and abs(j) <= 2
                        else "residential"
                    )
                    G.add_edge(u, v, 0, length=dist_m, highway=hw)
                    G.add_edge(v, u, 0, length=dist_m, highway=hw)
        return G

    def compute_dynamic_weights(
        self, vibe_preference: str, weather: str = "clear", hour: int = 14
    ) -> Tuple[str, List[str]]:
        pref = vibe_preference.lower()
        avoid_trees = any(
            w in pref
            for w in [
                "allergy",
                "pollen",
                "avoid trees",
                "no parks",
                "avoid parks",
            ]
        )
        want_historic = any(
            w in pref
            for w in [
                "history",
                "historic",
                "architecture",
                "monument",
                "old town",
                "cobblestone",
            ]
        )
        avoid_traffic = any(
            w in pref
            for w in [
                "quiet",
                "traffic",
                "no cars",
                "peaceful",
                "silence",
                "avoid cars",
            ]
        )

        applied_rules = []
        weight_attr = "dynamic_cost"

        if avoid_traffic:
            applied_rules.append(
                "Car Traffic Penalty: Primary/Secondary streets penalised x4.5"
            )

        if avoid_trees:
            applied_rules.append(
                "Pollen Allergy Guard: Park alleys & dirt trails penalised x6.0"
            )
        else:
            applied_rules.append(
                "Green Bonus: Footpaths & quiet park trails discounted x0.7"
            )

        if want_historic:
            applied_rules.append(
                "Historic Heritage: Cobblestone & pedestrian boulevards discounted x0.5"
            )

        is_rainy = weather.lower() in ["rain", "rainy", "wet", "muddy"]
        if is_rainy:
            applied_rules.append(
                "Rain & Mud Guard: Dirt/unpaved paths penalised x8.0; paved sidewalks prioritized"
            )

        # Нічний період
        is_dark = hour >= 20 or hour <= 6
        if is_dark:
            applied_rules.append(
                "Night Safety Guard: Unlit paths & dark lake trails penalised x25.0; illuminated roads prioritized"
            )

        for u, v, k, data in self.graph.edges(keys=True, data=True):
            length = float(data.get("length", 80.0))
            hw = data.get("highway", "residential")
            if isinstance(hw, list):
                hw = hw[0]

            surface = data.get("surface", "")
            if isinstance(surface, list):
                surface = surface[0]

            lit = data.get("lit", "")
            if isinstance(lit, list):
                lit = lit[0]

            multiplier = 1.0

            if hw in ["primary", "secondary", "tertiary"]:
                multiplier *= 4.5 if avoid_traffic else 1.5

            if hw in ["pedestrian", "living_street"]:
                multiplier *= 0.6

            if hw in ["footway", "path", "track"]:
                if avoid_trees:
                    multiplier *= 6.0
                elif is_rainy:
                    multiplier *= 5.0
                else:
                    multiplier *= 0.7

            if is_rainy and surface in [
                "ground",
                "dirt",
                "grass",
                "sand",
                "unpaved",
            ]:
                multiplier *= 8.0

            if want_historic and (
                surface in ["cobblestone", "paving_stones", "sett"]
                or hw == "pedestrian"
            ):
                multiplier *= 0.5

            # Жорсткий нічний фільтр для уникнення темних стежок біля ставків і пустирів
            if is_dark:
                if hw in ["path", "track", "footway", "steps"]:
                    if lit != "yes":
                        multiplier *= 25.0

                if surface in ["unpaved", "dirt", "ground", "grass", "gravel"]:
                    if lit != "yes":
                        multiplier *= 30.0

                if lit == "yes" or hw in [
                    "residential",
                    "living_street",
                    "primary",
                    "secondary",
                    "tertiary",
                ]:
                    multiplier *= 0.8

            data[weight_attr] = length * max(multiplier, 0.15)

        return weight_attr, applied_rules

    def _find_nearest_node(self, lat: float, lon: float) -> int:
        best_node = None
        min_dist = float("inf")
        for node_id, data in self.graph.nodes(data=True):
            n_lat, n_lon = data.get("y"), data.get("x")
            if n_lat is None or n_lon is None:
                continue
            d = (n_lat - lat) ** 2 + (n_lon - lon) ** 2
            if d < min_dist:
                min_dist = d
                best_node = node_id
        return best_node

    def audit_path_edges(self, node_path: List[int]) -> Dict[str, Any]:
        total_len = 0.0
        pedestrian_len = 0.0
        quiet_residential_len = 0.0
        traffic_len = 0.0
        cobblestone_len = 0.0

        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            edge_data = self.graph.get_edge_data(u, v)
            if not edge_data:
                continue
            first_key = list(edge_data.keys())[0]
            d = edge_data[first_key]
            l = float(d.get("length", 50.0))
            total_len += l

            hw = d.get("highway", "residential")
            if isinstance(hw, list):
                hw = hw[0]

            surface = d.get("surface", "")
            if isinstance(surface, list):
                surface = surface[0]

            if hw in ["pedestrian", "footway", "path", "living_street"]:
                pedestrian_len += l
            elif hw in ["residential", "unclassified"]:
                quiet_residential_len += l
            elif hw in ["primary", "secondary", "tertiary"]:
                traffic_len += l

            if surface in ["cobblestone", "paving_stones", "sett"]:
                cobblestone_len += l

        if total_len == 0:
            return {
                "pedestrian_share_pct": 0.0,
                "quiet_residential_pct": 0.0,
                "traffic_avoided_pct": 100.0,
                "cobblestone_pct": 0.0,
                "summary": "Direct path",
            }

        ped_pct = round((pedestrian_len / total_len) * 100.0, 1)
        quiet_pct = round((quiet_residential_len / total_len) * 100.0, 1)
        traffic_pct = round((traffic_len / total_len) * 100.0, 1)
        avoided_traffic_pct = round(100.0 - traffic_pct, 1)
        cobb_pct = round((cobblestone_len / total_len) * 100.0, 1)

        summary = (
            f"{ped_pct}% pedestrian/park paths, {quiet_pct}% quiet streets. "
            f"Successfully kept {avoided_traffic_pct}% of the route away from main car traffic."
        )

        return {
            "pedestrian_share_pct": ped_pct,
            "quiet_residential_pct": quiet_pct,
            "traffic_avoided_pct": avoided_traffic_pct,
            "cobblestone_pct": cobb_pct,
            "summary": summary,
        }

    def find_pois_on_the_way(
        self,
        path_coords: List[List[float]],
        candidate_pois: List[Dict[str, Any]],
        stops_coords: List[Tuple[float, float]],
        max_dist_m: float = 65.0,
    ) -> List[Dict[str, Any]]:
        on_the_way = []
        if len(path_coords) < 2:
            return on_the_way

        for poi in candidate_pois:
            p_lat, p_lon = poi["lat"], poi["lon"]
            min_dist = float("inf")
            best_coord_idx = 0

            for i in range(len(path_coords) - 1):
                a_lat, a_lon = path_coords[i]
                b_lat, b_lon = path_coords[i + 1]
                dist = distance_point_to_segment(
                    p_lat, p_lon, a_lat, a_lon, b_lat, b_lon
                )
                if dist < min_dist:
                    min_dist = dist
                    best_coord_idx = i

            if min_dist <= max_dist_m:
                insert_after_stop = 0
                stop_indices = []
                for s_lat, s_lon in stops_coords:
                    idx_min = min(
                        range(len(path_coords)),
                        key=lambda k: (path_coords[k][0] - s_lat) ** 2
                        + (path_coords[k][1] - s_lon) ** 2,
                    )
                    stop_indices.append(idx_min)

                for s_idx, p_idx in enumerate(stop_indices):
                    if best_coord_idx >= p_idx:
                        insert_after_stop = s_idx + 1
                    else:
                        break

                on_the_way.append(
                    {
                        "name": poi["name"],
                        "category": poi.get("category", "attraction"),
                        "description": poi.get("description", ""),
                        "distance_to_path_m": round(min_dist, 1),
                        "lat": p_lat,
                        "lon": p_lon,
                        "insert_after_order": insert_after_stop,
                    }
                )

        on_the_way.sort(key=lambda x: x["distance_to_path_m"])
        return on_the_way[:4]

    def build_itinerary(
        self,
        start_lat: float,
        start_lon: float,
        stops: List[Dict[str, Any]],
        final_coord: Optional[Tuple[float, float]] = None,
        weight_attr: str = "length",
        prevent_backtracking: bool = True,
    ) -> Dict[str, Any]:
        if not stops and final_coord is None:
            return {
                "total_distance_km": 0.0,
                "estimated_walking_time_min": 0.0,
                "estimated_steps": 0,
                "path_coordinates": [],
                "node_path": [],
                "stops_sequence": [],
                "leg_durations_min": [],
            }

        waypoints = [(start_lat, start_lon)] + [
            (s["lat"], s["lon"]) for s in stops
        ]
        if final_coord:
            waypoints.append(final_coord)

        nearest_nodes = [
            self._find_nearest_node(lat, lon) for lat, lon in waypoints
        ]

        full_node_path = []
        total_distance_m = 0.0
        used_edges = set()
        leg_durations_min = []

        for i in range(len(nearest_nodes) - 1):
            source = nearest_nodes[i]
            target = nearest_nodes[i + 1]
            if source is None or target is None:
                leg_durations_min.append(10.0)
                continue

            def edge_weight(u, v, d):
                base_w = d.get(weight_attr, d.get("length", 80.0))
                if prevent_backtracking:
                    if (u, v) in used_edges or (v, u) in used_edges:
                        return base_w * 7.0
                return base_w

            try:
                segment_path = nx.shortest_path(
                    self.graph, source, target, weight=edge_weight
                )
                segment_len = nx.path_weight(
                    self.graph, segment_path, weight="length"
                )

                for idx in range(len(segment_path) - 1):
                    used_edges.add((segment_path[idx], segment_path[idx + 1]))

                if full_node_path and segment_path:
                    full_node_path.extend(segment_path[1:])
                else:
                    full_node_path.extend(segment_path)

                total_distance_m += segment_len
                leg_durations_min.append(round(segment_len / 75.0, 1))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                leg_durations_min.append(10.0)
                continue

        path_coords = [
            [
                round(self.graph.nodes[n]["y"], 5),
                round(self.graph.nodes[n]["x"], 5),
            ]
            for n in full_node_path
            if "y" in self.graph.nodes[n] and "x" in self.graph.nodes[n]
        ]

        walking_time_min = round(total_distance_m / 75.0, 1)
        estimated_steps = int(round(total_distance_m / 0.75))

        return {
            "total_distance_km": round(total_distance_m / 1000.0, 2),
            "total_distance_m": round(total_distance_m, 1),
            "estimated_walking_time_min": walking_time_min,
            "estimated_steps": estimated_steps,
            "path_coordinates": path_coords,
            "node_path": full_node_path,
            "stops_sequence": [s["name"] for s in stops],
            "leg_durations_min": leg_durations_min,
        }

    def optimize_multistage_tour(
        self,
        start_lat: float,
        start_lon: float,
        candidates_per_stage: List[List[Dict[str, Any]]],
        dwell_times: List[int],
        max_total_minutes: float,
        finish_mode: str = "return_to_start",
        finish_coord: Optional[Tuple[float, float]] = None,
        routing_mode: str = "custom",
        vibe_preference: str = "",
        target_steps: int = 0,
        weather: str = "clear",
        hour: int = 14,
    ) -> Dict[str, Any]:
        import itertools

        applied_rules = []
        if routing_mode == "fastest":
            weight_attr = "length"
            applied_rules = [
                "Shortest Euclidean graph distance (Dijkstra over length)"
            ]
        else:
            weight_attr, applied_rules = self.compute_dynamic_weights(
                vibe_preference, weather=weather, hour=hour
            )

        applied_rules.append(
            "Anti-Backtracking Loop Guard: Visited segments penalized x7.0"
        )

        final_target = None
        if finish_mode == "return_to_start":
            final_target = (start_lat, start_lon)
            applied_rules.append(
                "Closed Loop Enforced: Returning to starting anchor"
            )
        elif finish_mode == "custom" and finish_coord:
            final_target = finish_coord
            applied_rules.append(
                f"Custom Destination Enforced: Finishing at ({finish_coord[0]:.4f}, {finish_coord[1]:.4f})"
            )

        valid_stages = [
            stage for stage in candidates_per_stage if len(stage) > 0
        ]
        if not valid_stages:
            return None

        best_itinerary = None
        best_objective_score = -float("inf")

        for combination in itertools.product(*valid_stages):
            stop_names = [c["name"] for c in combination]
            has_duplicates = len(stop_names) != len(set(stop_names))

            if (
                has_duplicates
                and len(valid_stages) > 1
                and all(len(s) > 1 for s in valid_stages)
            ):
                continue

            itinerary = self.build_itinerary(
                start_lat,
                start_lon,
                list(combination),
                final_coord=final_target,
                weight_attr=weight_attr,
                prevent_backtracking=True,
            )
            walk_time = itinerary["estimated_walking_time_min"]
            total_time = walk_time + sum(dwell_times[: len(combination)])
            steps = itinerary["estimated_steps"]

            semantic_score = sum(c.get("score", 0.0) * 1.5 for c in combination)

            if total_time <= max_total_minutes:
                budget_penalty = 0.0
            elif total_time <= max_total_minutes + 16.0:
                budget_penalty = -20.0 * (
                    (total_time - max_total_minutes) / 16.0
                )
            else:
                budget_penalty = (
                    -1000.0 - (total_time - max_total_minutes) * 10.0
                )

            diversity_bonus = min(itinerary["total_distance_km"] * 1.5, 6.0)

            if routing_mode == "fastest":
                objective_score = (
                    semantic_score - (walk_time * 0.25) + budget_penalty
                )
            elif routing_mode == "target_steps" and target_steps > 0:
                step_diff_ratio = abs(steps - target_steps) / float(
                    target_steps
                )
                objective_score = (
                    semantic_score - (step_diff_ratio * 7.0) + budget_penalty
                )
            else:
                objective_score = (
                    semantic_score
                    - (walk_time * 0.08)
                    + diversity_bonus
                    + budget_penalty
                )

            if objective_score > best_objective_score:
                best_objective_score = objective_score
                vibe_audit = self.audit_path_edges(itinerary["node_path"])
                best_itinerary = {
                    "itinerary": itinerary,
                    "stops": list(combination),
                    "total_time": total_time,
                    "walking_time": walk_time,
                    "dwell_time": sum(dwell_times[: len(combination)]),
                    "estimated_steps": steps,
                    "is_within_budget": total_time <= max_total_minutes,
                    "objective_score": round(objective_score, 2),
                    "vibe_audit": vibe_audit,
                    "applied_rules": applied_rules,
                    "leg_durations_min": itinerary.get(
                        "leg_durations_min", [10.0] * len(combination)
                    ),
                }

        if best_itinerary is None and valid_stages:
            fallback_comb = [s[0] for s in valid_stages]
            itinerary = self.build_itinerary(
                start_lat,
                start_lon,
                fallback_comb,
                final_coord=final_target,
                weight_attr=weight_attr,
                prevent_backtracking=False,
            )
            walk_time = itinerary["estimated_walking_time_min"]
            total_time = walk_time + sum(dwell_times[: len(fallback_comb)])
            best_itinerary = {
                "itinerary": itinerary,
                "stops": fallback_comb,
                "total_time": total_time,
                "walking_time": walk_time,
                "dwell_time": sum(dwell_times[: len(fallback_comb)]),
                "estimated_steps": itinerary["estimated_steps"],
                "is_within_budget": total_time <= max_total_minutes,
                "objective_score": 0.0,
                "vibe_audit": self.audit_path_edges(itinerary["node_path"]),
                "applied_rules": applied_rules,
                "leg_durations_min": itinerary.get(
                    "leg_durations_min", [10.0] * len(fallback_comb)
                ),
            }

        return best_itinerary