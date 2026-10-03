from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PlaceResult(BaseModel):
    name: str
    category: str
    description: str
    distance_km: float
    bi_encoder_score: float
    cross_encoder_score: Optional[float] = None

class OnTheWayPOI(BaseModel):
    name: str
    category: str
    description: str
    distance_to_path_m: float
    lat: float
    lon: float
    insert_after_order: int = Field(default=1)

class VibeAuditMetrics(BaseModel):
    pedestrian_share_pct: float
    quiet_residential_pct: float
    traffic_avoided_pct: float
    cobblestone_pct: float
    summary: str

class RoutePlan(BaseModel):
    total_distance_km: float
    estimated_walking_time_min: float
    estimated_steps: int
    stops_sequence: List[str]
    path_coordinates: List[List[float]]

class SearchRequest(BaseModel):
    query: str = Field(..., example="quiet place to study or work with power sockets")
    lat: float = Field(..., ge=-90.0, le=90.0, example=49.5891)
    lon: float = Field(..., ge=-180.0, le=180.0, example=34.5513)
    radius_km: float = Field(default=7.5, gt=0, le=50.0)
    limit: int = Field(default=3, ge=1, le=10)
    use_reranker: bool = Field(default=True)
    build_route: bool = Field(default=True)

class SearchResponse(BaseModel):
    query: str
    reranked: bool
    total_found: int
    results: List[PlaceResult]
    route: Optional[RoutePlan] = None

class TourStagePreference(BaseModel):
    intent: str = Field(..., example="specialty coffee and pastry")
    expected_stay_min: int = Field(default=30, ge=5, le=120)
    forced_place_name: Optional[str] = Field(default=None)

class AssistantAction(BaseModel):
    action_type: str
    label: str
    description: str
    payload: Dict[str, Any]

class CandidatePOI(BaseModel):
    name: str
    category: str
    score: float
    lat: float
    lon: float
    distance_from_prev_km: float = Field(default=0.0, description="Дистанція від попередньої точки маршруту в км")

class TourStop(BaseModel):
    order: int
    intent: str
    name: str
    category: str
    description: str
    dwell_time_min: int
    lat: float
    lon: float
    opening_hours_raw: Optional[str] = None
    open_status: Optional[str] = "open"  # 'open', 'closing_soon', 'closed', 'unknown'
    open_status_note: Optional[str] = None  # e.g. "Closes at 22:00" or "Closed at this hour"
    alternative_candidates: Optional[List[CandidatePOI]] = []

class TourPlanRequest(BaseModel):
    lat: float = Field(..., example=49.5891)
    lon: float = Field(..., example=34.5513)
    finish_mode: str = Field(default="return_to_start", description="'return_to_start' | 'custom' | 'open_ended'")
    finish_lat: Optional[float] = Field(default=None)
    finish_lon: Optional[float] = Field(default=None)
    total_budget_hours: float = Field(default=2.5, ge=0.5, le=8.0)
    routing_mode: str = Field(default="custom")
    route_vibe_preference: str = Field(default="historic architecture, avoid cars")
    target_steps: int = Field(default=0)
    weather_condition: str = Field(default="clear")
    start_time_hour: int = Field(default=14, ge=0, le=23)
    stages: List[TourStagePreference]
    expand_loop: bool = Field(default=False)
    force_trim_dwell: bool = Field(default=False)
    excluded_place_names: List[str] = Field(default=[])

class FreeTextTourRequest(BaseModel):
    user_prompt: str = Field(..., example="I have 2.5 hours. Grab a coffee, walk in park, viewpoint.")
    lat: float = Field(default=49.5891)
    lon: float = Field(default=34.5513)
    finish_mode: str = Field(default="return_to_start")
    finish_lat: Optional[float] = None
    finish_lon: Optional[float] = None
    routing_mode: str = Field(default="custom")
    route_vibe_preference: str = Field(default="historic architecture, avoid traffic noise")
    target_steps: int = Field(default=0)
    weather_condition: str = Field(default="clear")
    start_time_hour: int = Field(default=14)
    expand_loop: bool = Field(default=False)
    force_trim_dwell: bool = Field(default=False)
    excluded_place_names: List[str] = Field(default=[])
    forced_replacements: Dict[str, str] = Field(default={})
    explicit_stages: Optional[List[TourStagePreference]] = None

class TourPlanResponse(BaseModel):
    status: str
    total_time_min: float
    walking_time_min: float
    stops_time_min: float
    time_budget_min: float
    is_within_budget: bool
    total_distance_km: float
    estimated_steps: int
    stops: List[TourStop]
    path_coordinates: List[List[float]]
    summary: str
    vibe_audit: Optional[VibeAuditMetrics] = None
    on_the_way: List[OnTheWayPOI] = []
    assistant_recommendations: List[str] = []
    available_actions: List[AssistantAction] = []
    reasoning_log: Optional[str] = Field(default="")
    gpx_xml: Optional[str] = None
    google_maps_url: Optional[str] = None