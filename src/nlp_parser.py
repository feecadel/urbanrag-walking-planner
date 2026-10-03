import re
from typing import Tuple, List, Dict, Any

class NaturalLanguageTourParser:
    def __init__(self):
        pass

    def parse(self, prompt: str) -> Tuple[float, List[Dict[str, Any]]]:
        hours = self._extract_time_budget(prompt)
        stages = self._extract_stages(prompt)
        return hours, stages

    def _extract_time_budget(self, text: str) -> float:
        hour_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hours|hour|hrs|hr)\b', text, re.IGNORECASE)
        if hour_match:
            return float(hour_match.group(1))

        min_match = re.search(r'(\d+)\s*(?:minutes|mins|min)\b', text, re.IGNORECASE)
        if min_match:
            return round(float(min_match.group(1)) / 60.0, 2)

        return 2.5

    def _extract_stages(self, text: str) -> List[Dict[str, Any]]:

        cleaned = re.sub(
            r'^\s*(?:I\s+(?:only\s+)?have|give\s+me|around|about)?\s*\d+(?:\.\d+)?\s*(?:hours|hour|hrs|hr|minutes|mins|min)(?:\s+in\s+the\s+[a-z]+)?\b[\.\,\;]?\s*',
            '', text, flags=re.IGNORECASE
        )
        cleaned = re.sub(r'^(?:first|initially|start\s+with)\s+', '', cleaned, flags=re.IGNORECASE)

        
        split_pattern = (
            r'\b(?:then|and then|after that|afterwards|finally|next|followed by|later|'
            r'(?:and\s+)?finish(?:\s+[a-z]+)?(?:\s+with|\s+at|\s+by|\s+in)?|'
            r'(?:and\s+)?end(?:\s+[a-z]+)?(?:\s+with|\s+at|\s+by|\s+in)?)\b|(?:\s*->\s*)'
        )
        raw_parts = re.split(split_pattern, cleaned, flags=re.IGNORECASE)

        stages = []
        for part in raw_parts:
            segment = part.strip(" ,.;\n")
            if len(segment) > 4 and not re.match(r'^(?:sitting|walking|visiting|with|at|by|in)\s*$', segment, re.IGNORECASE):
                clean_seg = re.sub(r'^(?:sitting\s+in|visiting|going\s+to|with|at|by)\s+', '', segment, flags=re.IGNORECASE).strip()
                
                stay_min = 30
                lower_seg = clean_seg.lower()
                if any(w in lower_seg for w in ["work", "study", "laptop", "tasks", "read", "coworking", "library"]):
                    stay_min = 40
                elif any(w in lower_seg for w in ["coffee", "tea", "drink", "snack", "pastry", "espresso", "cappuccino"]):
                    stay_min = 25
                elif any(w in lower_seg for w in ["walk", "park", "stroll", "jog", "ravines"]):
                    stay_min = 35
                elif any(w in lower_seg for w in ["water", "river", "beach", "lake", "view", "sunset", "viewpoint", "square", "bench"]):
                    stay_min = 25

                stages.append({
                    "intent": clean_seg,
                    "expected_stay_min": stay_min
                })

       
        if len(stages) <= 1 and "," in cleaned:
            comma_parts = [p.strip() for p in cleaned.split(",") if len(p.strip()) > 5]
            if len(comma_parts) > 1:
                stages = [{"intent": p, "expected_stay_min": 25} for p in comma_parts]

        return stages if stages else [{"intent": text.strip(), "expected_stay_min": 35}]