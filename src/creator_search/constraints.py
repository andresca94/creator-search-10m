from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class Constraints:
    min_followers: int = 0
    country: Optional[str] = None
    city: Optional[str] = None
    languages: Optional[List[str]] = None

def parse_constraints(req_obj: Dict[str, Any]) -> Constraints:
    hc = req_obj.get("hard_constraints") or {}
    geo = hc.get("geo") or {}
    langs = hc.get("languages") or hc.get("language")

    languages = None
    if isinstance(langs, list):
        languages = [str(x) for x in langs]
    elif isinstance(langs, str):
        languages = [langs]

    return Constraints(
        min_followers=int(hc.get("min_followers", 0) or 0),
        country=str(geo.get("country")) if geo.get("country") else None,
        city=str(geo.get("city")) if geo.get("city") else None,
        languages=languages,
    )
