from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Location(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None

class RecentPost(BaseModel):
    post_id: str
    timestamp: str  # YYYY-MM-DD
    caption: Optional[str] = ""
    hashtags: List[str] = Field(default_factory=list)
    likes: Optional[int] = 0
    comments: Optional[int] = 0
    transcript: Optional[str] = None

class CreatorProfile(BaseModel):
    creator_id: str
    name: Optional[str] = None
    verticals: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Location = Field(default_factory=Location)
    language: Optional[str] = None

    bio: str = ""
    recent_posts: List[RecentPost] = Field(default_factory=list)

    follower_count: int = 0
    engagement_rate_30d: Optional[float] = None
    avg_views_30d: Optional[int] = None

    content_safety_labels: List[str] = Field(default_factory=lambda: ["none"])
    is_suspected_spam: bool = False
    authenticity_score: Optional[float] = None

class BrandRequest(BaseModel):
    description: str = ""
    hard_constraints: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
