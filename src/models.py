from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class AppListing:
    name: str
    publisher: str
    description: str
    permissions: str
    id: Optional[int] = None
    risk_score: float = 0.0
    risk_reasons: str = "Pending Analysis"

    def to_dict(self):
        return asdict(self)