from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

class CountingMethod(Enum):
    UNIQUE_USERS = "unique_users"
    EVENT_TOTALS = "event_totals"
    UNIQUE_PAIRS = "unique_pairs"

class ReentryMode(Enum):
    FIRST_ONLY = "first_only"
    OPTIMIZED_REENTRY = "optimized_reentry"

class FunnelOrder(Enum):
    ORDERED = "ordered"
    UNORDERED = "unordered"

@dataclass
class FunnelConfig:
    """Configuration for funnel analysis"""
    conversion_window_hours: int = 168  # 7 days default
    counting_method: CountingMethod = CountingMethod.UNIQUE_USERS
    reentry_mode: ReentryMode = ReentryMode.FIRST_ONLY
    funnel_order: FunnelOrder = FunnelOrder.ORDERED
    segment_by: Optional[str] = None
    segment_values: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'conversion_window_hours': self.conversion_window_hours,
            'counting_method': self.counting_method.value,
            'reentry_mode': self.reentry_mode.value,
            'funnel_order': self.funnel_order.value,
            'segment_by': self.segment_by,
            'segment_values': self.segment_values
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunnelConfig':
        """Create from dictionary for JSON deserialization"""
        return cls(
            conversion_window_hours=data.get('conversion_window_hours', 168),
            counting_method=CountingMethod(data.get('counting_method', 'unique_users')),
            reentry_mode=ReentryMode(data.get('reentry_mode', 'first_only')),
            funnel_order=FunnelOrder(data.get('funnel_order', 'ordered')),
            segment_by=data.get('segment_by'),
            segment_values=data.get('segment_values')
        )

@dataclass
class PathAnalysisData:
    """Path analysis data"""
    dropoff_paths: Dict[str, Dict[str, int]]  # step -> {next_event: count}
    between_steps_events: Dict[str, Dict[str, int]]  # step_pair -> {event: count} 