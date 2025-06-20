from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional


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
    segment_values: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "conversion_window_hours": self.conversion_window_hours,
            "counting_method": self.counting_method.value,
            "reentry_mode": self.reentry_mode.value,
            "funnel_order": self.funnel_order.value,
            "segment_by": self.segment_by,
            "segment_values": self.segment_values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunnelConfig":
        """Create from dictionary for JSON deserialization"""
        return cls(
            conversion_window_hours=data.get("conversion_window_hours", 168),
            counting_method=CountingMethod(data.get("counting_method", "unique_users")),
            reentry_mode=ReentryMode(data.get("reentry_mode", "first_only")),
            funnel_order=FunnelOrder(data.get("funnel_order", "ordered")),
            segment_by=data.get("segment_by"),
            segment_values=data.get("segment_values"),
        )


@dataclass
class TimeToConvertStats:
    """Statistics for time to convert analysis"""

    step_from: str
    step_to: str
    mean_hours: float
    median_hours: float
    p25_hours: float
    p75_hours: float
    p90_hours: float
    std_hours: float
    conversion_times: list[float]


@dataclass
class CohortData:
    """Cohort analysis data"""

    cohort_period: str
    cohort_sizes: dict[str, int]
    conversion_rates: dict[str, list[float]]
    cohort_labels: list[str]


@dataclass
class StatSignificanceResult:
    """Statistical significance test result"""

    segment_a: str
    segment_b: str
    conversion_a: float
    conversion_b: float
    p_value: float
    is_significant: bool
    confidence_interval: tuple[float, float]
    z_score: float


@dataclass
class PathAnalysisData:
    """Path analysis data"""

    dropoff_paths: dict[str, dict[str, int]]  # step -> {next_event: count}
    between_steps_events: dict[str, dict[str, int]]  # step_pair -> {event: count}


@dataclass
class FunnelResults:
    """Results of funnel analysis"""

    steps: list[str]
    users_count: list[int]
    conversion_rates: list[float]
    drop_offs: list[int]
    drop_off_rates: list[float]
    cohort_data: Optional[CohortData] = None
    segment_data: Optional[dict[str, list[int]]] = None
    time_to_convert: Optional[list[TimeToConvertStats]] = None
    path_analysis: Optional[PathAnalysisData] = None
    statistical_tests: Optional[list[StatSignificanceResult]] = None


@dataclass
class ProcessMiningData:
    """Process mining analysis results for user journey discovery"""

    activities: dict[
        str, dict[str, Any]
    ]  # activity_name -> {users, avg_time, type, success_rate}
    transitions: dict[
        tuple[str, str], dict[str, Any]
    ]  # (from, to) -> {frequency, users, avg_duration, probability}
    cycles: list[
        dict[str, Any]
    ]  # [{path: [steps], frequency: int, type: 'loop'|'cycle', impact: 'positive'|'negative'}]
    variants: list[
        dict[str, Any]
    ]  # [{path: [steps], frequency: int, success_rate: float, avg_duration: float}]
    start_activities: list[str]  # Process start events
    end_activities: list[str]  # Process end events
    statistics: dict[
        str, float
    ]  # {total_cases, avg_duration, completion_rate, unique_paths}
    insights: list[str]  # Auto-generated insights about the process

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessMiningData":
        """Create from dictionary"""
        return cls(**data)
