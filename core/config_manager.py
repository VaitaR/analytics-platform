"""
Configuration management for funnel analytics.

This module handles saving, loading, and managing funnel configurations
including JSON serialization and download link generation.
"""

import base64
import json
from datetime import datetime
from typing import Tuple

from models import FunnelConfig


class FunnelConfigManager:
    """Manages saving and loading of funnel configurations"""

    @staticmethod
    def save_config(config: FunnelConfig, steps: list[str], name: str) -> str:
        """Save funnel configuration to JSON string"""
        config_data = {
            "name": name,
            "steps": steps,
            "config": config.to_dict(),
            "saved_at": datetime.now().isoformat(),
        }
        return json.dumps(config_data, indent=2)

    @staticmethod
    def load_config(config_json: str) -> Tuple[FunnelConfig, list[str], str]:
        """Load funnel configuration from JSON string"""
        config_data = json.loads(config_json)

        config = FunnelConfig.from_dict(config_data["config"])
        steps = config_data["steps"]
        name = config_data["name"]

        return config, steps, name

    @staticmethod
    def create_download_link(config_json: str, filename: str) -> str:
        """Create download link for configuration"""
        b64 = base64.b64encode(config_json.encode()).decode()
        return f'<a href="data:application/json;base64,{b64}" download="{filename}">Download Configuration</a>'
