#!/usr/bin/env python3
"""
Comprehensive Test Suite for FunnelConfigManager
===============================================

This module provides comprehensive testing for FunnelConfigManager functionality,
which was completely missing from the test suite.

Test Categories:
1. Configuration Saving and Loading
2. JSON Serialization/Deserialization
3. Download Link Generation
4. Error Handling and Edge Cases
5. Configuration Validation
6. Backward Compati            "config": {
                "conversion_window_hours": 72,
                "counting_method": "unique_users",
                "reentry_mode": "first_only", 
                "funnel_order": "ordered"
            },

Professional Testing Standards:
- Comprehensive edge case coverage
- JSON schema validation
- File I/O error simulation
- Cross-platform compatibility testing
- Type safety validation
"""

import pytest
import json
import tempfile
import base64
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from app import FunnelConfigManager
from models import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder


@pytest.mark.config_management
class TestFunnelConfigManagerSaveLoad:
    """Test core save and load functionality for funnel configurations."""
    
    @pytest.fixture
    def sample_config(self):
        """Standard funnel configuration for testing."""
        return FunnelConfig(
            conversion_window_hours=168,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED
        )
    
    @pytest.fixture
    def sample_steps(self):
        """Standard funnel steps for testing."""
        return ["Sign Up", "Email Verification", "First Login", "First Purchase"]
    
    @pytest.fixture
    def sample_config_name(self):
        """Standard configuration name for testing."""
        return "Test_Funnel_Config_v1"
    
    def test_save_config_basic(self, sample_config, sample_steps, sample_config_name):
        """Test basic configuration saving functionality."""
        config_json = FunnelConfigManager.save_config(
            sample_config, 
            sample_steps, 
            sample_config_name
        )
        
        # Validate JSON structure
        assert config_json is not None, "Should return JSON string"
        assert isinstance(config_json, str), "Should return string"
        
        # Parse and validate JSON content
        config_data = json.loads(config_json)
        
        # Validate required fields
        required_fields = ['name', 'steps', 'config', 'saved_at']
        for field in required_fields:
            assert field in config_data, f"Missing required field: {field}"
        
        # Validate field values
        assert config_data['name'] == sample_config_name
        assert config_data['steps'] == sample_steps
        assert isinstance(config_data['config'], dict)
        assert 'saved_at' in config_data
        
        # Validate datetime format
        saved_at = datetime.fromisoformat(config_data['saved_at'])
        assert saved_at is not None, "Should have valid timestamp"
        
        print("✅ Basic config save test passed")
    
    def test_load_config_basic(self, sample_config, sample_steps, sample_config_name):
        """Test basic configuration loading functionality."""
        # First save a config
        config_json = FunnelConfigManager.save_config(
            sample_config, 
            sample_steps, 
            sample_config_name
        )
        
        # Then load it back
        loaded_config, loaded_steps, loaded_name = FunnelConfigManager.load_config(config_json)
        
        # Validate loaded configuration
        assert isinstance(loaded_config, FunnelConfig), "Should return FunnelConfig object"
        assert loaded_steps == sample_steps, "Steps should match"
        assert loaded_name == sample_config_name, "Name should match"
        
        # Validate configuration values
        assert loaded_config.conversion_window_hours == sample_config.conversion_window_hours
        assert loaded_config.counting_method == sample_config.counting_method
        assert loaded_config.reentry_mode == sample_config.reentry_mode
        assert loaded_config.funnel_order == sample_config.funnel_order
        
        print("✅ Basic config load test passed")
    
    def test_save_load_roundtrip(self, sample_config, sample_steps, sample_config_name):
        """Test complete save-load roundtrip maintains data integrity."""
        # Save config
        saved_json = FunnelConfigManager.save_config(
            sample_config, 
            sample_steps, 
            sample_config_name
        )
        
        # Load config
        loaded_config, loaded_steps, loaded_name = FunnelConfigManager.load_config(saved_json)
        
        # Save again
        resaved_json = FunnelConfigManager.save_config(
            loaded_config, 
            loaded_steps, 
            loaded_name
        )
        
        # Parse both JSON strings for comparison (ignoring timestamps)
        original_data = json.loads(saved_json)
        resaved_data = json.loads(resaved_json)
        
        # Compare core data (excluding timestamps)
        assert original_data['name'] == resaved_data['name']
        assert original_data['steps'] == resaved_data['steps']
        assert original_data['config'] == resaved_data['config']
        
        print("✅ Save-load roundtrip test passed")
    
    def test_all_counting_methods_save_load(self, sample_steps, sample_config_name):
        """Test save/load works for all counting methods."""
        for counting_method in CountingMethod:
            config = FunnelConfig(
                conversion_window_hours=24,
                counting_method=counting_method,
                reentry_mode=ReentryMode.FIRST_ONLY,
                funnel_order=FunnelOrder.ORDERED
            )
            
            # Save and load
            saved_json = FunnelConfigManager.save_config(config, sample_steps, sample_config_name)
            loaded_config, _, _ = FunnelConfigManager.load_config(saved_json)
            
            # Validate counting method preserved
            assert loaded_config.counting_method == counting_method, f"Failed for {counting_method}"
        
        print("✅ All counting methods save/load test passed")
    
    def test_all_reentry_modes_save_load(self, sample_steps, sample_config_name):
        """Test save/load works for all reentry modes."""
        for reentry_mode in ReentryMode:
            config = FunnelConfig(
                conversion_window_hours=24,
                counting_method=CountingMethod.UNIQUE_USERS,
                reentry_mode=reentry_mode,
                funnel_order=FunnelOrder.ORDERED
            )
            
            # Save and load
            saved_json = FunnelConfigManager.save_config(config, sample_steps, sample_config_name)
            loaded_config, _, _ = FunnelConfigManager.load_config(saved_json)
            
            # Validate reentry mode preserved
            assert loaded_config.reentry_mode == reentry_mode, f"Failed for {reentry_mode}"
        
        print("✅ All reentry modes save/load test passed")
    
    def test_all_funnel_orders_save_load(self, sample_steps, sample_config_name):
        """Test save/load works for all funnel orders."""
        for funnel_order in FunnelOrder:
            config = FunnelConfig(
                conversion_window_hours=24,
                counting_method=CountingMethod.UNIQUE_USERS,
                reentry_mode=ReentryMode.FIRST_ONLY,
                funnel_order=funnel_order
            )
            
            # Save and load
            saved_json = FunnelConfigManager.save_config(config, sample_steps, sample_config_name)
            loaded_config, _, _ = FunnelConfigManager.load_config(saved_json)
            
            # Validate funnel order preserved
            assert loaded_config.funnel_order == funnel_order, f"Failed for {funnel_order}"
        
        print("✅ All funnel orders save/load test passed")


@pytest.mark.config_management
class TestFunnelConfigManagerDownloadLinks:
    """Test download link generation functionality."""
    
    def test_create_download_link_basic(self):
        """Test basic download link creation."""
        config_json = '{"name": "test", "steps": ["A", "B"], "config": {}}'
        filename = "test_config.json"
        
        download_link = FunnelConfigManager.create_download_link(config_json, filename)
        
        # Validate link structure
        assert download_link is not None, "Should return download link"
        assert isinstance(download_link, str), "Should return string"
        assert 'href="data:application/json;base64,' in download_link, "Should contain base64 data URL"
        assert f'download="{filename}"' in download_link, "Should contain filename"
        
        print("✅ Basic download link creation test passed")
    
    def test_download_link_base64_encoding(self):
        """Test base64 encoding in download links is correct."""
        test_json = '{"test": "data", "number": 123}'
        filename = "test.json"
        
        download_link = FunnelConfigManager.create_download_link(test_json, filename)
        
        # Extract base64 data from link
        start_marker = 'data:application/json;base64,'
        end_marker = '" download='
        
        start_idx = download_link.find(start_marker) + len(start_marker)
        end_idx = download_link.find(end_marker)
        
        base64_data = download_link[start_idx:end_idx]
        
        # Decode and verify
        decoded_data = base64.b64decode(base64_data).decode('utf-8')
        assert decoded_data == test_json, "Base64 encoding/decoding should preserve data"
        
        print("✅ Download link base64 encoding test passed")
    
    def test_download_link_special_characters(self):
        """Test download links work with special characters in filenames."""
        config_json = '{"name": "test"}'
        special_filenames = [
            "config with spaces.json",
            "config-with-dashes.json", 
            "config_with_underscores.json",
            "config123.json",
            "config.v2.0.json"
        ]
        
        for filename in special_filenames:
            download_link = FunnelConfigManager.create_download_link(config_json, filename)
            
            # Should create valid link
            assert download_link is not None, f"Should handle filename: {filename}"
            assert 'href="data:' in download_link, f"Should be valid data URL for: {filename}"
            assert f'download="{filename}"' in download_link, f"Should preserve filename: {filename}"
        
        print("✅ Download link special characters test passed")
    
    def test_download_link_large_config(self):
        """Test download links work with large configuration files."""
        # Create large config with many steps
        large_config = {
            "name": "Large_Funnel_Config",
            "steps": [f"Step_{i:03d}" for i in range(1, 101)],  # 100 steps
            "config": {
                "conversion_window_hours": 168,
                "counting_method": "UNIQUE_USERS",
                "reentry_mode": "FIRST_ONLY",
                "funnel_order": "ORDERED",
                "metadata": {
                    "description": "A" * 1000,  # Large description
                    "tags": [f"tag_{i}" for i in range(50)]  # Many tags
                }
            },
            "saved_at": datetime.now().isoformat()
        }
        
        large_json = json.dumps(large_config, indent=2)
        
        download_link = FunnelConfigManager.create_download_link(large_json, "large_config.json")
        
        # Should handle large data
        assert download_link is not None, "Should handle large configurations"
        assert len(download_link) > 1000, "Link should contain substantial data"
        assert 'data:application/json;base64,' in download_link, "Should use correct data URL format"
        
        print("✅ Download link large config test passed")


@pytest.mark.config_management
class TestFunnelConfigManagerEdgeCases:
    """Test edge cases and error handling for configuration management."""
    
    def test_save_empty_steps(self):
        """Test saving configuration with empty steps."""
        config = FunnelConfig()
        empty_steps = []
        config_name = "Empty_Steps_Config"
        
        # Should handle gracefully
        config_json = FunnelConfigManager.save_config(config, empty_steps, config_name)
        
        assert config_json is not None, "Should handle empty steps"
        
        # Validate can be loaded back
        loaded_config, loaded_steps, loaded_name = FunnelConfigManager.load_config(config_json)
        assert loaded_steps == [], "Empty steps should be preserved"
        
        print("✅ Save empty steps test passed")
    
    def test_save_long_config_name(self):
        """Test saving configuration with very long name."""
        config = FunnelConfig()
        steps = ["A", "B", "C"]
        long_name = "Very_Long_Configuration_Name_" + "X" * 200
        
        # Should handle long names
        config_json = FunnelConfigManager.save_config(config, steps, long_name)
        
        assert config_json is not None, "Should handle long names"
        
        # Validate can be loaded back
        loaded_config, loaded_steps, loaded_name = FunnelConfigManager.load_config(config_json)
        assert loaded_name == long_name, "Long name should be preserved"
        
        print("✅ Save long config name test passed")
    
    def test_save_unicode_characters(self):
        """Test saving configuration with Unicode characters."""
        config = FunnelConfig()
        unicode_steps = ["🔥 Sign Up", "✉️ Email Verify", "🎯 First Login", "💰 Purchase"]
        unicode_name = "Funnel_配置_🚀"
        
        # Should handle Unicode
        config_json = FunnelConfigManager.save_config(config, unicode_steps, unicode_name)
        
        assert config_json is not None, "Should handle Unicode characters"
        
        # Validate can be loaded back
        loaded_config, loaded_steps, loaded_name = FunnelConfigManager.load_config(config_json)
        assert loaded_steps == unicode_steps, "Unicode steps should be preserved"
        assert loaded_name == unicode_name, "Unicode name should be preserved"
        
        print("✅ Save Unicode characters test passed")
    
    def test_load_malformed_json(self):
        """Test loading malformed JSON gracefully fails."""
        malformed_jsons = [
            "",  # Empty string
            "{",  # Incomplete JSON
            "{'invalid': 'quotes'}",  # Single quotes
            '{"missing": }',  # Missing value
            "not json at all",  # Not JSON
            '{"number": NaN}',  # Invalid number
        ]
        
        for malformed_json in malformed_jsons:
            with pytest.raises((json.JSONDecodeError, KeyError, ValueError)):
                FunnelConfigManager.load_config(malformed_json)
        
        print("✅ Load malformed JSON test passed")
    
    def test_load_missing_required_fields(self):
        """Test loading JSON with missing required fields."""
        incomplete_configs = [
            '{}',  # Empty object
            '{"name": "test"}',  # Missing steps and config
            '{"steps": ["A", "B"]}',  # Missing name and config
            '{"config": {}}',  # Missing name and steps
            '{"name": "test", "steps": ["A"]}',  # Missing config
        ]
        
        for incomplete_config in incomplete_configs:
            with pytest.raises(KeyError):
                FunnelConfigManager.load_config(incomplete_config)
        
        print("✅ Load missing required fields test passed")
    
    def test_load_invalid_config_structure(self):
        """Test loading JSON with invalid config structure."""
        # Valid JSON but invalid config structure
        invalid_config = json.dumps({
            "name": "test",
            "steps": ["A", "B"],
            "config": {
                "conversion_window_hours": "invalid",  # Should be number
                "counting_method": "INVALID_METHOD",   # Invalid enum
                "reentry_mode": 999,                   # Invalid type
                "funnel_order": None                   # Invalid value
            }
        })
        
        # Should raise appropriate exception when creating FunnelConfig
        with pytest.raises((ValueError, TypeError, AttributeError)):
            FunnelConfigManager.load_config(invalid_config)
        
        print("✅ Load invalid config structure test passed")
    
    def test_extreme_conversion_window_values(self):
        """Test save/load with extreme conversion window values."""
        extreme_values = [0, 1, 8760, 87600, 999999]  # 0 hours to very large
        
        for hours in extreme_values:
            config = FunnelConfig(conversion_window_hours=hours)
            steps = ["A", "B"]
            name = f"Config_{hours}h"
            
            # Save and load
            saved_json = FunnelConfigManager.save_config(config, steps, name)
            loaded_config, _, _ = FunnelConfigManager.load_config(saved_json)
            
            # Validate value preserved
            assert loaded_config.conversion_window_hours == hours, f"Failed for {hours} hours"
        
        print("✅ Extreme conversion window values test passed")


@pytest.mark.config_management  
class TestFunnelConfigManagerBackwardCompatibility:
    """Test backward compatibility with older configuration formats."""
    
    def test_load_minimal_config_format(self):
        """Test loading minimal configuration format."""
        # Minimal config that might come from older versions
        minimal_config = json.dumps({
            "name": "Minimal_Config",
            "steps": ["Step1", "Step2"],
            "config": {
                "conversion_window_hours": 24
                # Missing other fields - should use defaults
            }
        })
        
        # Should load with defaults for missing fields
        loaded_config, loaded_steps, loaded_name = FunnelConfigManager.load_config(minimal_config)
        
        assert loaded_config is not None, "Should load minimal config"
        assert loaded_config.conversion_window_hours == 24, "Should preserve specified value"
        # Other fields should use FunnelConfig defaults
        
        print("✅ Load minimal config format test passed")
    
    def test_load_config_with_extra_fields(self):
        """Test loading configuration with extra unknown fields."""
        # Config with extra fields that might be added in future versions
        extended_config = json.dumps({
            "name": "Extended_Config",
            "steps": ["A", "B", "C"],
            "config": {
                "conversion_window_hours": 168,
                "counting_method": "unique_users",
                "reentry_mode": "first_only", 
                "funnel_order": "ordered"
            },
            "version": "2.0",  # Extra field
            "metadata": {      # Extra nested field
                "created_by": "user123",
                "tags": ["test", "production"]
            },
            "settings": {      # Extra field
                "advanced_mode": True
            }
        })
        
        # Should load successfully, ignoring extra fields
        loaded_config, loaded_steps, loaded_name = FunnelConfigManager.load_config(extended_config)
        
        assert loaded_config is not None, "Should load config with extra fields"
        assert loaded_steps == ["A", "B", "C"], "Steps should be preserved"
        assert loaded_name == "Extended_Config", "Name should be preserved"
        
        print("✅ Load config with extra fields test passed")
    
    def test_config_format_evolution(self):
        """Test configuration format can evolve while maintaining compatibility."""
        # Create config with current format
        current_config = FunnelConfig(
            conversion_window_hours=72,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.OPTIMIZED_REENTRY,
            funnel_order=FunnelOrder.UNORDERED
        )
        
        steps = ["Registration", "Activation", "Retention"]
        name = "Evolution_Test"
        
        # Save with current format
        saved_json = FunnelConfigManager.save_config(current_config, steps, name)
        
        # Parse and modify to simulate future format changes
        config_data = json.loads(saved_json)
        
        # Add hypothetical future fields
        config_data["format_version"] = "3.0"
        config_data["config"]["new_feature_enabled"] = True
        config_data["config"]["advanced_settings"] = {
            "parallel_processing": True,
            "cache_enabled": False
        }
        
        modified_json = json.dumps(config_data)
        
        # Should still load successfully
        loaded_config, loaded_steps, loaded_name = FunnelConfigManager.load_config(modified_json)
        
        assert loaded_config is not None, "Should handle evolved format"
        assert loaded_config.conversion_window_hours == 72, "Core config should be preserved"
        assert loaded_steps == steps, "Steps should be preserved"
        assert loaded_name == name, "Name should be preserved"
        
        print("✅ Config format evolution test passed")


# Integration test to verify config manager works with real file I/O
@pytest.mark.config_management
@pytest.mark.integration
class TestFunnelConfigManagerFileIntegration:
    """Test configuration manager with actual file operations."""
    
    def test_save_load_to_actual_file(self):
        """Test saving and loading configuration to/from actual files."""
        config = FunnelConfig(
            conversion_window_hours=120,
            counting_method=CountingMethod.EVENT_TOTALS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED
        )
        steps = ["Acquisition", "Activation", "Revenue"]
        name = "File_Integration_Test"
        
        # Save config to JSON
        config_json = FunnelConfigManager.save_config(config, steps, name)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(config_json)
            temp_file_path = f.name
        
        try:
            # Read from file
            with open(temp_file_path, 'r') as f:
                loaded_json = f.read()
            
            # Load config from file content
            loaded_config, loaded_steps, loaded_name = FunnelConfigManager.load_config(loaded_json)
            
            # Validate file round-trip
            assert loaded_config.conversion_window_hours == config.conversion_window_hours
            assert loaded_config.counting_method == config.counting_method
            assert loaded_config.reentry_mode == config.reentry_mode
            assert loaded_config.funnel_order == config.funnel_order
            assert loaded_steps == steps
            assert loaded_name == name
            
        finally:
            # Clean up temporary file
            Path(temp_file_path).unlink(missing_ok=True)
        
        print("✅ Save/load to actual file test passed")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "--tb=short"])
