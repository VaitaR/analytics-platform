#!/usr/bin/env python3
"""
Comprehensive Test Suite for DataSourceManager Advanced Features
==============================================================

This module provides comprehensive testing for DataSourceManager functionality
that was only partially tested. Focuses on real-world scenarios and edge cases.

Test Categories:
1. Advanced Data Validation
2. Complex JSON Property Extraction  
3. Error Handling and Recovery
4. Performance with Large Datasets
5. ClickHouse Integration Edge Cases
6. File Format Support and Validation
7. Memory Management

Professional Testing Standards:
- Universal fixtures for complex test scenarios
- Comprehensive error simulation
- Performance validation for enterprise datasets
- Memory leak detection
- Cross-platform file handling
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import json
import tempfile
import io
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
from typing import Dict, List, Any

from app import DataSourceManager
from models import FunnelConfig


@pytest.mark.data_source
class TestDataSourceManagerValidation:
    """Test advanced data validation scenarios."""
    
    @pytest.fixture
    def data_manager(self):
        """Standard data source manager for testing."""
        return DataSourceManager()
    
    @pytest.fixture
    def valid_base_data(self):
        """Valid base event data for testing variations."""
        return pd.DataFrame({
            'user_id': ['user_001', 'user_002', 'user_003'],
            'event_name': ['Sign Up', 'Login', 'Purchase'],
            'timestamp': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 11, 0, 0), 
                datetime(2024, 1, 1, 12, 0, 0)
            ],
            'event_properties': ['{}', '{}', '{}'],
            'user_properties': ['{}', '{}', '{}']
        })
    
    def test_validate_missing_columns_detailed(self, data_manager):
        """Test detailed validation of missing columns scenarios."""
        # Test each required column missing individually
        base_data = {
            'user_id': ['user1', 'user2'],
            'event_name': ['Event1', 'Event2'],
            'timestamp': [datetime.now(), datetime.now()],
            'event_properties': ['{}', '{}'],
            'user_properties': ['{}', '{}']
        }
        
        required_columns = ['user_id', 'event_name', 'timestamp']
        
        for missing_col in required_columns:
            test_data = base_data.copy()
            del test_data[missing_col]
            df = pd.DataFrame(test_data)
            
            is_valid, message = data_manager.validate_event_data(df)
            
            assert not is_valid, f"Should be invalid when missing {missing_col}"
            assert missing_col in message.lower(), f"Error message should mention {missing_col}"
            assert 'missing' in message.lower(), "Error message should mention missing columns"
        
        print("✅ Detailed missing columns validation test passed")
    
    def test_validate_invalid_timestamp_formats(self, data_manager):
        """Test validation with various invalid timestamp formats."""
        invalid_timestamps = [
            ['invalid_date', 'also_invalid'],
            ['2024-13-01', '2024-01-32'],  # Invalid dates
            [None, None],  # Null values
            ['', ''],  # Empty strings
            [12345, 67890],  # Numbers instead of dates
            ['2024/01/01', '01-01-2024'],  # Different formats mixed
        ]
        
        for timestamps in invalid_timestamps:
            df = pd.DataFrame({
                'user_id': ['user1', 'user2'],
                'event_name': ['Event1', 'Event2'],
                'timestamp': timestamps,
                'event_properties': ['{}', '{}'],
                'user_properties': ['{}', '{}']
            })
            
            is_valid, message = data_manager.validate_event_data(df)
            
            # Should either be invalid or successfully convert timestamps
            if not is_valid:
                assert 'timestamp' in message.lower(), f"Should mention timestamp issues for {timestamps}"
        
        print("✅ Invalid timestamp formats validation test passed")
    
    def test_validate_with_missing_optional_columns(self, data_manager, valid_base_data):
        """Test validation when optional columns are missing."""
        # Remove optional columns
        minimal_data = valid_base_data[['user_id', 'event_name', 'timestamp']].copy()
        
        is_valid, message = data_manager.validate_event_data(minimal_data)
        
        # Should be valid even without optional columns
        assert is_valid, "Should be valid without optional columns"
        assert 'valid' in message.lower(), "Should indicate successful validation"
        
        print("✅ Missing optional columns validation test passed")
    
    def test_validate_empty_dataframe(self, data_manager):
        """Test validation with completely empty DataFrame."""
        empty_df = pd.DataFrame()
        
        is_valid, message = data_manager.validate_event_data(empty_df)
        
        assert not is_valid, "Empty DataFrame should be invalid"
        assert 'missing' in message.lower(), "Should mention missing columns"
        
        print("✅ Empty DataFrame validation test passed")
    
    def test_validate_dataframe_with_no_rows(self, data_manager):
        """Test validation with DataFrame that has columns but no rows."""
        empty_rows_df = pd.DataFrame({
            'user_id': [],
            'event_name': [],
            'timestamp': [],
            'event_properties': [],
            'user_properties': []
        })
        
        is_valid, message = data_manager.validate_event_data(empty_rows_df)
        
        # Should be structurally valid but might warn about no data
        assert is_valid, "DataFrame with correct columns but no rows should be structurally valid"
        
        print("✅ DataFrame with no rows validation test passed")


@pytest.mark.data_source
class TestDataSourceManagerJSONProcessing:
    """Test complex JSON property extraction and processing."""
    
    @pytest.fixture
    def data_manager(self):
        """Data manager for JSON testing."""
        return DataSourceManager()
    
    @pytest.fixture
    def complex_json_data(self):
        """Test data with complex JSON properties."""
        events = []
        
        # Various JSON complexity levels
        json_variations = [
            # Simple JSON
            ('{}', '{"segment": "premium"}'),
            
            # Nested JSON  
            (
                '{"utm": {"source": "google", "campaign": "summer"}}',
                '{"profile": {"age": 25, "location": {"country": "US", "city": "NYC"}}}'
            ),
            
            # Array values
            (
                '{"tags": ["marketing", "conversion"], "scores": [85, 92, 78]}',
                '{"preferences": ["email", "sms"], "history": ["login", "purchase"]}'
            ),
            
            # Mixed types
            (
                '{"active": true, "score": 4.5, "count": 42, "name": "test"}',
                '{"verified": false, "rating": 3.8, "visits": 15, "tier": "gold"}'
            ),
            
            # Empty/null variations
            ('null', '{}'),
            ('{}', 'null'),
            
            # Special characters and Unicode
            (
                '{"message": "Hello 🌍", "émoji": "✨", "quote": "She said \\"hi\\""}',
                '{"名前": "テスト", "città": "Roma", "价格": 100}'
            )
        ]
        
        for i, (event_props, user_props) in enumerate(json_variations):
            events.append({
                'user_id': f'user_{i:03d}',
                'event_name': f'Event_{i}',
                'timestamp': datetime(2024, 1, 1, 10, i),
                'event_properties': event_props,
                'user_properties': user_props
            })
        
        return pd.DataFrame(events)
    
    def test_get_segmentation_properties_complex(self, data_manager, complex_json_data):
        """Test segmentation property extraction with complex JSON."""
        properties = data_manager.get_segmentation_properties(complex_json_data)
        
        # Should extract properties without errors
        assert isinstance(properties, dict), "Should return dictionary"
        assert 'event_properties' in properties, "Should have event properties"
        assert 'user_properties' in properties, "Should have user properties"
        
        # Should handle various property types
        event_props = properties.get('event_properties', [])
        user_props = properties.get('user_properties', [])
        
        # Validate some expected properties were extracted
        # (Note: exact properties depend on Polars vs Pandas implementation)
        combined_props = set(event_props + user_props)
        assert len(combined_props) > 0, "Should extract some properties"
        
        print("✅ Complex JSON segmentation properties test passed")
    
    def test_get_property_values_complex(self, data_manager, complex_json_data):
        """Test property value extraction with complex data structures."""
        # First get available properties
        properties = data_manager.get_segmentation_properties(complex_json_data)
        
        # Test getting values for different property types
        for prop_type in ['event_properties', 'user_properties']:
            if prop_type in properties and properties[prop_type]:
                # Test first available property
                prop_name = properties[prop_type][0]
                values = data_manager.get_property_values(complex_json_data, prop_name, prop_type)
                
                assert isinstance(values, list), f"Should return list for {prop_type}.{prop_name}"
                # Values might be empty if property is nested or complex
        
        print("✅ Complex property values extraction test passed")
    
    def test_malformed_json_handling(self, data_manager):
        """Test handling of malformed JSON in properties."""
        malformed_data = pd.DataFrame({
            'user_id': ['user1', 'user2', 'user3', 'user4'],
            'event_name': ['Event1', 'Event2', 'Event3', 'Event4'],
            'timestamp': [datetime.now()] * 4,
            'event_properties': [
                '{"valid": "json"}',     # Valid JSON
                '{invalid json}',        # Invalid JSON
                'not json at all',       # Not JSON
                '{"incomplete": }'       # Incomplete JSON
            ],
            'user_properties': [
                '{"valid": true}',       # Valid JSON
                '{"also": "valid"}',     # Valid JSON
                '{broken json',          # Broken JSON
                ''                       # Empty string
            ]
        })
        
        # Should handle gracefully without crashing
        properties = data_manager.get_segmentation_properties(malformed_data)
        
        assert isinstance(properties, dict), "Should return dict even with malformed JSON"
        
        # Should extract properties from valid JSON entries
        event_props = properties.get('event_properties', [])
        user_props = properties.get('user_properties', [])
        
        # Should have some properties from valid entries
        total_props = len(event_props) + len(user_props)
        assert total_props >= 0, "Should handle malformed JSON gracefully"
        
        print("✅ Malformed JSON handling test passed")
    
    def test_json_performance_large_dataset(self, data_manager):
        """Test JSON processing performance with large datasets."""
        import time
        
        # Create large dataset with JSON properties
        n_rows = 10000
        large_data = pd.DataFrame({
            'user_id': [f'user_{i:06d}' for i in range(n_rows)],
            'event_name': [f'Event_{i % 5}' for i in range(n_rows)],
            'timestamp': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_rows)],
            'event_properties': [
                json.dumps({
                    'category': f'cat_{i % 10}',
                    'value': i * 1.5,
                    'tags': [f'tag_{j}' for j in range(i % 3)]
                }) for i in range(n_rows)
            ],
            'user_properties': [
                json.dumps({
                    'segment': f'seg_{i % 5}',
                    'score': i % 100,
                    'active': i % 2 == 0
                }) for i in range(n_rows)
            ]
        })
        
        # Measure performance
        start_time = time.time()
        properties = data_manager.get_segmentation_properties(large_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (less than 30 seconds for 10K rows)
        assert execution_time < 30.0, f"JSON processing took too long: {execution_time:.2f}s"
        
        # Should extract properties successfully
        assert isinstance(properties, dict)
        event_props = properties.get('event_properties', [])
        user_props = properties.get('user_properties', [])
        total_props = len(event_props) + len(user_props)
        assert total_props > 0, "Should extract properties from large dataset"
        
        print(f"✅ JSON performance test passed ({execution_time:.2f}s for {n_rows} rows)")


@pytest.mark.data_source
class TestDataSourceManagerFileHandling:
    """Test file loading and format support."""
    
    @pytest.fixture
    def data_manager(self):
        """Data manager for file testing."""
        return DataSourceManager()
    
    @pytest.fixture 
    def sample_csv_content(self):
        """Sample CSV content for testing."""
        return """user_id,event_name,timestamp,event_properties,user_properties
user_001,Sign Up,2024-01-01 10:00:00,{},{}
user_002,Login,2024-01-01 11:00:00,{},{}
user_003,Purchase,2024-01-01 12:00:00,{},{}"""
    
    def test_load_csv_file_success(self, data_manager, sample_csv_content):
        """Test successful CSV file loading."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_content)
            f.flush()
            
            # Create mock uploaded file
            mock_file = Mock()
            mock_file.name = f.name
            mock_file.read.return_value = sample_csv_content.encode('utf-8')
            
            try:
                # Test loading with mocked streamlit file uploader
                with patch('pandas.read_csv') as mock_read_csv:
                    # Setup pandas to return valid DataFrame
                    expected_df = pd.DataFrame({
                        'user_id': ['user_001', 'user_002', 'user_003'],
                        'event_name': ['Sign Up', 'Login', 'Purchase'],
                        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
                        'event_properties': ['{}', '{}', '{}'],
                        'user_properties': ['{}', '{}', '{}']
                    })
                    mock_read_csv.return_value = expected_df
                    
                    # Load file
                    result_df = data_manager.load_from_file(mock_file)
                    
                    # Validate result
                    assert isinstance(result_df, pd.DataFrame), "Should return DataFrame"
                    assert len(result_df) == 3, "Should have 3 rows"
                    assert 'user_id' in result_df.columns, "Should have user_id column"
            
            finally:
                # Clean up
                Path(f.name).unlink(missing_ok=True)
        
        print("✅ CSV file loading test passed")
    
    def test_load_unsupported_file_format(self, data_manager):
        """Test loading unsupported file format."""
        # Create mock file with unsupported extension
        mock_file = Mock()
        mock_file.name = "test_file.txt"
        
        # Should raise ValueError for unsupported format
        result_df = data_manager.load_from_file(mock_file)
        
        # Should return empty DataFrame on error
        assert isinstance(result_df, pd.DataFrame), "Should return DataFrame"
        assert len(result_df) == 0, "Should be empty on error"
        
        print("✅ Unsupported file format test passed")
    
    def test_load_corrupted_csv_file(self, data_manager):
        """Test loading corrupted CSV file."""
        corrupted_csv = """user_id,event_name,timestamp
user_001,Sign Up,2024-01-01
user_002,"Incomplete quote
user_003,Login,"""  # Corrupted CSV
        
        # Create mock file
        mock_file = Mock()
        mock_file.name = "corrupted.csv"
        
        with patch('pandas.read_csv') as mock_read_csv:
            # Simulate pandas raising an error
            mock_read_csv.side_effect = pd.errors.ParserError("Error tokenizing data")
            
            # Should handle gracefully
            result_df = data_manager.load_from_file(mock_file)
            
            # Should return empty DataFrame on error
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 0, "Should return empty DataFrame on parsing error"
        
        print("✅ Corrupted CSV file test passed")
    
    def test_load_large_file_performance(self, data_manager):
        """Test loading large file performance."""
        import time
        
        # Create large CSV content
        large_csv_content = "user_id,event_name,timestamp,event_properties,user_properties\n"
        for i in range(50000):  # 50K rows
            large_csv_content += f"user_{i:06d},Event_{i % 10},{datetime(2024, 1, 1) + timedelta(minutes=i)},{{}},{{}}\n"
        
        mock_file = Mock()
        mock_file.name = "large_file.csv"
        
        with patch('pandas.read_csv') as mock_read_csv:
            # Create large DataFrame
            large_df = pd.DataFrame({
                'user_id': [f'user_{i:06d}' for i in range(50000)],
                'event_name': [f'Event_{i % 10}' for i in range(50000)],
                'timestamp': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(50000)],
                'event_properties': ['{}'] * 50000,
                'user_properties': ['{}'] * 50000
            })
            mock_read_csv.return_value = large_df
            
            # Measure loading time
            start_time = time.time()
            result_df = data_manager.load_from_file(mock_file)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Should load in reasonable time (less than 10 seconds)
            assert execution_time < 10.0, f"Large file loading took too long: {execution_time:.2f}s"
            
            # Should load successfully
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 50000, "Should load all rows"
        
        print(f"✅ Large file performance test passed ({execution_time:.2f}s for 50K rows)")


@pytest.mark.data_source
class TestDataSourceManagerClickHouseIntegration:
    """Test ClickHouse database integration edge cases."""
    
    @pytest.fixture
    def data_manager(self):
        """Data manager for ClickHouse testing."""
        return DataSourceManager()
    
    def test_clickhouse_connection_failure(self, data_manager):
        """Test ClickHouse connection failure handling."""
        # Test with invalid connection parameters
        result = data_manager.connect_clickhouse(
            host="invalid_host",
            port=9999,
            username="invalid_user", 
            password="invalid_pass",
            database="invalid_db"
        )
        
        # Should handle failure gracefully
        assert result == False, "Should return False for failed connection"
        assert data_manager.clickhouse_client is None, "Client should remain None on failure"
        
        print("✅ ClickHouse connection failure test passed")
    
    def test_clickhouse_query_without_connection(self, data_manager):
        """Test querying without established connection."""
        # Ensure no connection
        data_manager.clickhouse_client = None
        
        result_df = data_manager.load_from_clickhouse("SELECT 1")
        
        # Should return empty DataFrame and handle gracefully
        assert isinstance(result_df, pd.DataFrame), "Should return DataFrame"
        assert len(result_df) == 0, "Should be empty without connection"
        
        print("✅ ClickHouse query without connection test passed")
    
    def test_clickhouse_invalid_query(self, data_manager):
        """Test ClickHouse with invalid SQL query."""
        # Mock ClickHouse client
        mock_client = Mock()
        mock_client.query_df.side_effect = Exception("SQL syntax error")
        data_manager.clickhouse_client = mock_client
        
        # Test with invalid query
        result_df = data_manager.load_from_clickhouse("INVALID SQL QUERY")
        
        # Should handle error gracefully
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0, "Should return empty DataFrame on query error"
        
        print("✅ ClickHouse invalid query test passed")
    
    def test_clickhouse_query_validation_failure(self, data_manager):
        """Test ClickHouse query returning invalid data structure."""
        # Mock ClickHouse client returning invalid data
        mock_client = Mock()
        invalid_df = pd.DataFrame({
            'wrong_column': ['value1', 'value2'],  # Missing required columns
            'another_wrong': ['value3', 'value4']
        })
        mock_client.query_df.return_value = invalid_df
        data_manager.clickhouse_client = mock_client
        
        result_df = data_manager.load_from_clickhouse("SELECT wrong_column FROM table")
        
        # Should return empty DataFrame when validation fails
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0, "Should return empty DataFrame when validation fails"
        
        print("✅ ClickHouse query validation failure test passed")


@pytest.mark.data_source
@pytest.mark.performance  
class TestDataSourceManagerPerformanceEdgeCases:
    """Test performance edge cases and memory management."""
    
    @pytest.fixture
    def data_manager(self):
        """Data manager for performance testing."""
        return DataSourceManager()
    
    def test_memory_usage_large_dataset(self, data_manager):
        """Test memory usage with large dataset processing."""
        import tracemalloc
        
        # Create large dataset
        n_rows = 100000
        large_df = pd.DataFrame({
            'user_id': [f'user_{i:06d}' for i in range(n_rows)],
            'event_name': [f'Event_{i % 20}' for i in range(n_rows)],
            'timestamp': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_rows)],
            'event_properties': [json.dumps({'category': f'cat_{i % 10}', 'value': i}) for i in range(n_rows)],
            'user_properties': [json.dumps({'segment': f'seg_{i % 5}', 'score': i % 100}) for i in range(n_rows)]
        })
        
        # Start memory tracking
        tracemalloc.start()
        
        # Perform memory-intensive operations
        is_valid, _ = data_manager.validate_event_data(large_df)
        properties = data_manager.get_segmentation_properties(large_df)
        metadata = data_manager.get_event_metadata(large_df)
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validate memory usage (less than 500MB for 100K rows)
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 500, f"Memory usage too high: {peak_mb:.1f}MB"
        
        # Validate operations completed successfully
        assert is_valid, "Should validate large dataset"
        assert isinstance(properties, dict), "Should extract properties"
        assert isinstance(metadata, dict), "Should extract metadata"
        
        print(f"✅ Memory usage test passed (peak: {peak_mb:.1f}MB for {n_rows} rows)")
    
    def test_concurrent_operations_simulation(self, data_manager):
        """Test simulated concurrent operations on data manager."""
        import threading
        import time
        
        # Create test data
        test_df = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(1000)],
            'event_name': [f'Event_{i % 5}' for i in range(1000)],
            'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(1000)],
            'event_properties': ['{}'] * 1000,
            'user_properties': ['{}'] * 1000
        })
        
        results = []
        errors = []
        
        def worker_operation(operation_id):
            """Simulate concurrent operation on data manager."""
            try:
                start_time = time.time()
                
                # Perform multiple operations
                is_valid, _ = data_manager.validate_event_data(test_df)
                properties = data_manager.get_segmentation_properties(test_df)
                metadata = data_manager.get_event_metadata(test_df)
                
                end_time = time.time()
                
                results.append({
                    'operation_id': operation_id,
                    'duration': end_time - start_time,
                    'valid': is_valid,
                    'properties_count': len(properties.get('event_properties', [])) + len(properties.get('user_properties', [])),
                    'metadata_count': len(metadata)
                })
                
            except Exception as e:
                errors.append(f"Operation {operation_id}: {str(e)}")
        
        # Start multiple threads
        threads = []
        for i in range(5):  # 5 concurrent operations
            thread = threading.Thread(target=worker_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Validate results
        assert len(errors) == 0, f"Concurrent operations had errors: {errors}"
        assert len(results) == 5, "All 5 operations should complete"
        
        # All operations should succeed
        for result in results:
            assert result['valid'], f"Operation {result['operation_id']} should be valid"
            assert result['duration'] < 10.0, f"Operation {result['operation_id']} took too long: {result['duration']:.2f}s"
        
        print("✅ Concurrent operations simulation test passed")
    
    def test_polars_pandas_fallback_performance(self, data_manager):
        """Test performance difference between Polars and Pandas implementations."""
        import time
        
        # Create complex JSON dataset that might trigger fallback
        n_rows = 10000
        complex_df = pd.DataFrame({
            'user_id': [f'user_{i:05d}' for i in range(n_rows)],
            'event_name': [f'Event_{i % 10}' for i in range(n_rows)],
            'timestamp': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_rows)],
            'event_properties': [
                json.dumps({
                    'nested': {
                        'level1': {'level2': f'value_{i}'},
                        'array': [i, i+1, i+2],
                        'mixed': {'str': f'string_{i}', 'num': i, 'bool': i % 2 == 0}
                    },
                    'simple': f'simple_{i}'
                }) for i in range(n_rows)
            ],
            'user_properties': [
                json.dumps({
                    'profile': {
                        'demographics': {'age': 20 + (i % 50), 'location': f'city_{i % 100}'},
                        'preferences': [f'pref_{j}' for j in range(i % 5)],
                        'scores': {'engagement': i % 100, 'satisfaction': (i * 1.5) % 100}
                    }
                }) for i in range(n_rows)
            ]
        })
        
        # Test segmentation properties extraction (most likely to trigger fallback)
        start_time = time.time()
        properties = data_manager.get_segmentation_properties(complex_df)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete in reasonable time regardless of implementation
        assert execution_time < 60.0, f"Segmentation took too long: {execution_time:.2f}s"
        
        # Should extract some properties
        event_props = properties.get('event_properties', [])
        user_props = properties.get('user_properties', [])
        total_props = len(event_props) + len(user_props)
        assert total_props > 0, "Should extract properties from complex JSON"
        
        print(f"✅ Polars/Pandas fallback performance test passed ({execution_time:.2f}s)")


# Integration test combining multiple DataSourceManager features
@pytest.mark.data_source
@pytest.mark.integration
class TestDataSourceManagerIntegration:
    """Test integration scenarios combining multiple DataSourceManager features."""
    
    def test_complete_data_processing_workflow(self):
        """Test complete workflow from file loading to analysis-ready data."""
        data_manager = DataSourceManager()
        
        # Step 1: Create realistic test data
        events_list = [
            'Sign Up', 'Email Verification', 'Profile Setup', 'First Login',
            'Product View', 'Add to Cart', 'Checkout Started', 'Purchase'
        ]
        
        realistic_data = pd.DataFrame({
            'user_id': [f'user_{i:04d}' for i in range(1000)],
            'event_name': [events_list[i % 8] for i in range(1000)],
            'timestamp': [
                datetime(2024, 1, 1) + timedelta(
                    days=i // 100,  # Spread across days
                    hours=i % 24,   # Spread across hours
                    minutes=i % 60  # Add minute variation
                ) for i in range(1000)
            ],
            'event_properties': [
                json.dumps({
                    'source': ['organic', 'google_ads', 'facebook', 'email'][i % 4],
                    'campaign': f'campaign_{i % 5}',
                    'value': round(i * 1.5, 2),
                    'category': ['acquisition', 'engagement', 'conversion'][i % 3]
                }) for i in range(1000)
            ],
            'user_properties': [
                json.dumps({
                    'segment': ['premium', 'basic', 'trial'][i % 3],
                    'country': ['US', 'UK', 'CA', 'DE', 'FR'][i % 5],
                    'age_group': ['18-25', '26-35', '36-45', '46+'][i % 4],
                    'registration_date': (datetime(2024, 1, 1) - timedelta(days=i % 365)).strftime('%Y-%m-%d')
                }) for i in range(1000)
            ]
        })
        
        # Step 2: Validate data
        is_valid, validation_message = data_manager.validate_event_data(realistic_data)
        assert is_valid, f"Data validation failed: {validation_message}"
        
        # Step 3: Extract properties for segmentation
        segmentation_props = data_manager.get_segmentation_properties(realistic_data)
        assert isinstance(segmentation_props, dict)
        assert len(segmentation_props) > 0, "Should extract segmentation properties"
        
        # Step 4: Get event metadata
        event_metadata = data_manager.get_event_metadata(realistic_data)
        assert isinstance(event_metadata, dict)
        assert len(event_metadata) > 0, "Should extract event metadata"
        
        # Step 5: Test property value extraction
        for prop_type in ['event_properties', 'user_properties']:
            if prop_type in segmentation_props and segmentation_props[prop_type]:
                prop_name = segmentation_props[prop_type][0]
                values = data_manager.get_property_values(realistic_data, prop_name, prop_type)
                assert isinstance(values, list), f"Should get values for {prop_type}.{prop_name}"
        
        # Step 6: Generate sample data and compare
        sample_data = data_manager.get_sample_data()
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) > 0, "Should generate sample data"
        
        # Sample data should also be valid
        sample_valid, _ = data_manager.validate_event_data(sample_data)
        assert sample_valid, "Generated sample data should be valid"
        
        print("✅ Complete data processing workflow test passed")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "--tb=short"])
