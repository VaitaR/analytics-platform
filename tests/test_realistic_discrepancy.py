"""
Test to reproduce the actual user-reported discrepancy with realistic data patterns.

This test aims to identify the specific scenario where hourly timeseries conversion rates
are dramatically higher than overall funnel conversion rates.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from app import FunnelCalculator, FunnelConfig, CountingMethod, FunnelOrder, ReentryMode


class TestRealWorldDiscrepancy:
    """Test to reproduce the user-reported conversion rate issue."""
    
    @pytest.fixture
    def problematic_data(self):
        """Create data that could cause the discrepancy issue."""
        data = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Scenario: Many users start the funnel across multiple days,
        # but very few complete it overall (0.9% completion rate)
        # However, in specific hours, completion rates appear much higher
        
        # Day 1: 100 users start, 1 completes
        for i in range(100):
            # Users start throughout the day
            start_time = base_time + timedelta(hours=i//10, minutes=i%10*6)
            data.append({
                'user_id': f'user_day1_{i}',
                'event_name': 'step1',
                'timestamp': start_time,
                'event_properties': '{}'
            })
            
            # Only user 50 completes step2, but much later (outside most hourly windows)
            if i == 50:
                data.append({
                    'user_id': f'user_day1_{i}',
                    'event_name': 'step2',
                    'timestamp': start_time + timedelta(hours=25),  # Next day
                    'event_properties': '{}'
                })
        
        # Day 2: 50 users start, none complete
        for i in range(50):
            start_time = base_time + timedelta(days=1, hours=i//10, minutes=i%10*6)
            data.append({
                'user_id': f'user_day2_{i}',
                'event_name': 'step1',
                'timestamp': start_time,
                'event_properties': '{}'
            })
        
        # Day 3: A specific hour with high completion rate
        # 10 users start in hour 14:00-15:00, 8 complete within that hour
        for i in range(10):
            start_time = base_time + timedelta(days=2, hours=14, minutes=i*5)
            data.append({
                'user_id': f'user_day3_hour14_{i}',
                'event_name': 'step1', 
                'timestamp': start_time,
                'event_properties': '{}'
            })
            
            # 8 out of 10 complete within the same hour
            if i < 8:
                data.append({
                    'user_id': f'user_day3_hour14_{i}',
                    'event_name': 'step2',
                    'timestamp': start_time + timedelta(minutes=30),
                    'event_properties': '{}'
                })
        
        # Add many more users with no completions to dilute overall rate
        for day in range(4, 10):  # Days 4-9
            for i in range(200):  # 200 users per day
                start_time = base_time + timedelta(days=day, hours=i//20, minutes=i%20*3)
                data.append({
                    'user_id': f'user_day{day}_{i}',
                    'event_name': 'step1',
                    'timestamp': start_time,
                    'event_properties': '{}'
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture 
    def calculator(self):
        """Create calculator with standard configuration."""
        config = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            conversion_window_hours=48  # 48-hour window
        )
        return FunnelCalculator(config)
    
    def test_realistic_discrepancy_scenario(self, problematic_data, calculator):
        """
        Test a realistic scenario where timeseries and overall metrics diverge.
        
        Expected issue:
        - Overall: ~1500 users start, ~9 complete → ~0.6% conversion
        - Some hourly periods show 80% conversion (e.g., day 3 hour 14)
        """
        steps = ['step1', 'step2']
        
        print("\n=== REALISTIC DISCREPANCY TEST ===")
        print(f"Dataset summary:")
        print(f"- Total events: {len(problematic_data)}")
        print(f"- Unique users: {problematic_data['user_id'].nunique()}")
        print(f"- Time range: {problematic_data['timestamp'].min()} to {problematic_data['timestamp'].max()}")
        
        step1_users = problematic_data[problematic_data['event_name'] == 'step1']['user_id'].nunique()
        step2_users = problematic_data[problematic_data['event_name'] == 'step2']['user_id'].nunique()
        print(f"- Users who did step1: {step1_users}")
        print(f"- Users who did step2: {step2_users}")
        print(f"- Raw step2/step1 ratio: {(step2_users/step1_users*100):.2f}%")
        
        # 1. Calculate overall funnel metrics
        overall_results = calculator.calculate_funnel_metrics(problematic_data, steps)
        
        print(f"\n1. OVERALL FUNNEL RESULTS:")
        print(f"   Steps: {overall_results.steps}")
        print(f"   User counts: {overall_results.users_count}")
        print(f"   Conversion rates: {overall_results.conversion_rates}")
        
        overall_final_conversion = overall_results.conversion_rates[-1] if overall_results.conversion_rates else 0
        
        # 2. Calculate hourly timeseries metrics
        timeseries_results = calculator.calculate_timeseries_metrics(problematic_data, steps, '1h')
        
        print(f"\n2. HOURLY TIMESERIES RESULTS:")
        print(f"   Shape: {timeseries_results.shape}")
        
        if not timeseries_results.empty:
            # Find the highest conversion rate periods
            max_conversion_idx = timeseries_results['conversion_rate'].idxmax()
            max_conversion_row = timeseries_results.loc[max_conversion_idx]
            
            print(f"\n   HIGHEST CONVERSION PERIOD:")
            print(f"   Period: {max_conversion_row['period_date']}")
            print(f"   Started: {max_conversion_row['started_funnel_users']}")
            print(f"   Completed: {max_conversion_row['completed_funnel_users']}")
            print(f"   Conversion rate: {max_conversion_row['conversion_rate']:.2f}%")
            
            # Show a few more high-conversion periods
            top_periods = timeseries_results.nlargest(3, 'conversion_rate')
            print(f"\n   TOP 3 CONVERSION PERIODS:")
            for idx, row in top_periods.iterrows():
                print(f"     {row['period_date']}: {row['conversion_rate']:.2f}% "
                      f"({row['completed_funnel_users']}/{row['started_funnel_users']})")
            
            # Calculate aggregate from timeseries
            total_started = timeseries_results['started_funnel_users'].sum()
            total_completed = timeseries_results['completed_funnel_users'].sum()
            aggregate_conversion = (total_completed / total_started * 100) if total_started > 0 else 0
            
            print(f"\n3. AGGREGATE FROM TIMESERIES:")
            print(f"   Total started: {total_started}")
            print(f"   Total completed: {total_completed}")
            print(f"   Aggregate conversion rate: {aggregate_conversion:.2f}%")
            
            # 4. Compare and identify the issue
            print(f"\n4. COMPARISON:")
            print(f"   Overall funnel conversion: {overall_final_conversion:.2f}%")
            print(f"   Timeseries aggregate conversion: {aggregate_conversion:.2f}%")
            print(f"   Difference: {abs(overall_final_conversion - aggregate_conversion):.2f}%")
            print(f"   Max hourly conversion: {max_conversion_row['conversion_rate']:.2f}%")
            
            # This is the key insight: 
            print(f"\n5. ROOT CAUSE ANALYSIS:")
            conversion_diff = abs(overall_final_conversion - aggregate_conversion)
            if conversion_diff > 5:
                print(f"   ⚠️  DISCREPANCY DETECTED: {conversion_diff:.2f}% difference")
                print("   This suggests the methodologies are fundamentally different:")
                print("   - Overall funnel: Counts all users who ever did step1, checks if they later did step2")
                print("   - Timeseries: Uses COHORT analysis - only counts users who STARTED in each period")
                print("   - Users who start in one period but convert in another period may be counted differently")
            else:
                print("   ✓ Methodologies are consistent")
            
            # Check if we have periods with very high conversion rates that could confuse users
            high_conversion_periods = timeseries_results[timeseries_results['conversion_rate'] > 50]
            if not high_conversion_periods.empty:
                print(f"\n   ⚠️  HIGH CONVERSION PERIODS DETECTED ({len(high_conversion_periods)} periods > 50%)")
                print("   These periods could mislead users about overall performance:")
                for idx, row in high_conversion_periods.iterrows():
                    print(f"     {row['period_date']}: {row['conversion_rate']:.1f}% "
                          f"({row['completed_funnel_users']}/{row['started_funnel_users']} users)")
    
    def test_cross_period_conversion_issue(self, problematic_data, calculator):
        """
        Test the specific issue where users start in one period but convert in another.
        This is likely the root cause of the discrepancy.
        """
        steps = ['step1', 'step2']
        
        print("\n=== CROSS-PERIOD CONVERSION ANALYSIS ===")
        
        # Analyze specific users to understand the conversion timing
        step1_events = problematic_data[problematic_data['event_name'] == 'step1']
        step2_events = problematic_data[problematic_data['event_name'] == 'step2']
        
        print(f"Users with conversions:")
        converting_users = step2_events['user_id'].unique()
        
        for user_id in converting_users[:5]:  # Just first 5
            user_step1 = step1_events[step1_events['user_id'] == user_id].iloc[0]
            user_step2 = step2_events[step2_events['user_id'] == user_id].iloc[0]
            
            step1_hour = user_step1['timestamp'].floor('H')
            step2_hour = user_step2['timestamp'].floor('H')
            
            time_diff = user_step2['timestamp'] - user_step1['timestamp']
            
            print(f"  {user_id}:")
            print(f"    step1: {user_step1['timestamp']} (hour: {step1_hour})")
            print(f"    step2: {user_step2['timestamp']} (hour: {step2_hour})")
            print(f"    time_diff: {time_diff}")
            print(f"    same_hour: {step1_hour == step2_hour}")
    
    def test_cohort_vs_overall_explanation(self, problematic_data, calculator):
        """
        Clearly demonstrate the difference between cohort-based and overall metrics.
        """
        steps = ['step1', 'step2']
        
        print("\n=== COHORT VS OVERALL METHODOLOGY COMPARISON ===")
        
        # Manual calculation of overall metrics (all users who ever did step1 vs step2)
        all_step1_users = set(problematic_data[problematic_data['event_name'] == 'step1']['user_id'])
        all_step2_users = set(problematic_data[problematic_data['event_name'] == 'step2']['user_id'])
        
        # Check which step2 users also did step1 (proper funnel completion)
        step2_users_who_did_step1 = all_step2_users.intersection(all_step1_users)
        
        manual_overall_conversion = len(step2_users_who_did_step1) / len(all_step1_users) * 100
        
        print(f"MANUAL OVERALL CALCULATION:")
        print(f"  All step1 users: {len(all_step1_users)}")
        print(f"  All step2 users: {len(all_step2_users)}")
        print(f"  Step2 users who also did step1: {len(step2_users_who_did_step1)}")
        print(f"  Manual overall conversion: {manual_overall_conversion:.2f}%")
        
        # Now show what cohort analysis does differently
        print(f"\nCOHORT ANALYSIS APPROACH:")
        print("  For each hour:")
        print("  1. Find users who STARTED the funnel in this hour (did step1)")
        print("  2. Check if those specific users completed step2 within conversion window")
        print("  3. Calculate conversion rate for THAT COHORT only")
        print("  4. Users who started in different hours are separate cohorts")
        
        # Calculate system results for comparison
        overall_results = calculator.calculate_funnel_metrics(problematic_data, steps)
        overall_system_conversion = overall_results.conversion_rates[-1] if overall_results.conversion_rates else 0
        
        print(f"\nSYSTEM OVERALL CALCULATION: {overall_system_conversion:.2f}%")
        print(f"MANUAL CALCULATION: {manual_overall_conversion:.2f}%")
        print(f"Difference: {abs(overall_system_conversion - manual_overall_conversion):.2f}%")
        
        if abs(overall_system_conversion - manual_overall_conversion) > 1:
            print("⚠️  The system is using additional logic beyond simple set intersection")
            print("   (likely conversion window enforcement or ordering requirements)")
