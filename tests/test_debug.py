#!/usr/bin/env python3
"""
Простой тест для проверки передачи событий в расчетное ядро
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Добавляем текущую директорию в путь
sys.path.append('.')

from app import FunnelCalculator, FunnelConfig, CountingMethod

def create_test_data():
    """Создает тестовые данные для проверки"""
    base_time = datetime(2024, 1, 1)
    
    events_data = []
    for user_id in range(1, 101):  # 100 пользователей
        user_id_str = f"user_{user_id:03d}"
        
        # Пользователь делает первое событие
        events_data.append({
            'user_id': user_id_str,
            'event_name': 'User Sign-Up',
            'timestamp': base_time + timedelta(minutes=user_id)
        })
        
        # 80% пользователей делают второе событие
        if user_id <= 80:
            events_data.append({
                'user_id': user_id_str,
                'event_name': 'Verify Email',
                'timestamp': base_time + timedelta(minutes=user_id + 10)
            })
        
        # 60% пользователей делают третье событие
        if user_id <= 60:
            events_data.append({
                'user_id': user_id_str,
                'event_name': 'First Login',
                'timestamp': base_time + timedelta(minutes=user_id + 20)
            })
    
    df = pd.DataFrame(events_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def test_funnel_calculation():
    """Тестируем расчет воронки"""
    print("🧪 Запускаем тест расчета воронки...")
    
    # Создаем тестовые данные
    events_df = create_test_data()
    print(f"📊 Создали {len(events_df)} событий для {events_df['user_id'].nunique()} пользователей")
    
    # Создаем конфигурацию
    config = FunnelConfig(
        conversion_window_hours=24,
        counting_method=CountingMethod.UNIQUE_USERS
    )
    print(f"⚙️ Конфигурация: {config.to_dict()}")
    
    # Список шагов воронки
    funnel_steps = ['User Sign-Up', 'Verify Email', 'First Login']
    print(f"🎯 Шаги воронки: {funnel_steps}")
    
    # Проверяем, что все события есть в данных
    available_events = set(events_df['event_name'].unique())
    missing_events = [step for step in funnel_steps if step not in available_events]
    if missing_events:
        print(f"❌ События не найдены в данных: {missing_events}")
        return False
    else:
        print(f"✅ Все события найдены в данных: {available_events}")
    
    # Создаем калькулятор и запускаем расчет
    calculator = FunnelCalculator(config)
    
    try:
        results = calculator.calculate_funnel_metrics(events_df, funnel_steps)
        
        print("\n📈 Результаты расчета:")
        print(f"  Steps: {results.steps}")
        print(f"  Users count: {results.users_count}")
        print(f"  Conversion rates: {results.conversion_rates}")
        print(f"  Drop offs: {results.drop_offs}")
        print(f"  Drop off rates: {results.drop_off_rates}")
        
        # Проверяем ожидаемые результаты
        expected_users = [100, 80, 60]  # Ожидаемое количество пользователей на каждом шаге
        
        if results.users_count == expected_users:
            print("✅ Результаты соответствуют ожиданиям!")
            return True
        else:
            print(f"❌ Результаты НЕ соответствуют ожиданиям. Ожидалось: {expected_users}, получено: {results.users_count}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при расчете: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_funnel_calculation()
    if success:
        print("\n🎉 Тест прошел успешно! Расчетное ядро работает корректно.")
    else:
        print("\n💥 Тест провален! Есть проблема с расчетным ядром.")
    
    sys.exit(0 if success else 1) 