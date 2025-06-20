"""
Test Data Generator - создает тестовые данные на лету для CI/CD
Заменяет необходимость хранить большие CSV файлы в Git
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path


class TestDataGenerator:
    """Генератор тестовых данных для фуннель-анализа"""
    
    def __init__(self, seed=42):
        """Инициализация с фиксированным seed для воспроизводимости"""
        np.random.seed(seed)
        self.seed = seed
    
    def generate_demo_events(self):
        """Генерирует демо события"""
        demo_events = [
            {
                "name": "Page View",
                "category": "Navigation",
                "description": "User views a page",
                "frequency": "high",
            },
            {
                "name": "User Sign-Up",
                "category": "Conversion",
                "description": "User creates an account",
                "frequency": "medium",
            },
            {
                "name": "Email Verification",
                "category": "Conversion",
                "description": "User verifies email address",
                "frequency": "medium",
            },
            {
                "name": "First Login",
                "category": "Engagement",
                "description": "User logs in for the first time",
                "frequency": "medium",
            },
            {
                "name": "Profile Setup",
                "category": "Engagement",
                "description": "User completes profile setup",
                "frequency": "medium",
            },
            {
                "name": "First Purchase",
                "category": "Revenue",
                "description": "User makes their first purchase",
                "frequency": "low",
            },
            {
                "name": "Add to Cart",
                "category": "Commerce",
                "description": "User adds item to shopping cart",
                "frequency": "medium",
            },
            {
                "name": "Checkout Started",
                "category": "Commerce",
                "description": "User begins checkout process",
                "frequency": "medium",
            },
            {
                "name": "Payment Completed",
                "category": "Revenue",
                "description": "User completes payment",
                "frequency": "low",
            },
        ]
        return pd.DataFrame(demo_events)
    
    def generate_sample_funnel(self):
        """Генерирует образец воронки"""
        sample_funnel_data = {
            "User Sign-Up": {"users": 10000, "conversion_rate": 100},
            "Verify Email": {"users": 7500, "conversion_rate": 75},
            "First Login": {"users": 6000, "conversion_rate": 60},
            "Profile Setup": {"users": 4500, "conversion_rate": 45},
            "Tutorial Completed": {"users": 3600, "conversion_rate": 36},
        }
        
        return pd.DataFrame({
            "step": list(sample_funnel_data.keys()),
            "users": [data["users"] for data in sample_funnel_data.values()],
            "conversion_rate": [data["conversion_rate"] for data in sample_funnel_data.values()],
        })
    
    def generate_time_series_data(self, days=30):
        """Генерирует временные ряды данных"""
        dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
        steps = [
            "User Sign-Up",
            "Verify Email", 
            "First Login",
            "Profile Setup",
            "Tutorial Completed",
        ]
        
        time_series_data = {}
        for i, step in enumerate(steps):
            base_value = 1000 * (0.8**i)
            values = np.random.normal(base_value, base_value * 0.1, size=len(dates))
            values = np.maximum(values + np.linspace(0, base_value * 0.2, len(dates)), 0)
            time_series_data[step] = values.round().astype(int)
        
        return pd.DataFrame(time_series_data, index=dates)
    
    def generate_segment_data(self):
        """Генерирует данные по сегментам"""
        segments = ["Mobile", "Desktop", "Tablet"]
        steps = ["User Sign-Up", "Verify Email", "First Login", "Profile Setup", "Tutorial Completed"]
        segment_data = {}
        
        for segment in segments:
            segment_data[segment] = {}
            base_value = 1000 if segment == "Mobile" else (800 if segment == "Desktop" else 400)
            for i, step in enumerate(steps):
                drop_rate = 0.75 if segment == "Mobile" else (0.85 if segment == "Desktop" else 0.70)
                segment_data[segment][step] = int(base_value * (drop_rate**i))
        
        df = pd.DataFrame(segment_data)
        df.index = steps
        return df
    
    def generate_time_to_convert_stats(self):
        """Генерирует статистику времени до конверсии"""
        time_to_convert = {
            "Verify Email": np.random.exponential(scale=0.5, size=1000) * 60,
            "First Login": np.random.exponential(scale=12, size=1000) * 60,
            "Profile Setup": np.random.exponential(scale=24, size=1000) * 60,
            "First Purchase": np.random.exponential(scale=72, size=1000) * 60,
        }
        
        df = pd.DataFrame(time_to_convert)
        return df.describe()
    
    def generate_ab_test_data(self):
        """Генерирует данные A/B тестов"""
        ab_test_data = {
            "Variant A": {
                "User Sign-Up": 5000,
                "Verify Email": 4000,
                "First Login": 3400,
                "Profile Setup": 2800,
                "First Purchase": 2100,
            },
            "Variant B": {
                "User Sign-Up": 5000,
                "Verify Email": 4200,
                "First Login": 3650,
                "Profile Setup": 3100,
                "First Purchase": 2400,
            },
        }
        return pd.DataFrame(ab_test_data)
    
    def generate_ab_test_rates(self):
        """Генерирует коэффициенты конверсии A/B тестов"""
        df_ab_test = self.generate_ab_test_data()
        return df_ab_test.div(df_ab_test.iloc[0]) * 100
    
    def generate_large_dataset(self, size=50000):
        """Генерирует большой датасет для тестов производительности"""
        np.random.seed(self.seed)
        
        # Генерируем пользователей
        user_ids = [f"user_{i:06d}" for i in range(1, size + 1)]
        
        # События воронки
        events = ["Page View", "Sign Up", "Email Verified", "First Login", "Purchase"]
        
        data = []
        for user_id in user_ids:
            # Каждый пользователь проходит через воронку с определенной вероятностью
            base_time = datetime.now() - timedelta(days=np.random.randint(1, 365))
            
            for i, event in enumerate(events):
                # Вероятность прохождения каждого шага уменьшается
                if np.random.random() < (0.8 ** i):
                    timestamp = base_time + timedelta(
                        hours=np.random.randint(0, 24 * (i + 1))
                    )
                    data.append({
                        "user_id": user_id,
                        "event": event,
                        "timestamp": timestamp,
                        "platform": np.random.choice(["mobile", "desktop", "tablet"], p=[0.6, 0.3, 0.1]),
                        "country": np.random.choice(["US", "UK", "CA", "AU", "DE"]),
                        "user_type": np.random.choice(["new", "returning", "premium"], p=[0.5, 0.4, 0.1])
                    })
                else:
                    break  # Пользователь выбыл из воронки
        
        return pd.DataFrame(data)
    
    def ensure_test_data_exists(self, force_regenerate=False):
        """
        Убеждается, что тестовые данные существуют
        Генерирует их если нужно (например, в CI)
        """
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        files_to_generate = {
            "demo_events.csv": self.generate_demo_events,
            "sample_funnel.csv": self.generate_sample_funnel,
            "time_series_data.csv": self.generate_time_series_data,
            "segment_data.csv": self.generate_segment_data,
            "time_to_convert_stats.csv": self.generate_time_to_convert_stats,
            "ab_test_data.csv": self.generate_ab_test_data,
            "ab_test_rates.csv": self.generate_ab_test_rates,
        }
        
        for filename, generator_func in files_to_generate.items():
            filepath = test_data_dir / filename
            
            if force_regenerate or not filepath.exists():
                print(f"🔄 Generating {filename}...")
                df = generator_func()
                df.to_csv(filepath, index=True if "time_series" in filename else False)
                print(f"✅ Generated {filename}")
        
        # Генерируем большие файлы только если их нет
        large_files = {
            "test_50k.csv": lambda: self.generate_large_dataset(50000),
            "test_200k.csv": lambda: self.generate_large_dataset(200000),
        }
        
        for filename, generator_func in large_files.items():
            filepath = test_data_dir / filename
            if force_regenerate or not filepath.exists():
                print(f"🔄 Generating large dataset {filename}...")
                df = generator_func()
                df.to_csv(filepath, index=False)
                print(f"✅ Generated {filename}")


# Глобальная функция для использования в тестах
def ensure_test_data():
    """Убеждается, что тестовые данные доступны"""
    generator = TestDataGenerator()
    generator.ensure_test_data_exists()


# Функция для быстрого создания временных тестовых данных
def create_temp_test_data():
    """Создает временные тестовые данные для тестов"""
    generator = TestDataGenerator()
    temp_dir = tempfile.mkdtemp()
    
    # Создаем основные файлы
    files = {
        "demo_events.csv": generator.generate_demo_events(),
        "sample_funnel.csv": generator.generate_sample_funnel(),
        "time_series_data.csv": generator.generate_time_series_data(),
    }
    
    for filename, df in files.items():
        filepath = os.path.join(temp_dir, filename)
        df.to_csv(filepath, index=False)
    
    return temp_dir


if __name__ == "__main__":
    # Запуск генератора напрямую
    print("🔄 Generating test data...")
    generator = TestDataGenerator()
    generator.ensure_test_data_exists(force_regenerate=True)
    print("✅ All test data generated successfully!") 