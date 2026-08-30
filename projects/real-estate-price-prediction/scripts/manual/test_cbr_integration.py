"""
Тест интеграции CBR API с economic_features
"""

import sys

sys.path.append(".")


import pandas as pd

from src.preprocessing.economic_features import EconomicFeatureEngineer

# Создать тестовый датасет
dates = pd.date_range(start="2024-01-01", periods=10, freq="30D")
test_df = pd.DataFrame({"date_published": dates, "city": ["Москва"] * 10, "price": [10000000] * 10})

print("=" * 70)
print("TEST: CBR API Integration")
print("=" * 70)

# Тест 1: С API
print("\n[1] Testing with CBR API...")
eco_fe_api = EconomicFeatureEngineer(use_api=True)
df_with_api = eco_fe_api.add_economic_features(test_df)

print(f"\nFeatures added: {list(df_with_api.columns)}")
print("\nSample data (with API):")
print(df_with_api[["date_published", "key_rate", "usd_rate", "inflation_rate"]].head())

# Тест 2: Без API (fallback)
print("\n[2] Testing with fallback data...")
eco_fe_fallback = EconomicFeatureEngineer(use_api=False)
df_fallback = eco_fe_fallback.add_economic_features(test_df)

print("\nSample data (fallback):")
print(df_fallback[["date_published", "key_rate", "usd_rate", "inflation_rate"]].head())

# Сравнение
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(
    f"API vs Fallback - USD rates match: {df_with_api['usd_rate'].equals(df_fallback['usd_rate'])}"
)
print(f"\nAPI USD rates (unique): {sorted(df_with_api['usd_rate'].unique())}")
print(f"Fallback USD rates (unique): {sorted(df_fallback['usd_rate'].unique())}")

print("\n✅ Test completed!")
