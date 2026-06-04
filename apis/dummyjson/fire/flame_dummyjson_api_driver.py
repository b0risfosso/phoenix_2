#!/usr/bin/env python3
"""
DummyJSON + Simple Flame Combustion API Driver
----------------------------------------------

This script lets the DummyJSON API interact with:
    simple_flame_combustion_simulation.py

Place this file in the same folder as:
    simple_flame_combustion_simulation.py

Run:
    python flame_dummyjson_api_driver.py

What it does:
- Downloads product records from DummyJSON.
- Converts each product into FlameParams.
- Runs the original simple flame run_round() function with API-selected settings.
- Uses product price, rating, discount, stock, category, brand, and title length
  as external experiment controls.
- Falls back to local product-like records if the network is unavailable.

No external packages are required.
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from typing import Any, Dict, List

import simple_flame_combustion_simulation as flame


BASE_URL = "https://dummyjson.com"


# -----------------------------------------------------------------------------
# API ACCESS
# -----------------------------------------------------------------------------

def request_json(endpoint: str, timeout_s: float = 10.0) -> Any:
    """Request JSON from DummyJSON using only the Python standard library."""
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    url = BASE_URL + endpoint
    request = urllib.request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": "dummyjson-flame-driver/1.0",
        },
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def get_api_products(limit: int) -> List[Dict[str, Any]]:
    """Download products from DummyJSON, or use fallback product-like items."""
    try:
        query = urllib.parse.urlencode({
            "limit": limit,
            "select": "title,description,category,brand,price,discountPercentage,rating,stock",
        })
        data = request_json(f"/products?{query}")
        products = data.get("products", []) if isinstance(data, dict) else []
        if products:
            return products[:limit]
        raise ValueError("DummyJSON returned no products.")

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as error:
        print("\nAPI request failed. Using local fallback product items.")
        print(f"Reason: {error}")

        return [
            {"id": 1, "title": "Fallback Gas Burner", "category": "kitchen-accessories", "brand": "Local", "price": 60, "discountPercentage": 4.5, "rating": 4.5, "stock": 75},
            {"id": 2, "title": "Fallback Fuel Rich Resin", "category": "fragrances", "brand": "Local", "price": 35, "discountPercentage": 12.0, "rating": 3.7, "stock": 18},
            {"id": 3, "title": "Fallback Oxygen Assisted Torch", "category": "automotive", "brand": "Local", "price": 120, "discountPercentage": 2.0, "rating": 4.8, "stock": 90},
            {"id": 4, "title": "Fallback Humid Weak Igniter", "category": "home-decoration", "brand": "Local", "price": 22, "discountPercentage": 25.0, "rating": 2.9, "stock": 8},
            {"id": 5, "title": "Fallback Blue Efficient Flame", "category": "lighting", "brand": "Local", "price": 180, "discountPercentage": 7.0, "rating": 4.9, "stock": 42},
            {"id": 6, "title": "Fallback Small Chamber Flash", "category": "motorcycle", "brand": "Local", "price": 95, "discountPercentage": 18.0, "rating": 4.1, "stock": 12},
            {"id": 7, "title": "Fallback Cooling Airflow Product", "category": "laptops", "brand": "Local", "price": 700, "discountPercentage": 10.0, "rating": 4.4, "stock": 25},
            {"id": 8, "title": "Fallback Smoky Combustion Sample", "category": "groceries", "brand": "Local", "price": 15, "discountPercentage": 30.0, "rating": 3.2, "stock": 5},
        ][:limit]


# -----------------------------------------------------------------------------
# MAPPING DUMMYJSON PRODUCTS TO FLAME PARAMETERS
# -----------------------------------------------------------------------------

def safe_float(item: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(item.get(key, default))
    except (TypeError, ValueError):
        return default


def safe_int(item: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(item.get(key, default))
    except (TypeError, ValueError):
        return default


def category_flame_profile(category: str) -> str:
    """Pick a fuel profile from a DummyJSON category."""
    c = category.lower()
    if any(word in c for word in ("fragrance", "beauty", "skin")):
        return "fuel_rich"
    if any(word in c for word in ("vehicle", "motorcycle", "automotive")):
        return "hot"
    if any(word in c for word in ("kitchen", "lighting", "home")):
        return "balanced"
    if any(word in c for word in ("groceries", "food")):
        return "humid"
    if any(word in c for word in ("laptop", "smartphone", "tablet")):
        return "cooling"
    return "experimental"


def map_product_to_flame_params(
    product: Dict[str, Any],
    base: flame.FlameParams,
    round_number: int,
) -> flame.FlameParams:
    """
    Convert a DummyJSON product into FlameParams.

    Mapping:
    - price controls spark energy and fuel supply
    - rating controls reaction efficiency
    - stock controls oxygen availability
    - discount controls instability, soot, and humidity
    - category selects a fuel profile
    - title length changes chamber volume and ignition character
    """

    p = replace(base)

    product_id = safe_int(product, "id", round_number)
    title = str(product.get("title", f"DummyJSON product {product_id}"))
    category = str(product.get("category", "misc"))
    brand = str(product.get("brand", "unknown"))

    price = safe_float(product, "price", 50.0)
    rating = safe_float(product, "rating", 4.0)
    discount = safe_float(product, "discountPercentage", 0.0)
    stock = safe_float(product, "stock", 25.0)

    price_score = flame.clamp(price / 1000.0, 0.0, 1.0)
    rating_score = flame.clamp((rating - 1.0) / 4.0, 0.0, 1.0)
    discount_score = flame.clamp(discount / 30.0, 0.0, 1.0)
    stock_score = flame.clamp(stock / 100.0, 0.0, 1.0)
    title_score = flame.clamp(len(title) / 80.0, 0.0, 1.0)

    profile = category_flame_profile(category)
    p.fuel_name = f"DummyJSON {profile}: {title[:30]}"

    # Base API-driven parameters.
    p.initial_fuel = flame.clamp(0.75 + 0.55 * price_score + 0.25 * title_score, 0.05, 2.0)
    p.initial_oxygen = flame.clamp(0.70 + 0.75 * stock_score + 0.25 * rating_score, 0.05, 2.0)
    p.fuel_supply_rate = flame.clamp(0.07 + 0.16 * price_score + 0.06 * discount_score, 0.0, 0.5)
    p.oxygen_supply_rate = flame.clamp(0.36 + 0.62 * stock_score + 0.18 * rating_score, 0.0, 1.4)

    p.spark_energy_J = flame.clamp(560.0 + 360.0 * rating_score + 140.0 * price_score, 0.0, 1200.0)
    p.reaction_strength = flame.clamp(1.75 + 1.55 * rating_score + 0.35 * price_score, 0.1, 6.0)

    p.humidity = flame.clamp(0.10 + 0.42 * discount_score + (0.10 if profile == "humid" else 0.0), 0.0, 1.0)
    p.soot_rate = flame.clamp(0.010 + 0.045 * discount_score + (0.020 if profile == "fuel_rich" else 0.0), 0.0, 0.12)
    p.chamber_volume = flame.clamp(0.70 + 0.55 * title_score + 0.20 * price_score, 0.3, 3.0)

    # Profile-specific changes.
    if profile == "fuel_rich":
        p.initial_fuel = flame.clamp(p.initial_fuel + 0.35, 0.05, 2.0)
        p.initial_oxygen = flame.clamp(p.initial_oxygen - 0.20, 0.05, 2.0)
        p.oxygen_supply_rate = flame.clamp(p.oxygen_supply_rate * 0.75, 0.0, 1.4)
        p.soot_rate = flame.clamp(p.soot_rate + 0.025, 0.0, 0.12)

    elif profile == "hot":
        p.spark_energy_J = flame.clamp(p.spark_energy_J + 160.0, 0.0, 1200.0)
        p.reaction_strength = flame.clamp(p.reaction_strength + 0.75, 0.1, 6.0)
        p.oxygen_supply_rate = flame.clamp(p.oxygen_supply_rate + 0.20, 0.0, 1.4)
        p.humidity = flame.clamp(p.humidity * 0.65, 0.0, 1.0)

    elif profile == "balanced":
        p.initial_oxygen = flame.clamp(p.initial_oxygen + 0.18, 0.05, 2.0)
        p.convective_loss_coeff = 0.68
        p.mixing_loss_coeff = 0.12

    elif profile == "humid":
        p.humidity = flame.clamp(p.humidity + 0.22, 0.0, 1.0)
        p.spark_energy_J = flame.clamp(p.spark_energy_J - 70.0, 0.0, 1200.0)
        p.reaction_strength = flame.clamp(p.reaction_strength - 0.25, 0.1, 6.0)

    elif profile == "cooling":
        p.convective_loss_coeff = flame.clamp(1.15 + 0.45 * price_score, 0.05, 3.0)
        p.mixing_loss_coeff = flame.clamp(0.18 + 0.20 * stock_score, 0.0, 0.8)
        p.oxygen_supply_rate = flame.clamp(p.oxygen_supply_rate + 0.10, 0.0, 1.4)

    else:
        p.convective_loss_coeff = flame.clamp(0.70 + 0.35 * discount_score, 0.05, 3.0)
        p.mixing_loss_coeff = flame.clamp(0.09 + 0.18 * title_score, 0.0, 0.8)

    return p


# -----------------------------------------------------------------------------
# DISPLAY AND RUN
# -----------------------------------------------------------------------------

def print_api_items(products: List[Dict[str, Any]]) -> None:
    print("\nDummyJSON products controlling this flame run:")
    print("-" * 118)
    for index, product in enumerate(products, start=1):
        title = str(product.get("title", ""))
        category = str(product.get("category", ""))
        profile = category_flame_profile(category)
        print(
            f"{index:>2}. id={product.get('id')!s:<3} "
            f"profile={profile:<12} "
            f"price={product.get('price')!s:<7} "
            f"rating={product.get('rating')!s:<4} "
            f"stock={product.get('stock')!s:<4} "
            f"discount={product.get('discountPercentage')!s:<5} "
            f"category={category:<20} "
            f"title={title}"
        )


def print_api_final_summary(summaries: List[flame.RoundSummary]) -> None:
    print("\n" + "=" * 132)
    print("FINAL DUMMYJSON FLAME COMPARISON")
    print("=" * 132)
    print(
        f"{'Round':>5} | {'API fuel model':>30} | {'Peak K':>8} | {'Burn s':>7} | "
        f"{'Energy J':>9} | {'Fuel':>7} | {'O2 used':>8} | {'Eff':>6} | {'Soot':>7} | End state"
    )
    print("-" * 132)
    for s in summaries:
        print(
            f"{s.round_number:5d} | "
            f"{s.params.fuel_name[:30]:>30} | "
            f"{s.peak_temperature_K:8.1f} | "
            f"{s.burn_time_s:7.2f} | "
            f"{s.total_energy_released_J:9.2f} | "
            f"{s.fuel_burned:7.4f} | "
            f"{s.oxygen_used:8.4f} | "
            f"{s.average_efficiency:6.3f} | "
            f"{s.final_soot:7.4f} | {s.extinction_cause}"
        )

    hottest = max(summaries, key=lambda s: s.peak_temperature_K)
    longest = max(summaries, key=lambda s: s.burn_time_s)
    cleanest = min(summaries, key=lambda s: s.final_soot)

    print("\nDummyJSON-driven observations:")
    print(f"  Hottest API flame: Round {hottest.round_number}, peak {hottest.peak_temperature_K:.1f} K")
    print(f"  Longest active burn: Round {longest.round_number}, {longest.burn_time_s:.2f} s")
    print(f"  Cleanest burn: Round {cleanest.round_number}, soot {cleanest.final_soot:.4f}")


def main() -> None:
    config = flame.SimConfig(
        rounds=8,
        duration_s=6.0,
        dt_s=0.01,
        print_every_s=0.30,
        seed=23,
    )
    random.seed(config.seed)

    products = get_api_products(config.rounds)
    print_api_items(products)

    base_params = flame.FlameParams()
    summaries: List[flame.RoundSummary] = []

    print("\nDummyJSON-driven Simple Flame Combustion Simulation")
    print("Each DummyJSON product is converted into flame fuel, oxygen, spark, cooling, humidity, soot, and chamber parameters.")

    for round_number, product in enumerate(products, start=1):
        params = map_product_to_flame_params(product, base_params, round_number)
        summary = flame.run_round(round_number, params, config)
        summaries.append(summary)

    print_api_final_summary(summaries)


if __name__ == "__main__":
    main()
