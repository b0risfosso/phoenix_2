#!/usr/bin/env python3
"""
DummyJSON Product Burn Interaction Driver - Ignition Tuned
-----------------------------------------

This script fetches real product records from DummyJSON, places each returned
product into a simulated flame chamber, sets the product on fire, and lets it
burn until it is extinguished.

Place this file in the same folder as:
    simple_flame_combustion_simulation.py

Run:
    python dummyjson_product_burn_driver_ignition_tuned.py

No external packages are required.

Important:
- This is a terminal simulation only.
- It is not real fire-safety or combustion-engineering guidance.
- Product "materials" are inferred from DummyJSON category/title/description.
- The model is a simplified toy model for API interaction experiments.
"""

from __future__ import annotations

import json
import math
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import simple_flame_combustion_simulation as flame


BASE_URL = "https://dummyjson.com"


# -----------------------------------------------------------------------------
# API ACCESS
# -----------------------------------------------------------------------------

def request_json(endpoint: str, timeout_s: float = 10.0) -> Any:
    """Request JSON from DummyJSON using only the Python standard library."""
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    request = urllib.request.Request(
        url=BASE_URL + endpoint,
        headers={
            "Accept": "application/json",
            "User-Agent": "dummyjson-product-burn-driver/1.0",
        },
        method="GET",
    )

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def fetch_products(limit: int = 8, category: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch product records from DummyJSON.

    Use one of:
        category="fragrances"
        search="phone"
        neither, to fetch the first products
    """
    try:
        select = "title,description,category,brand,price,discountPercentage,rating,stock,weight,dimensions,availabilityStatus"

        if category:
            query = urllib.parse.urlencode({"limit": limit, "select": select})
            endpoint = f"/products/category/{urllib.parse.quote(category)}?{query}"
        elif search:
            query = urllib.parse.urlencode({"q": search, "limit": limit, "select": select})
            endpoint = f"/products/search?{query}"
        else:
            query = urllib.parse.urlencode({"limit": limit, "select": select})
            endpoint = f"/products?{query}"

        data = request_json(endpoint)
        products = data.get("products", []) if isinstance(data, dict) else []

        if not products:
            raise ValueError("DummyJSON returned no products.")

        return products[:limit]

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as error:
        print("\nAPI request failed. Using local fallback product records.")
        print(f"Reason: {error}")

        return [
            {
                "id": 1,
                "title": "Fallback Fragrance Bottle",
                "description": "A volatile fragrance-like product.",
                "category": "fragrances",
                "brand": "Fallback",
                "price": 49.99,
                "rating": 4.4,
                "discountPercentage": 3.5,
                "stock": 30,
                "weight": 1,
            },
            {
                "id": 2,
                "title": "Fallback Cotton Shirt",
                "description": "A fabric-like apparel product.",
                "category": "mens-shirts",
                "brand": "Fallback",
                "price": 24.99,
                "rating": 4.1,
                "discountPercentage": 12.0,
                "stock": 80,
                "weight": 1,
            },
            {
                "id": 3,
                "title": "Fallback Smartphone",
                "description": "An electronics-like product with casing and battery.",
                "category": "smartphones",
                "brand": "Fallback",
                "price": 699.99,
                "rating": 4.7,
                "discountPercentage": 5.0,
                "stock": 15,
                "weight": 1,
            },
            {
                "id": 4,
                "title": "Fallback Food Box",
                "description": "A grocery-like product that smolders and chars.",
                "category": "groceries",
                "brand": "Fallback",
                "price": 8.99,
                "rating": 3.9,
                "discountPercentage": 18.0,
                "stock": 60,
                "weight": 2,
            },
        ][:limit]


# -----------------------------------------------------------------------------
# PRODUCT BURN MODEL
# -----------------------------------------------------------------------------

@dataclass
class ProductMaterial:
    name: str
    ignition_temperature_K: float
    extinction_temperature_K: float
    heat_release_J_per_mass: float
    burn_rate_coeff: float
    smoke_coeff: float
    soot_coeff: float
    char_coeff: float
    moisture: float
    melt_coeff: float
    oxygen_demand: float


@dataclass
class ProductBurnState:
    t_s: float
    object_temperature_K: float
    chamber_temperature_K: float
    product_mass: float
    remaining_mass: float
    consumed_mass: float
    char_mass: float
    smoke: float
    soot: float
    melted: float
    oxygen: float
    flame_energy_J: float
    heat_lost_J: float
    state_name: str
    extinguish_reason: str


@dataclass
class ProductBurnSummary:
    round_number: int
    product_id: int
    title: str
    category: str
    material: ProductMaterial
    initial_mass: float
    consumed_mass: float
    remaining_mass: float
    char_mass: float
    smoke: float
    soot: float
    melted: float
    peak_object_temperature_K: float
    peak_chamber_temperature_K: float
    burn_time_s: float
    total_energy_J: float
    extinguish_reason: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(product: Dict[str, Any], key: str, default: float) -> float:
    try:
        value = product.get(key, default)
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(product: Dict[str, Any], key: str, default: int) -> int:
    try:
        value = product.get(key, default)
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def infer_material(product: Dict[str, Any]) -> ProductMaterial:
    """
    Infer a toy material profile from product category/title/description.

    The purpose is not realism; it creates different API-driven burn behaviors.
    """
    category = str(product.get("category", "")).lower()
    title = str(product.get("title", "")).lower()
    description = str(product.get("description", "")).lower()
    text = f"{category} {title} {description}"

    if any(word in text for word in ("fragrance", "perfume", "eau", "cologne")):
        return ProductMaterial(
            name="volatile fragrance liquid",
            ignition_temperature_K=560.0,
            extinction_temperature_K=470.0,
            heat_release_J_per_mass=1450.0,
            burn_rate_coeff=0.110,
            smoke_coeff=0.030,
            soot_coeff=0.014,
            char_coeff=0.015,
            moisture=0.06,
            melt_coeff=0.050,
            oxygen_demand=1.65,
        )

    if any(word in text for word in ("beauty", "lipstick", "mascara", "nail", "cream", "cosmetic")):
        return ProductMaterial(
            name="wax/oil cosmetic compound",
            ignition_temperature_K=610.0,
            extinction_temperature_K=505.0,
            heat_release_J_per_mass=1180.0,
            burn_rate_coeff=0.070,
            smoke_coeff=0.105,
            soot_coeff=0.060,
            char_coeff=0.060,
            moisture=0.12,
            melt_coeff=0.150,
            oxygen_demand=1.85,
        )

    if any(word in text for word in ("shirt", "dress", "shoe", "clothing", "fabric", "cotton", "apparel")):
        return ProductMaterial(
            name="fabric/textile",
            ignition_temperature_K=640.0,
            extinction_temperature_K=525.0,
            heat_release_J_per_mass=850.0,
            burn_rate_coeff=0.055,
            smoke_coeff=0.090,
            soot_coeff=0.040,
            char_coeff=0.180,
            moisture=0.18,
            melt_coeff=0.020,
            oxygen_demand=1.45,
        )

    if any(word in text for word in ("smartphone", "laptop", "tablet", "electronics", "battery", "phone")):
        return ProductMaterial(
            name="electronics/plastic casing",
            ignition_temperature_K=720.0,
            extinction_temperature_K=590.0,
            heat_release_J_per_mass=760.0,
            burn_rate_coeff=0.032,
            smoke_coeff=0.180,
            soot_coeff=0.090,
            char_coeff=0.120,
            moisture=0.05,
            melt_coeff=0.260,
            oxygen_demand=1.70,
        )

    if any(word in text for word in ("groceries", "food", "fruit", "vegetable", "meat", "rice", "oil")):
        return ProductMaterial(
            name="moist food/organic material",
            ignition_temperature_K=700.0,
            extinction_temperature_K=560.0,
            heat_release_J_per_mass=520.0,
            burn_rate_coeff=0.030,
            smoke_coeff=0.140,
            soot_coeff=0.025,
            char_coeff=0.210,
            moisture=0.42,
            melt_coeff=0.010,
            oxygen_demand=1.20,
        )

    if any(word in text for word in ("furniture", "wood", "table", "chair", "cabinet", "home-decoration")):
        return ProductMaterial(
            name="wood/composite household material",
            ignition_temperature_K=650.0,
            extinction_temperature_K=535.0,
            heat_release_J_per_mass=920.0,
            burn_rate_coeff=0.048,
            smoke_coeff=0.100,
            soot_coeff=0.035,
            char_coeff=0.220,
            moisture=0.20,
            melt_coeff=0.010,
            oxygen_demand=1.35,
        )

    if any(word in text for word in ("vehicle", "automotive", "motorcycle", "tire")):
        return ProductMaterial(
            name="rubber/plastic automotive material",
            ignition_temperature_K=690.0,
            extinction_temperature_K=560.0,
            heat_release_J_per_mass=1120.0,
            burn_rate_coeff=0.052,
            smoke_coeff=0.220,
            soot_coeff=0.120,
            char_coeff=0.160,
            moisture=0.06,
            melt_coeff=0.200,
            oxygen_demand=1.90,
        )

    return ProductMaterial(
        name="mixed consumer product",
        ignition_temperature_K=660.0,
        extinction_temperature_K=535.0,
        heat_release_J_per_mass=820.0,
        burn_rate_coeff=0.045,
        smoke_coeff=0.110,
        soot_coeff=0.055,
        char_coeff=0.120,
        moisture=0.18,
        melt_coeff=0.080,
        oxygen_demand=1.50,
    )


def product_mass_from_api(product: Dict[str, Any]) -> float:
    """
    Estimate a normalized product mass from the API fields.

    DummyJSON may provide a weight field for some products. If missing, use
    price/title/stock as a toy proxy.
    """
    weight = safe_float(product, "weight", 0.0)
    price = safe_float(product, "price", 50.0)
    title_len = len(str(product.get("title", "")))

    if weight > 0:
        return clamp(0.70 + 0.18 * weight, 0.35, 6.0)

    return clamp(0.80 + price / 350.0 + title_len / 120.0, 0.35, 6.0)


def starting_flame_for_product(product: Dict[str, Any], material: ProductMaterial) -> flame.FlameParams:
    """
    Build a flame source for this product.

    This uses the original flame parameter dataclass, but the burn loop below
    adds product mass, char, smoke, melting, and extinguishing behavior.
    """
    price = safe_float(product, "price", 50.0)
    rating = safe_float(product, "rating", 4.0)
    stock = safe_float(product, "stock", 25.0)
    discount = safe_float(product, "discountPercentage", 0.0)

    price_score = clamp(price / 1000.0, 0.0, 1.0)
    rating_score = clamp((rating - 1.0) / 4.0, 0.0, 1.0)
    stock_score = clamp(stock / 100.0, 0.0, 1.0)
    discount_score = clamp(discount / 30.0, 0.0, 1.0)

    p = flame.FlameParams()
    p.fuel_name = f"ignition flame for {str(product.get('title', 'product'))[:30]}"

    p.initial_fuel = clamp(0.95 + 0.35 * price_score, 0.20, 2.0)
    p.initial_oxygen = clamp(1.05 + 0.45 * stock_score + 0.15 * rating_score, 0.20, 2.0)
    p.fuel_supply_rate = clamp(0.12 + 0.08 * price_score, 0.0, 0.5)
    p.oxygen_supply_rate = clamp(0.65 + 0.35 * stock_score + 0.18 * rating_score, 0.0, 1.4)

    p.spark_energy_J = clamp(740.0 + 180.0 * rating_score, 0.0, 1200.0)
    p.ignition_temperature_K = material.ignition_temperature_K
    p.extinction_temperature_K = material.extinction_temperature_K
    p.reaction_strength = clamp(2.6 + 1.0 * rating_score + 0.4 * price_score, 0.1, 6.0)
    p.heat_of_combustion_J_per_unit = clamp(780.0 + 220.0 * rating_score, 520.0, 1500.0)
    p.heat_capacity_J_per_K = clamp(0.72 + 0.12 * material.moisture, 0.45, 1.20)

    # Sustain-friendly flame source.
    p.convective_loss_coeff = clamp(0.28 + 0.18 * material.moisture, 0.05, 3.0)
    p.radiative_loss_coeff = 2.6e-9
    p.mixing_loss_coeff = clamp(0.045 + 0.05 * discount_score, 0.0, 0.8)
    p.humidity = clamp(material.moisture + 0.15 * discount_score, 0.0, 1.0)
    p.soot_rate = clamp(material.soot_coeff, 0.0, 0.12)
    p.chamber_volume = 0.72

    return p


def classify_burn_state(state: ProductBurnState, material: ProductMaterial) -> str:
    if state.remaining_mass <= 0.01:
        return "consumed"
    if state.oxygen <= 0.02:
        return "oxygen-starved"
    if state.object_temperature_K >= material.ignition_temperature_K and state.remaining_mass > 0:
        return "burning"
    if state.object_temperature_K >= material.extinction_temperature_K:
        return "smoldering"
    if state.object_temperature_K > 330.0:
        return "cooling"
    return "extinguished"


def simulate_product_burn(
    product: Dict[str, Any],
    round_number: int,
    config: flame.SimConfig,
    rng: random.Random,
) -> ProductBurnSummary:
    """
    Set the fetched product on fire and simulate until extinguished or max time.

    The flame source and the product object exchange heat. The object burns when
    its object temperature crosses the material ignition threshold. After fuel
    and oxygen stop supporting combustion, it cools and becomes extinguished.
    """
    material = infer_material(product)

    # Ignition-tuned toy adjustment:
    # Lower ignition/extinction thresholds slightly so the terminal experiment
    # visibly burns products instead of only warming them.
    material = ProductMaterial(
        name=material.name,
        ignition_temperature_K=max(430.0, material.ignition_temperature_K - 90.0),
        extinction_temperature_K=max(360.0, material.extinction_temperature_K - 70.0),
        heat_release_J_per_mass=material.heat_release_J_per_mass,
        burn_rate_coeff=material.burn_rate_coeff * 1.35,
        smoke_coeff=material.smoke_coeff,
        soot_coeff=material.soot_coeff,
        char_coeff=material.char_coeff,
        moisture=material.moisture,
        melt_coeff=material.melt_coeff,
        oxygen_demand=material.oxygen_demand,
    )

    params = starting_flame_for_product(product, material)

    product_id = safe_int(product, "id", round_number)
    title = str(product.get("title", f"Product {product_id}"))
    category = str(product.get("category", "unknown"))
    mass = product_mass_from_api(product)

    state = ProductBurnState(
        t_s=0.0,
        object_temperature_K=300.0,
        chamber_temperature_K=params.initial_temperature_K,
        product_mass=mass,
        remaining_mass=mass,
        consumed_mass=0.0,
        char_mass=0.0,
        smoke=0.0,
        soot=0.0,
        melted=0.0,
        oxygen=params.initial_oxygen,
        flame_energy_J=0.0,
        heat_lost_J=0.0,
        state_name="unignited",
        extinguish_reason="not finished",
    )

    flame_state = flame.FlameState(
        temperature_K=params.initial_temperature_K,
        fuel=params.initial_fuel,
        oxygen=params.initial_oxygen,
    )

    peak_object_temperature = state.object_temperature_K
    peak_chamber_temperature = state.chamber_temperature_K
    burn_time = 0.0
    next_print = 0.0

    max_steps = int(config.duration_s / config.dt_s)

    print("\n" + "=" * 132)
    print(f"PRODUCT BURN ROUND {round_number}: {title}")
    print("=" * 132)
    print(
        f"DummyJSON product id={product_id}, category={category}, brand={product.get('brand', 'unknown')}, "
        f"price={product.get('price')}, rating={product.get('rating')}, stock={product.get('stock')}"
    )
    print(
        f"inferred material={material.name}, mass={mass:.2f}, "
        f"ignition={material.ignition_temperature_K:.1f} K, extinction={material.extinction_temperature_K:.1f} K"
    )
    print("-" * 132)
    print(
        f"{'t(s)':>6} | {'state':>15} | {'obj K':>8} | {'chamber K':>9} | {'mass':>7} | "
        f"{'burned':>7} | {'char':>7} | {'O2':>6} | {'smoke':>7} | {'soot':>7} | {'melt':>7} | {'reason':>16}"
    )
    print("-" * 132)

    last_rate = 0.0

    for step in range(max_steps + 1):
        is_first_step = step == 0

        # Original flame step provides the ignition source and chamber heat.
        flame_state, diagnostics = flame.step_flame(flame_state, params, config.dt_s, is_first_step=is_first_step)

        # Maintain a short pilot flame while the object is still heating.
        # This prevents the chamber from flashing hot for only one instant and
        # cooling before the object can absorb enough energy to ignite.
        pilot_active = state.t_s < 4.0 and state.remaining_mass > 0.01
        if pilot_active:
            pilot_target_K = material.ignition_temperature_K + 260.0
            if flame_state.temperature_K < pilot_target_K:
                flame_state.temperature_K += (pilot_target_K - flame_state.temperature_K) * 0.030

        # Chamber exchanges heat with the product and ambient environment.
        chamber_heat = flame_state.temperature_K
        exposure = clamp((chamber_heat - state.object_temperature_K) / 1000.0, -0.15, 1.25)

        # Ignition-tuned heat transfer:
        # The previous version heated the chamber strongly but transferred too little
        # heat into the product object. This stronger direct-contact term lets
        # the fetched product actually reach ignition temperature when exposed
        # to a hot flame.
        direct_contact_boost = 1.0
        if chamber_heat > material.ignition_temperature_K:
            direct_contact_boost += 1.35
        if chamber_heat > material.ignition_temperature_K + 250.0:
            direct_contact_boost += 0.85

        heat_absorbed_K = (
            exposure
            * (65.0 + 30.0 / max(0.4, mass))
            * direct_contact_boost
            * config.dt_s
            * 10.0
        )

        # Burning product feeds heat back into the chamber.
        hot_enough = state.object_temperature_K >= material.ignition_temperature_K
        enough_oxygen = state.oxygen > 0.02
        enough_mass = state.remaining_mass > 0.01

        burn_rate = 0.0
        if hot_enough and enough_oxygen and enough_mass:
            thermal_excess = clamp(
                (state.object_temperature_K - material.ignition_temperature_K) / 500.0,
                0.0,
                2.0,
            )
            oxygen_factor = clamp(state.oxygen / 1.8, 0.05, 1.0)
            moisture_penalty = clamp(1.0 - material.moisture * 0.70, 0.20, 1.0)
            burn_rate = material.burn_rate_coeff * state.remaining_mass * (0.35 + thermal_excess) * oxygen_factor * moisture_penalty

        mass_burned = min(state.remaining_mass, burn_rate * config.dt_s)
        oxygen_used = mass_burned * material.oxygen_demand
        if oxygen_used > state.oxygen:
            mass_burned = state.oxygen / max(0.001, material.oxygen_demand)
            oxygen_used = state.oxygen

        heat_from_product = mass_burned * material.heat_release_J_per_mass
        state.flame_energy_J += heat_from_product

        # Product byproducts.
        state.remaining_mass -= mass_burned
        state.consumed_mass += mass_burned
        state.oxygen = clamp(
            state.oxygen
            + params.oxygen_supply_rate * config.dt_s
            - oxygen_used,
            0.0,
            params.max_oxygen,
        )

        char_created = mass_burned * material.char_coeff
        smoke_created = mass_burned * material.smoke_coeff * (1.0 + material.moisture)
        soot_created = mass_burned * material.soot_coeff
        melt_created = 0.0

        if state.object_temperature_K > material.ignition_temperature_K - 80.0:
            melt_created = material.melt_coeff * config.dt_s * clamp(
                (state.object_temperature_K - (material.ignition_temperature_K - 80.0)) / 220.0,
                0.0,
                2.0,
            )

        state.char_mass += char_created
        state.smoke += smoke_created
        state.soot += soot_created
        state.melted += melt_created

        # Temperature update.
        product_heat_capacity = 1.15 + 0.55 * material.moisture + 0.12 * state.remaining_mass
        product_heat_gain = heat_absorbed_K
        product_heat_gain += heat_from_product / max(0.20, product_heat_capacity) * 0.26

        cooling_loss_K = (
            0.85
            + 0.35 * material.moisture
            + 0.18 * state.char_mass
        ) * (state.object_temperature_K - 300.0) / 900.0 * config.dt_s * 18.0

        state.object_temperature_K += product_heat_gain - cooling_loss_K
        if state.object_temperature_K < 300.0:
            state.object_temperature_K = 300.0

        # Chamber receives a fraction of product heat but also loses heat.
        flame_state.temperature_K += heat_from_product / max(0.3, params.heat_capacity_J_per_K) * 0.06
        state.chamber_temperature_K = flame_state.temperature_K

        # Smoke slowly disperses.
        state.smoke *= math.exp(-0.040 * config.dt_s)

        state.t_s += config.dt_s
        state.state_name = classify_burn_state(state, material)
        last_rate = burn_rate

        peak_object_temperature = max(peak_object_temperature, state.object_temperature_K)
        peak_chamber_temperature = max(peak_chamber_temperature, state.chamber_temperature_K)

        if state.state_name in ("burning", "smoldering"):
            burn_time += config.dt_s

        # Determine whether the burn has finished.
        extinguished = False
        if state.remaining_mass <= 0.01:
            state.extinguish_reason = "product consumed"
            extinguished = True
        elif state.t_s > 2.50 and state.object_temperature_K < material.extinction_temperature_K and last_rate < 0.0005:
            state.extinguish_reason = "cooled below extinction"
            extinguished = True
        elif state.oxygen <= 0.01 and last_rate < 0.0005:
            state.extinguish_reason = "oxygen starved"
            extinguished = True

        if state.t_s + 1e-9 >= next_print or step == max_steps or extinguished:
            print(
                f"{state.t_s:6.2f} | {state.state_name:>15} | "
                f"{state.object_temperature_K:8.1f} | {state.chamber_temperature_K:9.1f} | "
                f"{state.remaining_mass:7.3f} | {state.consumed_mass:7.3f} | "
                f"{state.char_mass:7.3f} | {state.oxygen:6.3f} | "
                f"{state.smoke:7.4f} | {state.soot:7.4f} | {state.melted:7.4f} | "
                f"{state.extinguish_reason:>16}"
            )
            next_print += config.print_every_s

        if extinguished:
            break

    if state.extinguish_reason == "not finished":
        if state.state_name in ("burning", "smoldering"):
            state.extinguish_reason = "still burning at time limit"
        else:
            state.extinguish_reason = "time limit reached"

    summary = ProductBurnSummary(
        round_number=round_number,
        product_id=product_id,
        title=title,
        category=category,
        material=material,
        initial_mass=mass,
        consumed_mass=state.consumed_mass,
        remaining_mass=state.remaining_mass,
        char_mass=state.char_mass,
        smoke=state.smoke,
        soot=state.soot,
        melted=state.melted,
        peak_object_temperature_K=peak_object_temperature,
        peak_chamber_temperature_K=peak_chamber_temperature,
        burn_time_s=burn_time,
        total_energy_J=state.flame_energy_J,
        extinguish_reason=state.extinguish_reason,
    )

    print("-" * 132)
    print_product_summary(summary)
    return summary


def print_product_summary(summary: ProductBurnSummary) -> None:
    print("Product burn result:")
    print(f"  product: {summary.title}")
    print(f"  category: {summary.category}")
    print(f"  material: {summary.material.name}")
    print(f"  initial mass: {summary.initial_mass:.3f}")
    print(f"  consumed mass: {summary.consumed_mass:.3f}")
    print(f"  remaining mass: {summary.remaining_mass:.3f}")
    print(f"  char mass: {summary.char_mass:.3f}")
    print(f"  smoke: {summary.smoke:.4f}")
    print(f"  soot: {summary.soot:.4f}")
    print(f"  melted: {summary.melted:.4f}")
    print(f"  peak object temperature: {summary.peak_object_temperature_K:.1f} K")
    print(f"  peak chamber temperature: {summary.peak_chamber_temperature_K:.1f} K")
    print(f"  burn/smolder time: {summary.burn_time_s:.2f} s")
    print(f"  product energy released: {summary.total_energy_J:.2f} J")
    print(f"  end state: {summary.extinguish_reason}")


def print_final_summary(summaries: List[ProductBurnSummary]) -> None:
    print("\n" + "=" * 150)
    print("FINAL DUMMYJSON PRODUCT BURN COMPARISON")
    print("=" * 150)
    print(
        f"{'Round':>5} | {'Product':>28} | {'Material':>30} | {'Mass':>6} | "
        f"{'Burned':>7} | {'Remain':>7} | {'Char':>7} | {'Smoke':>7} | "
        f"{'Soot':>7} | {'Melt':>7} | {'PeakObjK':>8} | {'Time':>6} | End"
    )
    print("-" * 150)

    for s in summaries:
        print(
            f"{s.round_number:5d} | "
            f"{s.title[:28]:>28} | "
            f"{s.material.name[:30]:>30} | "
            f"{s.initial_mass:6.2f} | "
            f"{s.consumed_mass:7.3f} | "
            f"{s.remaining_mass:7.3f} | "
            f"{s.char_mass:7.3f} | "
            f"{s.smoke:7.4f} | "
            f"{s.soot:7.4f} | "
            f"{s.melted:7.4f} | "
            f"{s.peak_object_temperature_K:8.1f} | "
            f"{s.burn_time_s:6.2f} | {s.extinguish_reason}"
        )

    most_consumed = max(summaries, key=lambda s: s.consumed_mass / max(0.001, s.initial_mass))
    smokiest = max(summaries, key=lambda s: s.smoke)
    hottest = max(summaries, key=lambda s: s.peak_object_temperature_K)
    longest = max(summaries, key=lambda s: s.burn_time_s)

    print("\nAPI-driven product burn observations:")
    print(
        f"  Most consumed product: Round {most_consumed.round_number}, "
        f"{most_consumed.title}, {100 * most_consumed.consumed_mass / max(0.001, most_consumed.initial_mass):.1f}% consumed"
    )
    print(f"  Smokiest product: Round {smokiest.round_number}, {smokiest.title}, smoke={smokiest.smoke:.4f}")
    print(f"  Hottest object: Round {hottest.round_number}, {hottest.title}, peak={hottest.peak_object_temperature_K:.1f} K")
    print(f"  Longest burn/smolder time: Round {longest.round_number}, {longest.title}, {longest.burn_time_s:.2f} s")


# -----------------------------------------------------------------------------
# USER CONTROLS
# -----------------------------------------------------------------------------

def ask_int(prompt: str, default: int, low: int, high: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return max(low, min(high, int(raw)))
    except ValueError:
        print("Invalid number. Using default.")
        return default


def ask_text(prompt: str, default: str = "") -> str:
    raw = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return raw if raw else default


def choose_fetch_mode() -> tuple[Optional[str], Optional[str]]:
    print("\nChoose product fetch mode:")
    print("1. First products")
    print("2. Products by category")
    print("3. Product search")
    choice = input("Mode [1]: ").strip() or "1"

    if choice == "2":
        category = ask_text("Category slug", "fragrances")
        return category, None

    if choice == "3":
        search = ask_text("Search query", "phone")
        return None, search

    return None, None


def main() -> None:
    print("DummyJSON Product Burn Interaction Driver - Ignition Tuned")
    print("Fetches actual DummyJSON product records and simulates each returned product burning with stronger object ignition heating.")
    print("Terminal-only toy model. No external packages.")

    limit = ask_int("Number of products to fetch", 8, 1, 20)
    category, search = choose_fetch_mode()

    config = flame.SimConfig(
        rounds=limit,
        duration_s=24.0,
        dt_s=0.01,
        print_every_s=0.50,
        seed=41,
    )

    rng = random.Random(config.seed)
    products = fetch_products(limit=limit, category=category, search=search)

    print("\nFetched products:")
    print("-" * 120)
    for i, product in enumerate(products, start=1):
        print(
            f"{i:>2}. id={product.get('id')} | "
            f"category={product.get('category')} | "
            f"price={product.get('price')} | "
            f"rating={product.get('rating')} | "
            f"title={product.get('title')}"
        )

    summaries: List[ProductBurnSummary] = []

    for round_number, product in enumerate(products, start=1):
        summary = simulate_product_burn(product, round_number, config, rng)
        summaries.append(summary)

    print_final_summary(summaries)


if __name__ == "__main__":
    main()
