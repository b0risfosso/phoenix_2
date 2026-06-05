"""
Terminal Orange Chicken Cooking Simulation with AI-Controlled Rounds

A Python-only terminal simulation of cooking orange chicken.

Features:
- No external packages
- Multiple cooking rounds
- AI controller changes recipe/cooking parameters after each round
- Simulates chicken temperature, moisture, doneness, crispiness, oil temperature,
  batter thickness, sauce thickness, sweetness, acidity, orange flavor, coating,
  burn risk, and final quality score
- Prints state directly to the terminal
- Ends with a final comparison table

Run:
    python terminal_orange_chicken_ai_rounds.py

Note:
    This is a toy cooking simulation, not food-safety guidance.
"""

import random
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# User-adjustable settings
# ---------------------------------------------------------------------------

RANDOM_SEED = 31
ROUNDS = 8
DT_MIN = 0.10
PRINT_EVERY_MIN = 0.50

STARTING_CHICKEN_TEMP_C = 5.0
ROOM_TEMP_C = 22.0
SAFE_DONE_TEMP_C = 74.0

# Stages are simulated in minutes.
PREP_TIME_MIN = 1.0
COAT_TIME_MIN = 1.0
OIL_HEAT_TIME_MIN = 2.0
TOSS_TIME_MIN = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RecipeParams:
    goal: str
    oil_target_temp_c: float
    fry_time_min: float
    chicken_piece_size: float
    batter_thickness: float
    cornstarch_ratio: float
    sauce_heat: float
    sauce_simmer_time_min: float
    orange_juice_ml: float
    sugar_g: float
    soy_sauce_ml: float
    vinegar_ml: float
    garlic_ginger: float
    toss_time_min: float
    sauce_amount: float


@dataclass
class CookingState:
    time_min: float
    stage: str
    chicken_temp_c: float
    moisture: float
    doneness: float
    crispiness: float
    oil_temp_c: float
    sauce_thickness: float
    sweetness: float
    acidity: float
    saltiness: float
    orange_flavor: float
    garlic_ginger_intensity: float
    coating_level: float
    burn_risk: float
    sauce_burn_risk: float
    aroma: float
    quality_estimate: float


@dataclass
class RoundSummary:
    round_number: int
    goal: str
    oil_temp_c: float
    fry_time_min: float
    sauce_simmer_time_min: float
    orange_juice_ml: float
    sugar_g: float
    doneness: float
    crispiness: float
    moisture: float
    sauce_thickness: float
    orange_balance: float
    sweetness: float
    acidity: float
    coating_level: float
    burn_risk: float
    sauce_burn_risk: float
    quality_score: float
    result: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def bar(value, max_value=1.0, width=16):
    if max_value <= 0:
        filled = 0
    else:
        filled = int(width * clamp(value / max_value, 0.0, 1.0))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_round_header(round_number, params):
    print()
    print("=" * 118)
    print(f"ROUND {round_number}")
    print("-" * 118)
    print(f"goal: {params.goal}")
    print(
        "parameters: "
        f"oil_target={params.oil_target_temp_c:.1f}C, "
        f"fry_time={params.fry_time_min:.1f}min, "
        f"piece_size={params.chicken_piece_size:.2f}, "
        f"batter={params.batter_thickness:.2f}, "
        f"cornstarch={params.cornstarch_ratio:.2f}, "
        f"sauce_heat={params.sauce_heat:.2f}, "
        f"simmer={params.sauce_simmer_time_min:.1f}min, "
        f"orange_juice={params.orange_juice_ml:.0f}ml, "
        f"sugar={params.sugar_g:.0f}g, "
        f"soy={params.soy_sauce_ml:.0f}ml, "
        f"vinegar={params.vinegar_ml:.0f}ml, "
        f"garlic_ginger={params.garlic_ginger:.2f}, "
        f"toss_time={params.toss_time_min:.1f}min, "
        f"sauce_amount={params.sauce_amount:.2f}"
    )
    print("-" * 118)


def print_state(round_number, state):
    print(
        f"R{round_number:02d} "
        f"t={state.time_min:05.2f}min  "
        f"stage={state.stage:<14}  "
        f"chicken={state.chicken_temp_c:6.1f}C  "
        f"done={state.doneness:4.2f} {bar(state.doneness, 1.0, 8)}  "
        f"moist={state.moisture:4.2f}  "
        f"crisp={state.crispiness:4.2f} {bar(state.crispiness, 1.0, 8)}  "
        f"oil={state.oil_temp_c:6.1f}C  "
        f"sauce={state.sauce_thickness:4.2f}  "
        f"orange={state.orange_flavor:4.2f}  "
        f"coat={state.coating_level:4.2f}  "
        f"burn={state.burn_risk:4.2f}  "
        f"score~={state.quality_estimate:5.1f}"
    )


def flavor_balance_score(sweetness, acidity, saltiness, orange_flavor, garlic_ginger):
    """
    Scores balance around target orange-chicken profile:
    bright orange, moderate sweetness, mild acidity, moderate salt, supporting aromatics.
    """
    score = 1.0
    score -= abs(sweetness - 0.72) * 0.55
    score -= abs(acidity - 0.48) * 0.45
    score -= abs(saltiness - 0.42) * 0.40
    score -= abs(orange_flavor - 0.86) * 0.55
    score -= abs(garlic_ginger - 0.55) * 0.25
    return clamp(score, 0.0, 1.0)


def estimate_quality(state):
    flavor = flavor_balance_score(
        state.sweetness,
        state.acidity,
        state.saltiness,
        state.orange_flavor,
        state.garlic_ginger_intensity,
    )

    score = 100.0
    score -= abs(state.doneness - 1.0) * 28.0
    score -= abs(state.crispiness - 0.78) * 22.0
    score -= abs(state.moisture - 0.55) * 18.0
    score -= abs(state.sauce_thickness - 0.72) * 18.0
    score -= abs(state.coating_level - 0.82) * 15.0
    score += flavor * 18.0 - 9.0
    score -= state.burn_risk * 30.0
    score -= state.sauce_burn_risk * 24.0

    return clamp(score, 0.0, 100.0)


def classify_result(score, state):
    if state.burn_risk > 0.55 or state.sauce_burn_risk > 0.55:
        return "burnt notes"
    if state.doneness < 0.92:
        return "undercooked"
    if state.moisture < 0.32:
        return "dry chicken"
    if state.crispiness < 0.45:
        return "soft coating"
    if score >= 90:
        return "excellent orange chicken"
    if score >= 82:
        return "good orange chicken"
    if score >= 72:
        return "acceptable result"
    if score >= 60:
        return "uneven but edible"
    return "poor result"


# ---------------------------------------------------------------------------
# Simulation stages
# ---------------------------------------------------------------------------

def initial_state():
    return CookingState(
        time_min=0.0,
        stage="start",
        chicken_temp_c=STARTING_CHICKEN_TEMP_C,
        moisture=0.92,
        doneness=0.0,
        crispiness=0.0,
        oil_temp_c=ROOM_TEMP_C,
        sauce_thickness=0.0,
        sweetness=0.0,
        acidity=0.0,
        saltiness=0.0,
        orange_flavor=0.0,
        garlic_ginger_intensity=0.0,
        coating_level=0.0,
        burn_risk=0.0,
        sauce_burn_risk=0.0,
        aroma=0.0,
        quality_estimate=0.0,
    )


def simulate_prep_and_coating(state, params, round_number, next_print):
    stage_end = state.time_min + PREP_TIME_MIN + COAT_TIME_MIN

    while state.time_min <= stage_end:
        if state.time_min < PREP_TIME_MIN:
            state.stage = "prep"
            # Chicken warms slightly from fridge temperature.
            state.chicken_temp_c += (ROOM_TEMP_C - state.chicken_temp_c) * 0.04 * DT_MIN
            state.moisture -= 0.004 * DT_MIN
            state.aroma += 0.02 * DT_MIN
        else:
            state.stage = "coating"
            target_coating = clamp(
                0.35 + params.batter_thickness * 0.45 + params.cornstarch_ratio * 0.25,
                0.0,
                1.0,
            )
            state.coating_level += (target_coating - state.coating_level) * 0.18
            state.crispiness += 0.012 * params.cornstarch_ratio
            state.moisture -= 0.002 * DT_MIN

        state.doneness = clamp((state.chicken_temp_c - 45.0) / (SAFE_DONE_TEMP_C - 45.0), 0.0, 1.0)
        state.quality_estimate = estimate_quality(state)

        if state.time_min >= next_print[0]:
            print_state(round_number, state)
            next_print[0] += PRINT_EVERY_MIN

        state.time_min += DT_MIN


def simulate_oil_heating(state, params, round_number, next_print):
    stage_end = state.time_min + OIL_HEAT_TIME_MIN

    while state.time_min <= stage_end:
        state.stage = "heat oil"

        # Oil moves toward target.
        heat_rate = 0.12 + 0.08 * clamp(params.oil_target_temp_c / 190.0, 0.0, 1.4)
        state.oil_temp_c += (params.oil_target_temp_c - state.oil_temp_c) * heat_rate * DT_MIN

        # Chicken remains waiting.
        state.chicken_temp_c += (ROOM_TEMP_C - state.chicken_temp_c) * 0.015 * DT_MIN
        state.moisture -= 0.001 * DT_MIN
        state.quality_estimate = estimate_quality(state)

        if state.time_min >= next_print[0]:
            print_state(round_number, state)
            next_print[0] += PRINT_EVERY_MIN

        state.time_min += DT_MIN


def simulate_frying(state, params, round_number, next_print):
    stage_end = state.time_min + params.fry_time_min

    while state.time_min <= stage_end:
        state.stage = "frying"

        # Adding chicken pulls oil temperature down; burner pushes it back up.
        oil_drop = 0.08 * params.chicken_piece_size
        state.oil_temp_c -= oil_drop
        state.oil_temp_c += (params.oil_target_temp_c - state.oil_temp_c) * 0.035

        # Heat transfer into chicken. Smaller pieces cook faster.
        heat_transfer = 0.030 / max(params.chicken_piece_size, 0.35)
        state.chicken_temp_c += (state.oil_temp_c - state.chicken_temp_c) * heat_transfer * DT_MIN

        # Doneness rises as chicken temp approaches safe range.
        state.doneness = clamp((state.chicken_temp_c - 45.0) / (SAFE_DONE_TEMP_C - 45.0), 0.0, 1.0)

        # Moisture loss and crisping.
        evaporation = 0.0025 * max(0.0, state.oil_temp_c - 120.0) / 60.0
        evaporation *= (0.8 + 0.5 * params.chicken_piece_size)
        state.moisture -= evaporation * DT_MIN
        state.moisture = clamp(state.moisture, 0.05, 1.0)

        crisp_gain = (
            0.020
            * params.cornstarch_ratio
            * params.batter_thickness
            * max(0.0, state.oil_temp_c - 130.0) / 55.0
            * (1.05 - state.moisture)
        )
        state.crispiness += crisp_gain * DT_MIN

        # Too much batter can become heavy instead of crisp.
        if params.batter_thickness > 1.15 and state.moisture > 0.60:
            state.crispiness -= 0.006 * DT_MIN

        state.crispiness = clamp(state.crispiness, 0.0, 1.0)

        # Burn risk grows from too-hot oil, long frying, or very dry coating.
        if state.oil_temp_c > 188.0:
            state.burn_risk += (state.oil_temp_c - 188.0) * 0.0025 * DT_MIN
        if state.crispiness > 0.90 and state.moisture < 0.38:
            state.burn_risk += 0.020 * DT_MIN

        state.burn_risk = clamp(state.burn_risk, 0.0, 1.0)
        state.aroma += 0.020 * DT_MIN
        state.quality_estimate = estimate_quality(state)

        if state.time_min >= next_print[0]:
            print_state(round_number, state)
            next_print[0] += PRINT_EVERY_MIN

        state.time_min += DT_MIN


def initialize_sauce_flavors(state, params):
    total = max(
        params.orange_juice_ml
        + params.soy_sauce_ml
        + params.vinegar_ml
        + params.sugar_g * 0.8,
        1.0,
    )

    state.orange_flavor = clamp(params.orange_juice_ml / 150.0, 0.0, 1.25)
    state.sweetness = clamp(params.sugar_g / 50.0, 0.0, 1.25)
    state.acidity = clamp(params.vinegar_ml / 35.0 + params.orange_juice_ml / 500.0, 0.0, 1.25)
    state.saltiness = clamp(params.soy_sauce_ml / 45.0, 0.0, 1.25)
    state.garlic_ginger_intensity = clamp(params.garlic_ginger, 0.0, 1.25)

    # Thin at first; high sugar starts slightly thicker.
    state.sauce_thickness = clamp(0.10 + params.sugar_g / 250.0, 0.0, 0.55)


def simulate_sauce(state, params, round_number, next_print):
    state.stage = "mix sauce"
    initialize_sauce_flavors(state, params)

    stage_end = state.time_min + params.sauce_simmer_time_min

    while state.time_min <= stage_end:
        state.stage = "simmer sauce"

        # Sauce thickens with heat, sugar, and time.
        sugar_factor = clamp(params.sugar_g / 45.0, 0.25, 1.8)
        thickening = 0.055 * params.sauce_heat * sugar_factor
        state.sauce_thickness += thickening * DT_MIN

        # Simmer concentrates flavor.
        concentration = 0.010 * params.sauce_heat * DT_MIN
        state.orange_flavor += concentration * 0.60
        state.sweetness += concentration * 0.35
        state.saltiness += concentration * 0.25
        state.acidity += concentration * 0.10

        # Too much heat can dull orange and burn sugar.
        if params.sauce_heat > 1.10 and state.sauce_thickness > 0.75:
            state.sauce_burn_risk += (params.sauce_heat - 1.0) * 0.035 * DT_MIN
            state.orange_flavor -= 0.018 * DT_MIN

        # Too long creates sticky sauce.
        if state.sauce_thickness > 0.92:
            state.sauce_burn_risk += 0.018 * DT_MIN

        state.sauce_thickness = clamp(state.sauce_thickness, 0.0, 1.0)
        state.orange_flavor = clamp(state.orange_flavor, 0.0, 1.25)
        state.sweetness = clamp(state.sweetness, 0.0, 1.25)
        state.acidity = clamp(state.acidity, 0.0, 1.25)
        state.saltiness = clamp(state.saltiness, 0.0, 1.25)
        state.sauce_burn_risk = clamp(state.sauce_burn_risk, 0.0, 1.0)

        state.quality_estimate = estimate_quality(state)

        if state.time_min >= next_print[0]:
            print_state(round_number, state)
            next_print[0] += PRINT_EVERY_MIN

        state.time_min += DT_MIN


def simulate_tossing(state, params, round_number, next_print):
    toss_duration = params.toss_time_min
    stage_end = state.time_min + toss_duration

    while state.time_min <= stage_end:
        state.stage = "tossing"

        # Sauce coats chicken gradually.
        ideal_coating = clamp(0.25 + params.sauce_amount * 0.70, 0.0, 1.0)
        coating_rate = 0.18 + 0.08 * params.sauce_thickness
        state.coating_level += (ideal_coating - state.coating_level) * coating_rate

        # Sauce softens crispiness; thicker sauce sticks but still softens.
        sog_factor = params.sauce_amount * (0.030 + 0.025 * state.sauce_thickness)
        state.crispiness -= sog_factor * DT_MIN

        # Hot tossing can slightly continue cooking.
        state.chicken_temp_c += (88.0 - state.chicken_temp_c) * 0.006 * DT_MIN
        state.doneness = clamp((state.chicken_temp_c - 45.0) / (SAFE_DONE_TEMP_C - 45.0), 0.0, 1.0)

        # If sauce is burnt, tossing spreads burnt flavor.
        state.burn_risk += state.sauce_burn_risk * 0.006 * DT_MIN

        state.coating_level = clamp(state.coating_level, 0.0, 1.0)
        state.crispiness = clamp(state.crispiness, 0.0, 1.0)
        state.burn_risk = clamp(state.burn_risk, 0.0, 1.0)

        state.quality_estimate = estimate_quality(state)

        if state.time_min >= next_print[0]:
            print_state(round_number, state)
            next_print[0] += PRINT_EVERY_MIN

        state.time_min += DT_MIN


# ---------------------------------------------------------------------------
# Full round simulation
# ---------------------------------------------------------------------------

def simulate_round(round_number, params):
    print_round_header(round_number, params)

    state = initial_state()
    next_print = [0.0]

    simulate_prep_and_coating(state, params, round_number, next_print)
    simulate_oil_heating(state, params, round_number, next_print)
    simulate_frying(state, params, round_number, next_print)
    simulate_sauce(state, params, round_number, next_print)
    simulate_tossing(state, params, round_number, next_print)

    state.stage = "served"
    state.quality_estimate = estimate_quality(state)

    orange_balance = flavor_balance_score(
        state.sweetness,
        state.acidity,
        state.saltiness,
        state.orange_flavor,
        state.garlic_ginger_intensity,
    )

    result = classify_result(state.quality_estimate, state)

    summary = RoundSummary(
        round_number=round_number,
        goal=params.goal,
        oil_temp_c=params.oil_target_temp_c,
        fry_time_min=params.fry_time_min,
        sauce_simmer_time_min=params.sauce_simmer_time_min,
        orange_juice_ml=params.orange_juice_ml,
        sugar_g=params.sugar_g,
        doneness=state.doneness,
        crispiness=state.crispiness,
        moisture=state.moisture,
        sauce_thickness=state.sauce_thickness,
        orange_balance=orange_balance,
        sweetness=state.sweetness,
        acidity=state.acidity,
        coating_level=state.coating_level,
        burn_risk=state.burn_risk,
        sauce_burn_risk=state.sauce_burn_risk,
        quality_score=state.quality_estimate,
        result=result,
    )

    print("-" * 118)
    print(
        f"summary: doneness={summary.doneness:.3f}, "
        f"crispiness={summary.crispiness:.3f}, "
        f"moisture={summary.moisture:.3f}, "
        f"sauce_thickness={summary.sauce_thickness:.3f}, "
        f"orange_balance={summary.orange_balance:.3f}, "
        f"sweetness={summary.sweetness:.3f}, "
        f"acidity={summary.acidity:.3f}, "
        f"coating={summary.coating_level:.3f}, "
        f"burn_risk={summary.burn_risk:.3f}, "
        f"sauce_burn_risk={summary.sauce_burn_risk:.3f}, "
        f"quality_score={summary.quality_score:.1f}, "
        f"result={summary.result}"
    )

    return summary


# ---------------------------------------------------------------------------
# AI controller
# ---------------------------------------------------------------------------

def initial_params():
    return RecipeParams(
        goal="baseline orange chicken",
        oil_target_temp_c=175.0,
        fry_time_min=6.0,
        chicken_piece_size=1.00,
        batter_thickness=0.95,
        cornstarch_ratio=0.80,
        sauce_heat=0.85,
        sauce_simmer_time_min=3.0,
        orange_juice_ml=120.0,
        sugar_g=35.0,
        soy_sauce_ml=22.0,
        vinegar_ml=14.0,
        garlic_ginger=0.55,
        toss_time_min=1.0,
        sauce_amount=0.85,
    )


def ai_controller(previous_params, summary):
    next_params = RecipeParams(**asdict(previous_params))

    goal_cycle = [
        "crispier chicken",
        "thicker sauce",
        "brighter orange flavor",
        "lower burn risk",
        "sweeter takeout style",
        "faster cooking",
        "balanced restaurant style",
    ]

    next_goal = goal_cycle[(summary.round_number - 1) % len(goal_cycle)]
    next_params.goal = next_goal

    if next_goal == "crispier chicken":
        next_params.oil_target_temp_c = 182.0
        next_params.fry_time_min = 6.6
        next_params.batter_thickness = 1.05
        next_params.cornstarch_ratio = 1.00
        next_params.sauce_amount = 0.75
        next_params.toss_time_min = 0.75

    elif next_goal == "thicker sauce":
        next_params.oil_target_temp_c = 174.0
        next_params.fry_time_min = 6.1
        next_params.sauce_heat = 1.00
        next_params.sauce_simmer_time_min = 4.4
        next_params.sugar_g = 42.0
        next_params.orange_juice_ml = 115.0
        next_params.sauce_amount = 0.92

    elif next_goal == "brighter orange flavor":
        next_params.oil_target_temp_c = 176.0
        next_params.fry_time_min = 6.0
        next_params.sauce_heat = 0.78
        next_params.sauce_simmer_time_min = 2.6
        next_params.orange_juice_ml = 165.0
        next_params.sugar_g = 34.0
        next_params.vinegar_ml = 16.0
        next_params.garlic_ginger = 0.50
        next_params.sauce_amount = 0.86

    elif next_goal == "lower burn risk":
        next_params.oil_target_temp_c = 168.0
        next_params.fry_time_min = 6.8
        next_params.sauce_heat = 0.72
        next_params.sauce_simmer_time_min = 3.2
        next_params.batter_thickness = 0.90
        next_params.cornstarch_ratio = 0.85
        next_params.sugar_g = 32.0

    elif next_goal == "sweeter takeout style":
        next_params.oil_target_temp_c = 176.0
        next_params.fry_time_min = 6.2
        next_params.sauce_heat = 0.92
        next_params.sauce_simmer_time_min = 3.4
        next_params.orange_juice_ml = 125.0
        next_params.sugar_g = 50.0
        next_params.soy_sauce_ml = 24.0
        next_params.vinegar_ml = 12.0
        next_params.sauce_amount = 0.95

    elif next_goal == "faster cooking":
        next_params.oil_target_temp_c = 186.0
        next_params.fry_time_min = 4.8
        next_params.chicken_piece_size = 0.78
        next_params.batter_thickness = 0.88
        next_params.sauce_heat = 1.05
        next_params.sauce_simmer_time_min = 2.3
        next_params.toss_time_min = 0.65

    elif next_goal == "balanced restaurant style":
        next_params.oil_target_temp_c = 178.0
        next_params.fry_time_min = 6.3
        next_params.chicken_piece_size = 0.92
        next_params.batter_thickness = 0.98
        next_params.cornstarch_ratio = 0.95
        next_params.sauce_heat = 0.88
        next_params.sauce_simmer_time_min = 3.3
        next_params.orange_juice_ml = 135.0
        next_params.sugar_g = 38.0
        next_params.soy_sauce_ml = 23.0
        next_params.vinegar_ml = 15.0
        next_params.garlic_ginger = 0.58
        next_params.toss_time_min = 0.85
        next_params.sauce_amount = 0.82

    # Reactive corrections from prior result.
    if summary.doneness < 0.95:
        next_params.fry_time_min += 0.6
        next_params.oil_target_temp_c += 2.0

    if summary.moisture < 0.38:
        next_params.fry_time_min -= 0.5
        next_params.oil_target_temp_c -= 3.0

    if summary.crispiness < 0.55:
        next_params.cornstarch_ratio += 0.10
        next_params.batter_thickness += 0.06
        next_params.oil_target_temp_c += 2.0

    if summary.burn_risk > 0.25:
        next_params.oil_target_temp_c -= 5.0
        next_params.fry_time_min -= 0.4

    if summary.sauce_thickness < 0.55:
        next_params.sauce_simmer_time_min += 0.5
        next_params.sugar_g += 4.0

    if summary.sauce_thickness > 0.90:
        next_params.sauce_simmer_time_min -= 0.6
        next_params.sauce_heat -= 0.08

    if summary.orange_balance < 0.70:
        next_params.orange_juice_ml += 12.0
        next_params.vinegar_ml += 1.0
        next_params.sugar_g = max(25.0, next_params.sugar_g - 2.0)

    if summary.coating_level < 0.65:
        next_params.sauce_amount += 0.08
        next_params.toss_time_min += 0.10

    if summary.coating_level > 0.95 or summary.crispiness < 0.45:
        next_params.sauce_amount -= 0.08
        next_params.toss_time_min -= 0.10

    # Small recipe variation.
    next_params.oil_target_temp_c += random.uniform(-1.5, 1.5)
    next_params.fry_time_min += random.uniform(-0.15, 0.15)
    next_params.orange_juice_ml *= 1.0 + random.uniform(-0.025, 0.025)
    next_params.sugar_g *= 1.0 + random.uniform(-0.025, 0.025)

    # Clamp values.
    next_params.oil_target_temp_c = clamp(next_params.oil_target_temp_c, 155.0, 195.0)
    next_params.fry_time_min = clamp(next_params.fry_time_min, 3.5, 8.5)
    next_params.chicken_piece_size = clamp(next_params.chicken_piece_size, 0.55, 1.45)
    next_params.batter_thickness = clamp(next_params.batter_thickness, 0.45, 1.45)
    next_params.cornstarch_ratio = clamp(next_params.cornstarch_ratio, 0.30, 1.30)
    next_params.sauce_heat = clamp(next_params.sauce_heat, 0.45, 1.35)
    next_params.sauce_simmer_time_min = clamp(next_params.sauce_simmer_time_min, 1.2, 6.0)
    next_params.orange_juice_ml = clamp(next_params.orange_juice_ml, 60.0, 220.0)
    next_params.sugar_g = clamp(next_params.sugar_g, 15.0, 70.0)
    next_params.soy_sauce_ml = clamp(next_params.soy_sauce_ml, 8.0, 45.0)
    next_params.vinegar_ml = clamp(next_params.vinegar_ml, 4.0, 35.0)
    next_params.garlic_ginger = clamp(next_params.garlic_ginger, 0.0, 1.2)
    next_params.toss_time_min = clamp(next_params.toss_time_min, 0.4, 2.0)
    next_params.sauce_amount = clamp(next_params.sauce_amount, 0.35, 1.20)

    print(f"AI controller selected next goal: {next_params.goal}")
    return next_params


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_comparison(summaries):
    print()
    print("=" * 132)
    print("FINAL ROUND COMPARISON")
    print("=" * 132)
    print(
        f"{'round':>5} "
        f"{'goal':<25} "
        f"{'oilC':>6} "
        f"{'fry':>5} "
        f"{'simmer':>6} "
        f"{'orange':>7} "
        f"{'sugar':>6} "
        f"{'done':>6} "
        f"{'crisp':>6} "
        f"{'moist':>6} "
        f"{'sauce':>6} "
        f"{'flavor':>7} "
        f"{'burn':>6} "
        f"{'score':>7} "
        f"{'result':<24}"
    )

    for s in summaries:
        print(
            f"{s.round_number:5d} "
            f"{s.goal:<25.25} "
            f"{s.oil_temp_c:6.1f} "
            f"{s.fry_time_min:5.1f} "
            f"{s.sauce_simmer_time_min:6.1f} "
            f"{s.orange_juice_ml:7.0f} "
            f"{s.sugar_g:6.0f} "
            f"{s.doneness:6.2f} "
            f"{s.crispiness:6.2f} "
            f"{s.moisture:6.2f} "
            f"{s.sauce_thickness:6.2f} "
            f"{s.orange_balance:7.2f} "
            f"{s.burn_risk:6.2f} "
            f"{s.quality_score:7.1f} "
            f"{s.result:<24}"
        )

    best = max(summaries, key=lambda s: s.quality_score)
    crispiest = max(summaries, key=lambda s: s.crispiness)
    best_flavor = max(summaries, key=lambda s: s.orange_balance)
    lowest_burn = min(summaries, key=lambda s: s.burn_risk + s.sauce_burn_risk)

    print("-" * 132)
    print(f"best overall: round {best.round_number} ({best.quality_score:.1f}, goal={best.goal})")
    print(f"crispiest: round {crispiest.round_number} (crispiness={crispiest.crispiness:.2f}, goal={crispiest.goal})")
    print(f"best flavor balance: round {best_flavor.round_number} (flavor={best_flavor.orange_balance:.2f}, goal={best_flavor.goal})")
    print(
        f"lowest burn risk: round {lowest_burn.round_number} "
        f"(burn={lowest_burn.burn_risk + lowest_burn.sauce_burn_risk:.2f}, goal={lowest_burn.goal})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("Terminal Orange Chicken Cooking Simulation with AI-Controlled Rounds")
    print("No graphics. No external packages. State is printed directly to the terminal.")
    print("Toy model only. This is not food-safety guidance.")
    print(
        "The model includes chicken temperature, doneness, moisture, crispiness, oil heat, "
        "sauce thickness, orange flavor, sweetness, acidity, coating, burn risk, and quality."
    )

    params = initial_params()
    summaries = []

    for round_number in range(1, ROUNDS + 1):
        summary = simulate_round(round_number, params)
        summaries.append(summary)

        if round_number < ROUNDS:
            params = ai_controller(params, summary)

    print_final_comparison(summaries)


if __name__ == "__main__":
    main()
