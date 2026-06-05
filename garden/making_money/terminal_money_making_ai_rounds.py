"""
Terminal Money-Making Simulation with AI-Controlled Rounds

A Python-only terminal simulation of a person/business trying to grow money.

Features:
- No external packages
- Multiple rounds of money-making strategy
- AI controller changes strategy after each round
- Simulates cash, income, expenses, skill, reputation, customers, energy, risk, inventory, debt, marketing, and automation
- Prints state directly to the terminal
- Ends with a final comparison table

Run:
    python terminal_money_making_ai_rounds.py

Note:
    This is a toy simulation, not financial advice.
"""

import random
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# User-adjustable settings
# ---------------------------------------------------------------------------

RANDOM_SEED = 21
ROUNDS = 10
DAYS_PER_ROUND = 7

STARTING_CASH = 120.0
STARTING_DEBT = 0.0
BASE_LIVING_COST_PER_WEEK = 45.0

MAX_ENERGY = 100.0
MIN_CASH_BEFORE_DEBT = -100.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MoneyParams:
    goal: str
    work_hours: float
    learning_hours: float
    marketing_spend: float
    inventory_spend: float
    tool_spend: float
    quality_focus: float
    risk_level: float
    price_multiplier: float
    automation_investment: float
    hire_help: bool
    debt_allowed: bool


@dataclass
class MoneyState:
    cash: float
    debt: float
    skill: float
    reputation: float
    customers: int
    inventory_units: int
    energy: float
    automation: float
    business_value: float
    week: int


@dataclass
class RoundSummary:
    round_number: int
    goal: str
    starting_cash: float
    income: float
    expenses: float
    profit: float
    ending_cash: float
    debt: float
    skill: float
    reputation: float
    customers: int
    inventory_units: int
    energy: float
    automation: float
    business_value: float
    risk_event: str
    result: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def money(value):
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def bar(value, max_value, width=18):
    if max_value <= 0:
        filled = 0
    else:
        filled = int(width * clamp(value / max_value, 0.0, 1.0))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_header(round_number, state, params):
    print()
    print("=" * 112)
    print(f"ROUND {round_number}")
    print("-" * 112)
    print(f"goal: {params.goal}")
    print(
        "starting state: "
        f"cash={money(state.cash)}, "
        f"debt={money(state.debt)}, "
        f"skill={state.skill:.2f}, "
        f"reputation={state.reputation:.2f}, "
        f"customers={state.customers}, "
        f"inventory={state.inventory_units}, "
        f"energy={state.energy:.1f}, "
        f"automation={state.automation:.2f}, "
        f"business_value={money(state.business_value)}"
    )
    print(
        "strategy: "
        f"work_hours={params.work_hours:.1f}, "
        f"learning_hours={params.learning_hours:.1f}, "
        f"marketing={money(params.marketing_spend)}, "
        f"inventory_spend={money(params.inventory_spend)}, "
        f"tools={money(params.tool_spend)}, "
        f"quality={params.quality_focus:.2f}, "
        f"risk={params.risk_level:.2f}, "
        f"price_x={params.price_multiplier:.2f}, "
        f"automation_invest={money(params.automation_investment)}, "
        f"hire_help={params.hire_help}"
    )
    print("-" * 112)


def print_state_line(day, state, day_income, day_expense, day_profit, activity):
    print(
        f"day={day:02d}  "
        f"cash={money(state.cash):>10} {bar(max(state.cash, 0), 1000, 12)}  "
        f"income={money(day_income):>9}  "
        f"expense={money(day_expense):>9}  "
        f"profit={money(day_profit):>9}  "
        f"skill={state.skill:5.2f}  "
        f"rep={state.reputation:5.2f}  "
        f"cust={state.customers:3d}  "
        f"inv={state.inventory_units:3d}  "
        f"energy={state.energy:5.1f} {bar(state.energy, MAX_ENERGY, 10)}  "
        f"auto={state.automation:4.2f}  "
        f"activity={activity}"
    )


def classify_result(profit, state, risk_event):
    if state.cash <= MIN_CASH_BEFORE_DEBT and state.debt > 0:
        return "cash stress"
    if profit > 350:
        return "strong profitable round"
    if profit > 125:
        return "profitable round"
    if profit > 0:
        return "small profitable round"
    if risk_event != "none" and profit < 0:
        return "risk setback"
    if profit < -150:
        return "large loss"
    return "small loss"


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def initial_state():
    return MoneyState(
        cash=STARTING_CASH,
        debt=STARTING_DEBT,
        skill=1.00,
        reputation=0.20,
        customers=3,
        inventory_units=0,
        energy=95.0,
        automation=0.00,
        business_value=100.0,
        week=0,
    )


def initial_params():
    return MoneyParams(
        goal="baseline freelance work",
        work_hours=28.0,
        learning_hours=4.0,
        marketing_spend=20.0,
        inventory_spend=0.0,
        tool_spend=10.0,
        quality_focus=0.65,
        risk_level=0.15,
        price_multiplier=1.00,
        automation_investment=0.0,
        hire_help=False,
        debt_allowed=False,
    )


def simulate_round(round_number, state, params):
    print_header(round_number, state, params)

    starting_cash = state.cash
    total_income = 0.0
    total_expenses = 0.0
    risk_event = "none"

    # Up-front expenses.
    upfront_expense = (
        BASE_LIVING_COST_PER_WEEK
        + params.marketing_spend
        + params.inventory_spend
        + params.tool_spend
        + params.automation_investment
    )

    if params.hire_help:
        upfront_expense += 80.0

    state.cash -= upfront_expense
    total_expenses += upfront_expense

    # Debt behavior.
    if state.cash < 0 and params.debt_allowed:
        borrowed = abs(state.cash) + 50.0
        state.debt += borrowed
        state.cash += borrowed
    elif state.cash < MIN_CASH_BEFORE_DEBT:
        state.debt += abs(state.cash - MIN_CASH_BEFORE_DEBT)
        state.cash = MIN_CASH_BEFORE_DEBT

    # Investments affect capabilities.
    state.skill += 0.004 * params.tool_spend
    state.automation += 0.0025 * params.automation_investment
    state.automation = clamp(state.automation, 0.0, 1.0)

    purchased_units = int(params.inventory_spend / 12.0)
    state.inventory_units += purchased_units

    # Marketing can attract customers, but depends on reputation.
    marketing_customers = int((params.marketing_spend / 18.0) * (0.7 + state.reputation))
    state.customers += max(0, marketing_customers)

    print_state_line(
        0,
        state,
        0.0,
        upfront_expense,
        -upfront_expense,
        "paid weekly costs/investments",
    )

    # Spread weekly strategy over days.
    daily_work_hours = params.work_hours / DAYS_PER_ROUND
    daily_learning_hours = params.learning_hours / DAYS_PER_ROUND

    for day in range(1, DAYS_PER_ROUND + 1):
        day_income = 0.0
        day_expense = 0.0
        activity = "work"

        # Energy affects productivity.
        energy_multiplier = clamp(state.energy / MAX_ENERGY, 0.25, 1.10)
        automation_multiplier = 1.0 + 0.75 * state.automation
        helper_multiplier = 1.25 if params.hire_help else 1.0

        # Learning improves skill but costs energy and time.
        learning_gain = daily_learning_hours * 0.018 * (1.0 + params.quality_focus)
        state.skill += learning_gain

        # Main service/freelance income.
        demand = 0.65 + state.reputation + 0.035 * state.customers
        hourly_value = 10.0 * state.skill * demand * params.price_multiplier
        service_income = daily_work_hours * hourly_value * energy_multiplier * automation_multiplier * helper_multiplier

        # Higher price may reduce conversion if reputation is low.
        price_resistance = max(0.0, params.price_multiplier - 1.0) * max(0.0, 0.75 - state.reputation)
        service_income *= (1.0 - clamp(price_resistance, 0.0, 0.45))

        day_income += service_income

        # Product/inventory sales.
        if state.inventory_units > 0:
            likely_units = int(random.uniform(0.0, 1.0) * (1 + state.customers * 0.12 + state.reputation))
            units_sold = min(state.inventory_units, max(0, likely_units))
            product_revenue = units_sold * 24.0 * params.price_multiplier
            state.inventory_units -= units_sold
            day_income += product_revenue
            if units_sold > 0:
                activity = f"work + sold {units_sold} units"

        # Quality affects reputation.
        delivery_quality = (
            0.45 * params.quality_focus
            + 0.25 * clamp(state.skill / 5.0, 0.0, 1.0)
            + 0.20 * energy_multiplier
            + 0.10 * state.automation
        )
        reputation_gain = 0.012 * delivery_quality * max(1.0, daily_work_hours / 3.0)
        state.reputation += reputation_gain

        # Some customers are gained through successful work.
        if random.random() < clamp(state.reputation * 0.25 + params.quality_focus * 0.10, 0.02, 0.45):
            state.customers += 1

        # Energy use.
        energy_used = daily_work_hours * 2.1 + daily_learning_hours * 1.3
        if params.hire_help:
            energy_used *= 0.82
        state.energy -= energy_used

        # Partial overnight recovery.
        state.energy += 7.5
        state.energy = clamp(state.energy, 0.0, MAX_ENERGY)

        # Risk event.
        risk_probability = 0.04 + params.risk_level * 0.16
        if random.random() < risk_probability:
            event_roll = random.random()
            if event_roll < 0.35:
                penalty = random.uniform(25.0, 120.0) * (0.5 + params.risk_level)
                day_expense += penalty
                risk_event = "unexpected expense"
                activity += " | risk: expense"
            elif event_roll < 0.70:
                missed_income = day_income * random.uniform(0.20, 0.60)
                day_income -= missed_income
                risk_event = "missed opportunity"
                activity += " | risk: missed income"
            else:
                bonus = random.uniform(50.0, 180.0) * (0.5 + params.risk_level)
                day_income += bonus
                risk_event = "lucky opportunity"
                activity += " | risk: bonus"

        # Debt interest.
        if state.debt > 0:
            interest = state.debt * 0.008
            day_expense += interest
            state.debt += interest

        day_profit = day_income - day_expense
        state.cash += day_profit
        total_income += day_income
        total_expenses += day_expense

        # Borrow if allowed and cash is below zero.
        if state.cash < 0 and params.debt_allowed:
            borrowed = abs(state.cash) + 30.0
            state.debt += borrowed
            state.cash += borrowed

        # Business value changes with skill, reputation, customers, automation, and cash.
        state.business_value = (
            75.0
            + state.cash * 0.25
            + state.skill * 40.0
            + state.reputation * 220.0
            + state.customers * 8.0
            + state.automation * 350.0
            + state.inventory_units * 5.0
            - state.debt * 0.35
        )
        state.business_value = max(0.0, state.business_value)

        print_state_line(day, state, day_income, day_expense, day_profit, activity)

    # End-of-round cleanup.
    state.skill = clamp(state.skill, 0.1, 10.0)
    state.reputation = clamp(state.reputation, 0.0, 1.0)
    state.customers = max(0, state.customers)
    state.week += 1

    profit = total_income - total_expenses
    result = classify_result(profit, state, risk_event)

    summary = RoundSummary(
        round_number=round_number,
        goal=params.goal,
        starting_cash=starting_cash,
        income=total_income,
        expenses=total_expenses,
        profit=profit,
        ending_cash=state.cash,
        debt=state.debt,
        skill=state.skill,
        reputation=state.reputation,
        customers=state.customers,
        inventory_units=state.inventory_units,
        energy=state.energy,
        automation=state.automation,
        business_value=state.business_value,
        risk_event=risk_event,
        result=result,
    )

    print("-" * 112)
    print(
        f"summary: income={money(summary.income)}, "
        f"expenses={money(summary.expenses)}, "
        f"profit={money(summary.profit)}, "
        f"cash={money(summary.ending_cash)}, "
        f"debt={money(summary.debt)}, "
        f"skill={summary.skill:.2f}, "
        f"reputation={summary.reputation:.2f}, "
        f"customers={summary.customers}, "
        f"inventory={summary.inventory_units}, "
        f"energy={summary.energy:.1f}, "
        f"automation={summary.automation:.2f}, "
        f"business_value={money(summary.business_value)}, "
        f"risk_event={summary.risk_event}, "
        f"result={summary.result}"
    )

    return summary


# ---------------------------------------------------------------------------
# AI controller
# ---------------------------------------------------------------------------

def ai_controller(previous_params, summary):
    next_params = MoneyParams(**asdict(previous_params))

    goal_cycle = [
        "skill-building week",
        "product sales test",
        "marketing push",
        "premium pricing test",
        "high-risk opportunity",
        "hire help",
        "automation investment",
        "stabilize income",
        "scale balanced system",
    ]

    next_goal = goal_cycle[(summary.round_number - 1) % len(goal_cycle)]
    next_params.goal = next_goal

    if next_goal == "skill-building week":
        next_params.work_hours = 18.0
        next_params.learning_hours = 18.0
        next_params.marketing_spend = 10.0
        next_params.inventory_spend = 0.0
        next_params.tool_spend = 60.0
        next_params.quality_focus = 0.85
        next_params.risk_level = 0.10
        next_params.price_multiplier = 1.00
        next_params.automation_investment = 0.0
        next_params.hire_help = False
        next_params.debt_allowed = False

    elif next_goal == "product sales test":
        next_params.work_hours = 22.0
        next_params.learning_hours = 5.0
        next_params.marketing_spend = 35.0
        next_params.inventory_spend = 120.0
        next_params.tool_spend = 10.0
        next_params.quality_focus = 0.68
        next_params.risk_level = 0.20
        next_params.price_multiplier = 1.05
        next_params.automation_investment = 0.0
        next_params.hire_help = False
        next_params.debt_allowed = True

    elif next_goal == "marketing push":
        next_params.work_hours = 30.0
        next_params.learning_hours = 4.0
        next_params.marketing_spend = 110.0
        next_params.inventory_spend = 70.0
        next_params.tool_spend = 15.0
        next_params.quality_focus = 0.72
        next_params.risk_level = 0.22
        next_params.price_multiplier = 1.08
        next_params.automation_investment = 0.0
        next_params.hire_help = False
        next_params.debt_allowed = True

    elif next_goal == "premium pricing test":
        next_params.work_hours = 26.0
        next_params.learning_hours = 8.0
        next_params.marketing_spend = 55.0
        next_params.inventory_spend = 20.0
        next_params.tool_spend = 25.0
        next_params.quality_focus = 0.92
        next_params.risk_level = 0.18
        next_params.price_multiplier = 1.35
        next_params.automation_investment = 0.0
        next_params.hire_help = False
        next_params.debt_allowed = True

    elif next_goal == "high-risk opportunity":
        next_params.work_hours = 34.0
        next_params.learning_hours = 2.0
        next_params.marketing_spend = 80.0
        next_params.inventory_spend = 160.0
        next_params.tool_spend = 10.0
        next_params.quality_focus = 0.55
        next_params.risk_level = 0.65
        next_params.price_multiplier = 1.18
        next_params.automation_investment = 0.0
        next_params.hire_help = False
        next_params.debt_allowed = True

    elif next_goal == "hire help":
        next_params.work_hours = 38.0
        next_params.learning_hours = 3.0
        next_params.marketing_spend = 95.0
        next_params.inventory_spend = 100.0
        next_params.tool_spend = 20.0
        next_params.quality_focus = 0.70
        next_params.risk_level = 0.30
        next_params.price_multiplier = 1.12
        next_params.automation_investment = 0.0
        next_params.hire_help = True
        next_params.debt_allowed = True

    elif next_goal == "automation investment":
        next_params.work_hours = 24.0
        next_params.learning_hours = 8.0
        next_params.marketing_spend = 60.0
        next_params.inventory_spend = 50.0
        next_params.tool_spend = 30.0
        next_params.quality_focus = 0.78
        next_params.risk_level = 0.24
        next_params.price_multiplier = 1.15
        next_params.automation_investment = 180.0
        next_params.hire_help = False
        next_params.debt_allowed = True

    elif next_goal == "stabilize income":
        next_params.work_hours = 28.0
        next_params.learning_hours = 6.0
        next_params.marketing_spend = 35.0
        next_params.inventory_spend = 40.0
        next_params.tool_spend = 15.0
        next_params.quality_focus = 0.88
        next_params.risk_level = 0.08
        next_params.price_multiplier = 1.10
        next_params.automation_investment = 40.0
        next_params.hire_help = False
        next_params.debt_allowed = False

    elif next_goal == "scale balanced system":
        next_params.work_hours = 34.0
        next_params.learning_hours = 7.0
        next_params.marketing_spend = 130.0
        next_params.inventory_spend = 130.0
        next_params.tool_spend = 50.0
        next_params.quality_focus = 0.82
        next_params.risk_level = 0.20
        next_params.price_multiplier = 1.18
        next_params.automation_investment = 120.0
        next_params.hire_help = True
        next_params.debt_allowed = True

    # Reactive corrections based on the previous round.
    if summary.profit < 0:
        next_params.marketing_spend *= 0.80
        next_params.inventory_spend *= 0.75
        next_params.risk_level *= 0.75
        next_params.quality_focus = min(1.0, next_params.quality_focus + 0.08)

    if summary.energy < 35:
        next_params.work_hours *= 0.82
        next_params.learning_hours *= 0.90
        next_params.hire_help = True

    if summary.reputation < 0.35:
        next_params.quality_focus = min(1.0, next_params.quality_focus + 0.10)
        next_params.price_multiplier = min(next_params.price_multiplier, 1.12)

    if summary.debt > 400:
        next_params.debt_allowed = False
        next_params.inventory_spend *= 0.50
        next_params.marketing_spend *= 0.65
        next_params.automation_investment *= 0.50
        next_params.risk_level *= 0.50

    if summary.customers < 6:
        next_params.marketing_spend += 25.0

    if summary.profit > 250:
        next_params.marketing_spend *= 1.10
        next_params.tool_spend *= 1.10

    # Small controlled variation.
    next_params.work_hours *= 1.0 + random.uniform(-0.04, 0.04)
    next_params.marketing_spend *= 1.0 + random.uniform(-0.05, 0.05)
    next_params.inventory_spend *= 1.0 + random.uniform(-0.05, 0.05)
    next_params.price_multiplier *= 1.0 + random.uniform(-0.025, 0.025)

    # Clamps.
    next_params.work_hours = clamp(next_params.work_hours, 0.0, 55.0)
    next_params.learning_hours = clamp(next_params.learning_hours, 0.0, 30.0)
    next_params.marketing_spend = clamp(next_params.marketing_spend, 0.0, 400.0)
    next_params.inventory_spend = clamp(next_params.inventory_spend, 0.0, 500.0)
    next_params.tool_spend = clamp(next_params.tool_spend, 0.0, 300.0)
    next_params.quality_focus = clamp(next_params.quality_focus, 0.0, 1.0)
    next_params.risk_level = clamp(next_params.risk_level, 0.0, 1.0)
    next_params.price_multiplier = clamp(next_params.price_multiplier, 0.65, 2.00)
    next_params.automation_investment = clamp(next_params.automation_investment, 0.0, 600.0)

    print(f"AI controller selected next goal: {next_params.goal}")
    return next_params


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_comparison(summaries):
    print()
    print("=" * 140)
    print("FINAL ROUND COMPARISON")
    print("=" * 140)
    print(
        f"{'round':>5} "
        f"{'goal':<23} "
        f"{'start':>10} "
        f"{'income':>10} "
        f"{'expenses':>10} "
        f"{'profit':>10} "
        f"{'cash':>10} "
        f"{'debt':>10} "
        f"{'skill':>6} "
        f"{'rep':>5} "
        f"{'cust':>5} "
        f"{'energy':>7} "
        f"{'auto':>5} "
        f"{'value':>10} "
        f"{'result':<24}"
    )

    for s in summaries:
        print(
            f"{s.round_number:5d} "
            f"{s.goal:<23.23} "
            f"{money(s.starting_cash):>10} "
            f"{money(s.income):>10} "
            f"{money(s.expenses):>10} "
            f"{money(s.profit):>10} "
            f"{money(s.ending_cash):>10} "
            f"{money(s.debt):>10} "
            f"{s.skill:6.2f} "
            f"{s.reputation:5.2f} "
            f"{s.customers:5d} "
            f"{s.energy:7.1f} "
            f"{s.automation:5.2f} "
            f"{money(s.business_value):>10} "
            f"{s.result:<24}"
        )

    best_profit = max(summaries, key=lambda s: s.profit)
    best_cash = max(summaries, key=lambda s: s.ending_cash)
    best_value = max(summaries, key=lambda s: s.business_value)
    worst_loss = min(summaries, key=lambda s: s.profit)

    print("-" * 140)
    print(
        f"best profit: round {best_profit.round_number} "
        f"({money(best_profit.profit)}, goal={best_profit.goal})"
    )
    print(
        f"highest ending cash: round {best_cash.round_number} "
        f"({money(best_cash.ending_cash)}, goal={best_cash.goal})"
    )
    print(
        f"highest business value: round {best_value.round_number} "
        f"({money(best_value.business_value)}, goal={best_value.goal})"
    )
    print(
        f"worst loss: round {worst_loss.round_number} "
        f"({money(worst_loss.profit)}, goal={worst_loss.goal})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("Terminal Money-Making Simulation with AI-Controlled Rounds")
    print("No graphics. No external packages. State is printed directly to the terminal.")
    print("Toy model only. This is not financial advice.")
    print(
        "The model includes cash, debt, income, expenses, skill, reputation, customers, "
        "inventory, energy, risk, automation, and business value."
    )

    state = initial_state()
    params = initial_params()
    summaries = []

    for round_number in range(1, ROUNDS + 1):
        summary = simulate_round(round_number, state, params)
        summaries.append(summary)

        if round_number < ROUNDS:
            params = ai_controller(params, summary)

    print_final_comparison(summaries)


if __name__ == "__main__":
    main()
