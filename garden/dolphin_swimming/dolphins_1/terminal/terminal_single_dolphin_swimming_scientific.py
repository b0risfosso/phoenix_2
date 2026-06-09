#!/usr/bin/env python3
"""
Terminal-only scientific model of a single dolphin swimming in place.

This is the non-visual counterpart of the VPython single-dolphin swimming script.
It preserves the mode schedule, sinusoidal body/tail kinematics, yaw/pitch/roll
commands, thrust-drag balance, virtual distance, and energy accounting.

Run:
    python terminal_single_dolphin_swimming_scientific.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sin, cos, pi, radians, degrees


@dataclass
class SwimMode:
    name: str
    duration: float
    tail_hz: float
    tail_amp_deg: float
    body_amp_deg: float
    wavelength_body_lengths: float
    yaw_bias_deg: float
    pitch_bias_deg: float
    roll_amp_deg: float
    target_speed: float
    notes: str


@dataclass
class DolphinState:
    t: float = 0.0
    mode_index: int = 0
    phase: float = 0.0
    virtual_distance: float = 0.0
    speed: float = 1.2
    accel: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    depth_m: float = 0.0
    turn_rate_deg_s: float = 0.0
    vertical_speed_m_s: float = 0.0
    thrust_n: float = 0.0
    drag_n: float = 0.0
    body_curvature: float = 0.0
    tail_angle_deg: float = 0.0
    fluke_lift_index: float = 0.0
    energy_j: float = 0.0


BODY_LENGTH_M = 2.55
MASS_KG = 180.0
WATER_DENSITY_KG_M3 = 1025.0
REFERENCE_AREA_M2 = 0.22
DRAG_COEFF = 0.045
PROPULSIVE_EFFICIENCY = 0.72
TAIL_GAIN = 128.0
BODY_CURVATURE_GAIN = 0.38
PITCH_ASCENT_GAIN = 0.18
YAW_TURN_GAIN = 0.42

MODES = [
    SwimMode("steady cruising", 8.0, 1.15, 16.0, 7.0, 1.35, 0.0, 0.0, 4.0, 2.2, "balanced thrust and drag"),
    SwimMode("forward acceleration", 7.0, 1.85, 24.0, 10.0, 1.10, 0.0, 1.0, 5.0, 4.0, "larger tail beat, rising speed"),
    SwimMode("left turn", 7.0, 1.35, 20.0, 11.0, 1.05, 18.0, 0.0, 14.0, 2.5, "yaw-biased body bend"),
    SwimMode("ascending glide-pump", 8.0, 1.05, 15.0, 6.0, 1.45, 0.0, 16.0, 4.0, 2.1, "positive pitch for ascent"),
    SwimMode("barrel roll spin", 7.0, 1.45, 19.0, 8.5, 1.20, 5.0, 2.0, 85.0, 2.7, "roll-dominant spin posture"),
    SwimMode("right recovery turn", 7.0, 1.20, 18.0, 9.0, 1.25, -16.0, -6.0, 10.0, 2.0, "opposite yaw and slight descent"),
]


def active_mode(t: float):
    cycle = sum(m.duration for m in MODES)
    tc = t % cycle
    acc = 0.0
    for i, m in enumerate(MODES):
        if acc <= tc < acc + m.duration:
            return i, m, tc - acc
        acc += m.duration
    return len(MODES) - 1, MODES[-1], MODES[-1].duration


def smooth_transition(x: float, duration: float) -> float:
    edge = min(1.5, duration * 0.25)
    rise = min(1.0, x / edge)
    fall = min(1.0, (duration - x) / edge)
    return max(0.0, min(rise, fall))


def step(state: DolphinState, dt: float) -> DolphinState:
    mode_i, mode, local_t = active_mode(state.t)
    env = smooth_transition(local_t, mode.duration)
    omega = 2.0 * pi * mode.tail_hz
    state.phase += omega * dt

    body_amp = radians(mode.body_amp_deg)
    tail_amp = radians(mode.tail_amp_deg)
    wavelength_m = BODY_LENGTH_M * mode.wavelength_body_lengths
    wave_number = 2.0 * pi / wavelength_m

    curvature_wave = body_amp * sin(state.phase - wave_number * BODY_LENGTH_M * 0.65)
    steering_curve = radians(mode.yaw_bias_deg) * 0.018 * env
    state.body_curvature = BODY_CURVATURE_GAIN * (curvature_wave + steering_curve)
    state.tail_angle_deg = degrees(tail_amp * sin(state.phase + 0.55))

    tail_amp_factor = abs(sin(state.phase + 0.55))
    coordination = 0.70 + 0.30 * cos(curvature_wave - tail_amp * sin(state.phase + 0.55))
    raw_thrust = TAIL_GAIN * (mode.tail_hz ** 2) * (tail_amp ** 2) * (0.8 + 0.2 * state.speed) * coordination
    state.thrust_n = raw_thrust * (0.55 + 0.45 * tail_amp_factor)
    state.drag_n = 0.5 * WATER_DENSITY_KG_M3 * DRAG_COEFF * REFERENCE_AREA_M2 * state.speed ** 2

    control_accel = 0.55 * (mode.target_speed - state.speed)
    hydrodynamic_accel = (state.thrust_n - state.drag_n) / MASS_KG
    state.accel = hydrodynamic_accel + control_accel
    state.speed = max(0.05, state.speed + state.accel * dt)
    state.virtual_distance += state.speed * dt

    yaw_command = mode.yaw_bias_deg * env + 3.5 * sin(state.phase * 0.5)
    pitch_command = mode.pitch_bias_deg * env + 2.0 * sin(state.phase * 0.33)
    roll_command = mode.roll_amp_deg * sin(state.phase * 0.50) * env
    tau = 0.65
    state.yaw_deg += (yaw_command - state.yaw_deg) * dt / tau
    state.pitch_deg += (pitch_command - state.pitch_deg) * dt / tau
    state.roll_deg += (roll_command - state.roll_deg) * dt / tau

    state.turn_rate_deg_s = YAW_TURN_GAIN * state.speed * state.yaw_deg
    state.vertical_speed_m_s = PITCH_ASCENT_GAIN * state.speed * sin(radians(state.pitch_deg)) * 10.0
    state.depth_m = max(0.0, state.depth_m - state.vertical_speed_m_s * dt)
    state.fluke_lift_index = mode.tail_hz * sin(radians(state.tail_angle_deg)) * (1.0 + 0.15 * state.speed)

    power_w = max(0.0, state.thrust_n * state.speed / PROPULSIVE_EFFICIENCY)
    state.energy_j += power_w * dt
    state.t += dt
    state.mode_index = mode_i
    return state


def print_header():
    print("Single Dolphin Terminal Scientific Simulation")
    print("Model: sinusoidal body wave + tail thrust + quadratic drag + first-order orientation control")
    print("Equations: D=0.5*rho*Cd*A*v^2 | a=(T-D)/m + k(v_target-v) | power=max(T*v/eta,0)")
    print("-" * 132)
    print(
        "time  mode                     speed  accel   thrust   drag    tail   curve    yaw   pitch   roll   turn    v_z    dist   energy"
    )
    print(
        "s                              m/s    m/s^2   N        N       deg    1/L      deg   deg     deg    deg/s   m/s    m      kJ"
    )
    print("-" * 132)


def run(total_seconds: float = 88.0, dt: float = 0.025, print_every: float = 1.0):
    state = DolphinState()
    print_header()
    next_print = 0.0
    last_mode = None
    mode_stats = {m.name: {"n": 0, "speed": 0.0, "thrust": 0.0, "drag": 0.0, "energy0": None, "energy1": 0.0} for m in MODES}

    while state.t < total_seconds:
        step(state, dt)
        mode_i, mode, _ = active_mode(state.t)
        if mode.name != last_mode:
            print(f"\nMODE CHANGE -> {mode_i + 1}: {mode.name} | {mode.notes}")
            last_mode = mode.name
        st = mode_stats[mode.name]
        st["n"] += 1
        st["speed"] += state.speed
        st["thrust"] += state.thrust_n
        st["drag"] += state.drag_n
        if st["energy0"] is None:
            st["energy0"] = state.energy_j
        st["energy1"] = state.energy_j
        if state.t + 1e-9 >= next_print:
            print(
                f"{state.t:5.1f}  {mode.name:<23} {state.speed:5.2f}  {state.accel:6.2f}  "
                f"{state.thrust_n:7.1f}  {state.drag_n:6.1f}  {state.tail_angle_deg:6.1f}  "
                f"{state.body_curvature:7.3f}  {state.yaw_deg:5.1f}  {state.pitch_deg:6.1f}  "
                f"{state.roll_deg:6.1f}  {state.turn_rate_deg_s:6.1f}  {state.vertical_speed_m_s:6.2f}  "
                f"{state.virtual_distance:6.1f}  {state.energy_j/1000.0:7.2f}"
            )
            next_print += print_every

    print("\nMode summary")
    print("-" * 86)
    print("mode                     mean_speed  mean_thrust  mean_drag  energy_used_kJ")
    for name, st in mode_stats.items():
        if st["n"] == 0:
            continue
        print(
            f"{name:<24} {st['speed']/st['n']:10.2f}  {st['thrust']/st['n']:11.1f}  "
            f"{st['drag']/st['n']:9.1f}  {(st['energy1'] - (st['energy0'] or 0.0))/1000.0:14.2f}"
        )
    print(f"\nFinal virtual distance: {state.virtual_distance:.1f} m")
    print(f"Total mechanical energy estimate: {state.energy_j/1000.0:.2f} kJ")


if __name__ == "__main__":
    run()
