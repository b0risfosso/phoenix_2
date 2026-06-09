#!/usr/bin/env python3
"""
VPython visual simulation of a single dolphin swimming in place.

The visual model matches the terminal simulation's mode schedule and kinematics:
- body stays centered in the scene
- axial body points bend in a sinusoidal travelling wave
- tail/flukes oscillate for propulsion
- yaw, pitch, and roll show turning, ascent, descent, and spin intent
- ghost streamlines move backward to make the stationary dolphin look as if it is swimming forward

Run with:
    python vpython_single_dolphin_swimming_simulation.py

Requires:
    pip install vpython
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sin, cos, pi, radians, degrees
from vpython import (
    canvas,
    vector,
    color,
    rate,
    sphere,
    ellipsoid,
    cone,
    cylinder,
    curve,
    triangle,
    vertex,
    label,
    arrow,
    box,
    local_light,
)


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


def active_mode(t):
    cycle = sum(m.duration for m in MODES)
    tc = t % cycle
    acc = 0.0
    for i, m in enumerate(MODES):
        if acc <= tc < acc + m.duration:
            return i, m, tc - acc
        acc += m.duration
    return len(MODES) - 1, MODES[-1], MODES[-1].duration


def smooth_transition(x, duration):
    edge = min(1.5, duration * 0.25)
    rise = min(1.0, x / edge)
    fall = min(1.0, (duration - x) / edge)
    return max(0.0, min(rise, fall))


def step(state, dt):
    mode_i, mode, local_t = active_mode(state.t)
    env = smooth_transition(local_t, mode.duration)
    omega = 2.0 * pi * mode.tail_hz
    state.phase += omega * dt
    phase = state.phase

    body_amp = radians(mode.body_amp_deg)
    tail_amp = radians(mode.tail_amp_deg)
    wavelength_m = BODY_LENGTH_M * mode.wavelength_body_lengths
    wave_number = 2.0 * pi / wavelength_m

    curvature_wave = body_amp * sin(phase - wave_number * BODY_LENGTH_M * 0.65)
    steering_curve = radians(mode.yaw_bias_deg) * 0.018 * env
    state.body_curvature = BODY_CURVATURE_GAIN * (curvature_wave + steering_curve)
    state.tail_angle_deg = degrees(tail_amp * sin(phase + 0.55))

    tail_amp_factor = abs(sin(phase + 0.55))
    coordination = 0.70 + 0.30 * cos(curvature_wave - tail_amp * sin(phase + 0.55))
    raw_thrust = TAIL_GAIN * (mode.tail_hz ** 2) * (tail_amp ** 2) * (0.8 + 0.2 * state.speed) * coordination
    state.thrust_n = raw_thrust * (0.55 + 0.45 * tail_amp_factor)
    state.drag_n = 0.5 * WATER_DENSITY_KG_M3 * DRAG_COEFF * REFERENCE_AREA_M2 * state.speed ** 2

    control_accel = 0.55 * (mode.target_speed - state.speed)
    hydrodynamic_accel = (state.thrust_n - state.drag_n) / MASS_KG
    state.accel = hydrodynamic_accel + control_accel
    state.speed = max(0.05, state.speed + state.accel * dt)
    state.virtual_distance += state.speed * dt

    yaw_command = mode.yaw_bias_deg * env + 3.5 * sin(phase * 0.5)
    pitch_command = mode.pitch_bias_deg * env + 2.0 * sin(phase * 0.33)
    roll_command = mode.roll_amp_deg * sin(phase * 0.50) * env

    orient_tau = 0.65
    state.yaw_deg += (yaw_command - state.yaw_deg) * dt / orient_tau
    state.pitch_deg += (pitch_command - state.pitch_deg) * dt / orient_tau
    state.roll_deg += (roll_command - state.roll_deg) * dt / orient_tau

    state.turn_rate_deg_s = YAW_TURN_GAIN * state.speed * state.yaw_deg
    state.vertical_speed_m_s = PITCH_ASCENT_GAIN * state.speed * sin(radians(state.pitch_deg)) * 10.0
    state.depth_m -= state.vertical_speed_m_s * dt
    state.fluke_lift_index = mode.tail_hz * sin(radians(state.tail_angle_deg)) * (1.0 + 0.15 * state.speed)

    power_w = max(0.0, state.thrust_n * state.speed / PROPULSIVE_EFFICIENCY)
    state.energy_j += power_w * dt
    state.t += dt
    state.mode_index = mode_i
    return state


def rotate_yz(y, z, angle_rad):
    ca, sa = cos(angle_rad), sin(angle_rad)
    return y * ca - z * sa, y * sa + z * ca


def body_points(state, count=13):
    """Return body centerline points from head (+x) to tail (-x)."""
    mode_i, mode, local_t = active_mode(state.t)
    phase = state.phase
    body_amp = radians(mode.body_amp_deg)
    wavelength_m = BODY_LENGTH_M * mode.wavelength_body_lengths
    wave_number = 2.0 * pi / wavelength_m
    yaw_bias = radians(mode.yaw_bias_deg) * 0.06 * smooth_transition(local_t, mode.duration)
    pitch_bias = radians(mode.pitch_bias_deg) * 0.05 * smooth_transition(local_t, mode.duration)
    roll = radians(state.roll_deg)

    pts = []
    for i in range(count):
        u = i / (count - 1)
        x = BODY_LENGTH_M * (0.50 - u)
        tail_weight = u ** 1.55
        y = tail_weight * (0.36 * sin(phase - wave_number * (u * BODY_LENGTH_M)) + yaw_bias * u * BODY_LENGTH_M)
        z = tail_weight * (0.12 * sin(phase - wave_number * (u * BODY_LENGTH_M) + 1.1) + pitch_bias * u * BODY_LENGTH_M)
        y, z = rotate_yz(y, z, roll)
        pts.append(vector(x, y, z))
    return pts


def make_fluke_triangle(p0, p1, p2, shade):
    return triangle(v0=vertex(pos=p0, color=shade), v1=vertex(pos=p1, color=shade), v2=vertex(pos=p2, color=shade))


def main():
    scene = canvas(
        title="Single Dolphin Swimming in Place — Body/Tail Propulsion Model",
        width=1200,
        height=760,
        background=vector(0.80, 0.92, 1.0),
        center=vector(0, 0, 0),
    )
    scene.forward = vector(-2.8, -2.4, -1.2)
    scene.range = 4.6
    local_light(pos=vector(2, 3, 4), color=color.white)

    water = box(pos=vector(0, 0, -0.72), size=vector(7.5, 4.0, 0.025), color=vector(0.62, 0.84, 0.98), opacity=0.32)
    x_axis = arrow(pos=vector(-3.4, -1.75, -0.65), axis=vector(1.0, 0, 0), color=color.red, shaftwidth=0.025)
    y_axis = arrow(pos=vector(-3.4, -1.75, -0.65), axis=vector(0, 1.0, 0), color=color.green, shaftwidth=0.025)
    z_axis = arrow(pos=vector(-3.4, -1.75, -0.65), axis=vector(0, 0, 1.0), color=color.blue, shaftwidth=0.025)
    label(pos=x_axis.pos + x_axis.axis * 1.15, text="virtual forward", box=False, opacity=0, color=color.red, height=12)
    label(pos=y_axis.pos + y_axis.axis * 1.15, text="turn", box=False, opacity=0, color=color.green, height=12)
    label(pos=z_axis.pos + z_axis.axis * 1.15, text="ascend", box=False, opacity=0, color=color.blue, height=12)

    state = DolphinState()
    pts = body_points(state)

    # Dolphin visual pieces.
    body_color = vector(0.28, 0.48, 0.62)
    belly_color = vector(0.82, 0.90, 0.93)
    fin_color = vector(0.20, 0.36, 0.48)
    eye_color = vector(0.02, 0.02, 0.03)

    centerline = curve(pos=pts, radius=0.045, color=vector(0.12, 0.30, 0.42))
    segments = []
    radii = [0.22, 0.30, 0.36, 0.40, 0.38, 0.34, 0.28, 0.22, 0.17, 0.12, 0.08]
    for i, r in enumerate(radii):
        u = i / (len(radii) - 1)
        seg = ellipsoid(
            pos=pts[min(i + 1, len(pts) - 1)],
            length=0.42,
            height=r * 1.05,
            width=r * 1.65,
            color=body_color if i < 7 else fin_color,
            opacity=0.96,
        )
        segments.append(seg)

    head = ellipsoid(pos=pts[0] + vector(0.19, 0, 0), length=0.62, height=0.42, width=0.56, color=body_color)
    rostrum = cone(pos=pts[0] + vector(0.36, 0, 0), axis=vector(0.42, 0, 0), radius=0.17, color=body_color)
    belly = ellipsoid(pos=vector(0.15, 0, -0.13), length=1.35, height=0.13, width=0.35, color=belly_color, opacity=0.86)
    eye_l = sphere(pos=pts[0] + vector(0.28, -0.16, 0.10), radius=0.035, color=eye_color)
    eye_r = sphere(pos=pts[0] + vector(0.28, 0.16, 0.10), radius=0.035, color=eye_color)

    dorsal = cone(pos=vector(0.15, 0, 0.22), axis=vector(-0.10, 0, 0.45), radius=0.16, color=fin_color)
    pectoral_l = cone(pos=vector(0.58, -0.27, -0.05), axis=vector(-0.26, -0.55, -0.18), radius=0.10, color=fin_color)
    pectoral_r = cone(pos=vector(0.58, 0.27, -0.05), axis=vector(-0.26, 0.55, -0.18), radius=0.10, color=fin_color)

    fluke_left = make_fluke_triangle(vector(-1.45, 0, 0), vector(-1.80, -0.55, 0.10), vector(-1.62, -0.05, 0.02), fin_color)
    fluke_right = make_fluke_triangle(vector(-1.45, 0, 0), vector(-1.80, 0.55, 0.10), vector(-1.62, 0.05, 0.02), fin_color)

    thrust_arrow = arrow(pos=vector(-2.7, 1.45, 0.45), axis=vector(0.8, 0, 0), color=color.orange, shaftwidth=0.05)
    drag_arrow = arrow(pos=vector(2.7, 1.45, 0.45), axis=vector(-0.8, 0, 0), color=color.cyan, shaftwidth=0.05)
    swim_arrow = arrow(pos=vector(-0.8, -1.45, 0.1), axis=vector(1.6, 0, 0), color=color.yellow, shaftwidth=0.035)

    info = label(pos=vector(0, 1.95, 1.25), text="", box=False, opacity=0, height=15, color=vector(0.05, 0.13, 0.18))
    mode_label = label(pos=vector(0, -1.95, 1.0), text="", box=False, opacity=0, height=16, color=vector(0.08, 0.15, 0.20))

    # Backward streamlines represent water flow past the stationary dolphin.
    streamlines = []
    streamline_points = []
    for j in range(12):
        y = -1.55 + (j % 6) * 0.62
        z = -0.38 + (j // 6) * 0.58
        pts_line = [vector(3.2, y, z), vector(2.3, y + 0.05 * sin(j), z), vector(1.4, y, z)]
        line = curve(
            pos=pts_line,
            radius=0.012,
            color=vector(0.35, 0.66, 0.92),
        )
        streamlines.append(line)
        streamline_points.append(pts_line)

    last_terminal_print = -1.0
    dt = 0.025

    while True:
        rate(60)
        step(state, dt)
        pts = body_points(state)
        mode_i, mode, local_t = active_mode(state.t)

        # Update centerline and body segments.
        centerline.clear()
        for p in pts:
            centerline.append(pos=p)

        for i, seg in enumerate(segments):
            idx = min(i + 1, len(pts) - 2)
            p = pts[idx]
            q = pts[idx + 1]
            tangent = q - p
            seg.pos = p
            seg.axis = tangent
            seg.length = max(0.24, tangent.mag * 1.4)
            pulse = 1.0 + 0.06 * sin(state.phase - i * 0.55)
            seg.height = radii[i] * 1.05 * pulse
            seg.width = radii[i] * 1.65 * (2.0 - pulse)

        head.pos = pts[0] + vector(0.19, 0, 0)
        head.axis = pts[0] - pts[1]
        rostrum.pos = pts[0] + vector(0.33, 0, 0)
        rostrum.axis = vector(0.42, 0, 0)
        eye_l.pos = pts[0] + vector(0.28, -0.16, 0.10)
        eye_r.pos = pts[0] + vector(0.28, 0.16, 0.10)
        belly.pos = pts[3] + vector(0.05, 0, -0.18)

        # Fins follow the local body roll and tail/body motion.
        roll = radians(state.roll_deg)
        dorsal.pos = pts[4] + vector(0, -0.10 * sin(roll), 0.28 * cos(roll))
        dorsal.axis = vector(-0.10, -0.20 * sin(roll), 0.45 * cos(roll))
        pectoral_l.pos = pts[3] + vector(0.08, -0.30 * cos(roll), -0.06 - 0.20 * sin(roll))
        pectoral_r.pos = pts[3] + vector(0.08, 0.30 * cos(roll), -0.06 + 0.20 * sin(roll))
        pectoral_l.axis = vector(-0.30, -0.56 * cos(roll), -0.16 - 0.20 * sin(roll))
        pectoral_r.axis = vector(-0.30, 0.56 * cos(roll), -0.16 + 0.20 * sin(roll))

        tail_base = pts[-2]
        tail_tip = pts[-1]
        fluke_span = 0.66
        fluke_lift = 0.18 * sin(radians(state.tail_angle_deg))
        fluke_left.v0.pos = tail_base
        fluke_left.v1.pos = tail_tip + vector(-0.12, -fluke_span, fluke_lift)
        fluke_left.v2.pos = tail_tip + vector(0.18, -0.08, -0.04)
        fluke_right.v0.pos = tail_base
        fluke_right.v1.pos = tail_tip + vector(-0.12, fluke_span, -fluke_lift)
        fluke_right.v2.pos = tail_tip + vector(0.18, 0.08, -0.04)

        # Move water lines backward. Faster virtual speed makes flow pass faster.
        for j, line in enumerate(streamlines):
            # VPython curve objects in some installs do not expose a .pos list.
            # Keep our own point list and rewrite the curve each frame.
            shift = (state.speed * 0.015 + 0.01) * (1.0 + 0.03 * j)
            old_pts = streamline_points[j]
            new_pts = []
            for p in old_pts:
                nx = p.x - shift
                if nx < -3.4:
                    nx = 3.4
                wobble = 0.025 * sin(state.phase + j + p.x)
                new_pts.append(vector(nx, p.y + wobble, p.z))
            streamline_points[j] = new_pts
            line.clear()
            for p in new_pts:
                line.append(pos=p)

        thrust_arrow.axis = vector(max(0.15, state.thrust_n / 55.0), 0, 0)
        drag_arrow.axis = vector(-max(0.15, state.drag_n / 55.0), 0, 0)
        swim_arrow.axis = vector(max(0.4, state.speed / 1.3), 0, 0)

        info.text = (
            f"t={state.t:5.1f}s   speed={state.speed:4.2f} m/s   accel={state.accel:5.2f} m/s²   "
            f"tail={state.tail_angle_deg:5.1f}°   curvature={state.body_curvature:6.3f} 1/L\n"
            f"thrust={state.thrust_n:5.1f} N   drag={state.drag_n:5.1f} N   "
            f"yaw={state.yaw_deg:5.1f}°   pitch={state.pitch_deg:5.1f}°   roll={state.roll_deg:5.1f}°   "
            f"virtual distance={state.virtual_distance:6.1f} m"
        )
        mode_label.text = f"Mode {mode_i + 1}: {mode.name} — {mode.notes}"

        if state.t - last_terminal_print >= 1.0:
            print(
                f"t={state.t:5.1f}s | mode={mode.name:<20} | speed={state.speed:4.2f} m/s | "
                f"tail={state.tail_angle_deg:6.1f} deg | curve={state.body_curvature:7.3f} 1/L | "
                f"thrust={state.thrust_n:6.1f} N | drag={state.drag_n:5.1f} N | "
                f"yaw={state.yaw_deg:6.1f} pitch={state.pitch_deg:6.1f} roll={state.roll_deg:6.1f}"
            )
            last_terminal_print = state.t


if __name__ == "__main__":
    main()
