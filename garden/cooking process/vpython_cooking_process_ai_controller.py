from vpython import *
import random
import math

# VPython Simulation: Cooking Process — chemistry, heat transfer, timing, ingredient transformation
# Run: python vpython_cooking_process_ai_controller.py
# Controls:
#   Space : pause/resume
#   r     : reset
#   h     : toggle burner heat
#   +/=   : increase burner power
#   -     : decrease burner power
#   s     : stir ingredients
#   w     : add water splash / cooling pulse
#   i     : add ingredient pieces
#   a     : toggle AI cooking controller
#   c     : toggle cinematic camera
#   1     : whole kitchen view
#   2     : close pan view
#   3     : heat-transfer side view
#   4     : chemistry board view

scene.title = "Cooking Process — AI Controller, Chemistry, Heat Transfer, Ingredient Transformation"
scene.width = 1280
scene.height = 760
scene.background = vector(0.86, 0.92, 1.0)
scene.forward = vector(-0.65, -0.38, -0.66)
scene.center = vector(0, 1.4, 0)
scene.range = 9

# -----------------------------
# Constants and utility helpers
# -----------------------------
DT = 0.035
MAX_INGREDIENTS = 46
MAX_STEAM = 90
MAX_HEAT_PARTICLES = 80
MAX_AROMA = 70
PAN_RADIUS = 3.0
PAN_Y = 0.65
BURNER_Y = 0.12
ROOM_TEMP = 22.0
BOIL_TEMP = 100.0
BROWNING_TEMP = 145.0
BURN_TEMP = 210.0

paused = False
burner_on = True
burner_power = 0.72
stir_impulse = 0.0
water_cooldown = 0.0
cinematic_camera = True
manual_camera_mode = 0
clock_time = 0.0
round_number = 1
ai_controller = True
ai_mode = "balanced cook"
ai_timer = 0.0
ai_action_timer = 0.0
ai_last_action = "monitoring pan"
ai_target_brown = 0.42
ai_target_moisture = 0.32
ai_target_cooked = 0.86


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def lerp(a, b, t):
    return a + (b - a) * clamp(t, 0, 1)


def mix_color(c1, c2, t):
    t = clamp(t, 0, 1)
    return vector(lerp(c1.x, c2.x, t), lerp(c1.y, c2.y, t), lerp(c1.z, c2.z, t))


def rand_pan_pos(radius=2.2):
    ang = random.uniform(0, 2 * math.pi)
    r = radius * math.sqrt(random.random())
    return vector(r * math.cos(ang), PAN_Y + 0.20, r * math.sin(ang))


def safe_norm(v, fallback=vector(1, 0, 0)):
    if mag(v) < 1e-8:
        return fallback
    return norm(v)

# -----------------------------
# Scene objects: stove, burner, pan, ingredient board
# -----------------------------
base = box(pos=vector(0, -0.08, 0), size=vector(9, 0.22, 7), color=vector(0.72, 0.76, 0.78))
stove = box(pos=vector(0, 0.05, 0), size=vector(6.8, 0.22, 5.3), color=vector(0.42, 0.45, 0.47))
# Removed rear wall/panel to keep the scene open.

burner_rings = []
for rad in [1.0, 1.45, 1.9]:
    burner_rings.append(ring(pos=vector(0, BURNER_Y, 0), axis=vector(0, 1, 0), radius=rad, thickness=0.045, color=vector(0.05, 0.08, 0.09)))

flame_core = cylinder(pos=vector(0, BURNER_Y + 0.02, 0), axis=vector(0, 0.06, 0), radius=1.55, color=vector(1, 0.32, 0.02), opacity=0.45)
flame_outer = cylinder(pos=vector(0, BURNER_Y + 0.00, 0), axis=vector(0, 0.04, 0), radius=2.05, color=vector(1, 0.68, 0.12), opacity=0.22)

pan_body = cylinder(pos=vector(0, PAN_Y, 0), axis=vector(0, 0.34, 0), radius=PAN_RADIUS, color=vector(0.12, 0.13, 0.14), opacity=0.93)
pan_surface = cylinder(pos=vector(0, PAN_Y + 0.20, 0), axis=vector(0, 0.035, 0), radius=PAN_RADIUS * 0.92, color=vector(0.04, 0.045, 0.05), opacity=0.82)
pan_lip = ring(pos=vector(0, PAN_Y + 0.38, 0), axis=vector(0, 1, 0), radius=PAN_RADIUS, thickness=0.09, color=vector(0.06, 0.065, 0.07))
pan_handle = cylinder(pos=vector(PAN_RADIUS - 0.12, PAN_Y + 0.22, 0), axis=vector(3.25, 0, 0), radius=0.16, color=vector(0.07, 0.075, 0.08))

# Heat-gradient post beside the pan
heat_meter_base = box(pos=vector(-6.1, 1.25, -3.0), size=vector(0.18, 2.6, 0.18), color=vector(0.22, 0.24, 0.26))
heat_meter_fill = box(pos=vector(-6.1, 0.1, -3.0), size=vector(0.25, 0.1, 0.25), color=vector(1, 0.25, 0.05))

# Chemistry/timing board
board = box(pos=vector(6.1, 2.0, -2.6), size=vector(0.18, 3.3, 3.8), color=vector(0.95, 0.96, 0.92), opacity=0.76)

status_label = label(pos=vector(-6.2, 4.35, -3.2), text="", height=13, border=6, box=True, color=color.black, background=vector(0.94, 0.96, 0.98), opacity=0.85)
thermo_label = label(pos=vector(-6.6, 2.65, -3.0), text="", height=10, border=5, box=True, color=color.black, background=vector(0.95, 0.96, 0.98), opacity=0.82)
chem_label = label(pos=vector(6.25, 2.25, -2.6), text="", height=9, border=4, box=False, color=color.black, opacity=0)
control_label = label(pos=vector(0, -0.85, -4.8), text="", height=9, border=4, box=True, color=color.black, background=vector(0.94, 0.96, 0.98), opacity=0.75)

# -----------------------------
# Particle classes
# -----------------------------
class Ingredient:
    def __init__(self, kind, pos=None):
        self.kind = kind
        self.pos = rand_pan_pos() if pos is None else pos
        self.vel = vector(random.uniform(-0.1, 0.1), 0, random.uniform(-0.1, 0.1))
        self.temp = ROOM_TEMP + random.uniform(-2, 5)
        self.moisture = random.uniform(0.65, 0.95)
        self.cooked = 0.0
        self.browned = 0.0
        self.burned = 0.0
        self.age = 0.0
        self.size0 = random.uniform(0.20, 0.34)
        self.phase = "raw"
        if kind == "vegetable":
            self.raw_color = vector(0.20, 0.72, 0.24)
            self.cooked_color = vector(0.42, 0.60, 0.18)
            self.brown_color = vector(0.58, 0.34, 0.10)
            self.shape = box(pos=self.pos, size=vector(self.size0 * 1.5, self.size0 * 0.45, self.size0), color=self.raw_color)
        elif kind == "protein":
            self.raw_color = vector(0.95, 0.62, 0.48)
            self.cooked_color = vector(0.94, 0.80, 0.58)
            self.brown_color = vector(0.54, 0.30, 0.13)
            self.shape = sphere(pos=self.pos, radius=self.size0, color=self.raw_color)
            self.shape.scale = vector(1.25, 0.45, 1.0)
        else:
            self.raw_color = vector(0.96, 0.86, 0.35)
            self.cooked_color = vector(1.00, 0.70, 0.20)
            self.brown_color = vector(0.64, 0.35, 0.12)
            self.shape = cylinder(pos=self.pos, axis=vector(0, self.size0 * 0.34, 0), radius=self.size0 * 0.72, color=self.raw_color)

    def update(self, dt, pan_temp):
        global stir_impulse, water_cooldown
        self.age += dt
        heat_gain = (pan_temp - self.temp) * (0.018 + 0.020 * burner_power) * dt
        evaporative_loss = self.moisture * max(0, self.temp - 95) * 0.004 * dt
        water_loss = water_cooldown * 30.0 * dt
        self.temp += heat_gain - evaporative_loss - water_loss
        self.temp = clamp(self.temp, 5, 260)

        # Chemistry: denaturation/softening/starch gelation, evaporation, Maillard browning, burning.
        if self.temp > 55:
            self.cooked += (self.temp - 50) / 95.0 * dt * 0.06
        if self.temp > 92:
            self.moisture -= (self.temp - 88) / 90.0 * dt * 0.045
        if self.temp > BROWNING_TEMP and self.moisture < 0.45:
            self.browned += (self.temp - BROWNING_TEMP) / 110.0 * dt * 0.10
        if self.temp > BURN_TEMP:
            self.burned += (self.temp - BURN_TEMP) / 80.0 * dt * 0.16

        self.cooked = clamp(self.cooked, 0, 1)
        self.moisture = clamp(self.moisture, 0, 1)
        self.browned = clamp(self.browned, 0, 1)
        self.burned = clamp(self.burned, 0, 1)

        if self.burned > 0.22:
            self.phase = "burning"
        elif self.browned > 0.20:
            self.phase = "browning"
        elif self.cooked > 0.55:
            self.phase = "cooked"
        elif self.temp > 60:
            self.phase = "warming"
        else:
            self.phase = "raw"

        # Stirring and convection motion.
        swirl = vector(-self.pos.z, 0, self.pos.x)
        swirl = safe_norm(swirl, vector(1, 0, 0))
        center_pull = -vector(self.pos.x, 0, self.pos.z) * 0.04
        heat_jitter = (self.temp / 220.0) * 0.15
        self.vel += (swirl * stir_impulse * 1.9 + center_pull + vector(random.uniform(-heat_jitter, heat_jitter), 0, random.uniform(-heat_jitter, heat_jitter))) * dt
        self.vel *= 0.985
        self.pos += self.vel * dt

        # Keep inside pan.
        radial = vector(self.pos.x, 0, self.pos.z)
        if mag(radial) > PAN_RADIUS * 0.78:
            radial_dir = norm(radial)
            self.pos.x = radial_dir.x * PAN_RADIUS * 0.78
            self.pos.z = radial_dir.z * PAN_RADIUS * 0.78
            self.vel -= 1.5 * dot(self.vel, radial_dir) * radial_dir
        self.pos.y = PAN_Y + 0.25 + 0.035 * math.sin(self.age * 4.0 + self.pos.x)

        # Color and size transform.
        color_after_cooking = mix_color(self.raw_color, self.cooked_color, self.cooked)
        color_after_brown = mix_color(color_after_cooking, self.brown_color, self.browned)
        final_color = mix_color(color_after_brown, vector(0.08, 0.055, 0.035), self.burned)
        self.shape.color = final_color
        self.shape.pos = self.pos
        shrink = 1.0 - 0.20 * (1 - self.moisture) - 0.10 * self.browned
        if hasattr(self.shape, "radius"):
            self.shape.radius = self.size0 * clamp(shrink, 0.55, 1.15)
        if hasattr(self.shape, "size"):
            self.shape.size = vector(self.size0 * 1.5 * shrink, self.size0 * 0.45, self.size0 * shrink)


class RisingParticle:
    def __init__(self, pos, kind):
        self.kind = kind
        self.life = random.uniform(1.2, 2.7) if kind == "steam" else random.uniform(1.4, 3.2)
        self.max_life = self.life
        self.vel = vector(random.uniform(-0.25, 0.25), random.uniform(0.55, 1.15), random.uniform(-0.25, 0.25))
        if kind == "steam":
            self.obj = sphere(pos=pos, radius=random.uniform(0.035, 0.075), color=vector(0.92, 0.94, 0.96), opacity=0.35)
        else:
            self.obj = sphere(pos=pos, radius=random.uniform(0.025, 0.055), color=vector(0.95, 0.70, 0.30), opacity=0.45)

    def update(self, dt):
        self.life -= dt
        self.obj.pos += self.vel * dt
        self.vel.x += random.uniform(-0.05, 0.05) * dt
        self.vel.z += random.uniform(-0.05, 0.05) * dt
        t = clamp(self.life / self.max_life, 0, 1)
        self.obj.opacity = 0.40 * t
        self.obj.radius *= 1.004
        if self.life <= 0:
            self.obj.visible = False
            return True
        return False


class HeatParticle:
    def __init__(self):
        ang = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.25, 1.85)
        self.obj = sphere(pos=vector(r * math.cos(ang), BURNER_Y + 0.15, r * math.sin(ang)), radius=random.uniform(0.035, 0.07), color=vector(1, random.uniform(0.35, 0.75), 0.05), opacity=0.65)
        self.life = random.uniform(0.5, 1.05)
        self.max_life = self.life
        self.vel = vector(random.uniform(-0.05, 0.05), random.uniform(0.8, 1.8), random.uniform(-0.05, 0.05))

    def update(self, dt):
        self.life -= dt
        self.obj.pos += self.vel * dt
        t = clamp(self.life / self.max_life, 0, 1)
        self.obj.opacity = 0.70 * t
        self.obj.radius *= 0.995
        if self.life <= 0 or self.obj.pos.y > PAN_Y + 0.3:
            self.obj.visible = False
            return True
        return False

# -----------------------------
# Simulation state
# -----------------------------
ingredients = []
steam_particles = []
aroma_particles = []
heat_particles = []
pan_temp = ROOM_TEMP
reaction_score = 0.0


def clear_objects():
    global ingredients, steam_particles, aroma_particles, heat_particles
    for item in ingredients:
        item.shape.visible = False
    for p in steam_particles + aroma_particles + heat_particles:
        p.obj.visible = False
    ingredients = []
    steam_particles = []
    aroma_particles = []
    heat_particles = []


def add_ingredients(n=8):
    kinds = ["vegetable", "protein", "starch"]
    for _ in range(n):
        if len(ingredients) >= MAX_INGREDIENTS:
            return
        ingredients.append(Ingredient(random.choice(kinds)))


def apply_water_splash(strength=1.0):
    """Cool the pan and create a visible steam burst."""
    global water_cooldown
    water_cooldown = clamp(max(water_cooldown, strength), 0.0, 1.25)
    burst_count = int(10 + 8 * strength)
    for _ in range(burst_count):
        if len(steam_particles) < MAX_STEAM:
            steam_particles.append(RisingParticle(rand_pan_pos(2.0) + vector(0, 0.25, 0), "steam"))


def average_food_state():
    n = max(1, len(ingredients))
    return {
        "temp": sum(i.temp for i in ingredients) / n,
        "moisture": sum(i.moisture for i in ingredients) / n,
        "cooked": sum(i.cooked for i in ingredients) / n,
        "brown": sum(i.browned for i in ingredients) / n,
        "burn": sum(i.burned for i in ingredients) / n,
    }


def reset_simulation():
    global pan_temp, reaction_score, clock_time, stir_impulse, water_cooldown, round_number
    global ai_timer, ai_action_timer, ai_last_action, ai_target_brown, ai_target_moisture, ai_target_cooked
    clear_objects()
    pan_temp = ROOM_TEMP
    reaction_score = 0.0
    clock_time = 0.0
    stir_impulse = 0.0
    water_cooldown = 0.0
    ai_timer = 0.0
    ai_action_timer = 0.0
    ai_last_action = "new round: preheating and watching moisture"
    ai_target_brown = random.uniform(0.34, 0.52)
    ai_target_moisture = random.uniform(0.24, 0.42)
    ai_target_cooked = random.uniform(0.78, 0.94)
    add_ingredients(24)
    round_number += 1

reset_simulation()
round_number = 1

# -----------------------------
# Update logic
# -----------------------------
def update_ai_controller(dt):
    """Simple rule-based cooking controller.

    The controller tries to cook ingredients through, preserve some moisture,
    create browning, and prevent burning. It acts like a cook: lowers heat when
    the pan is too hot, stirs to distribute heat, splashes water to cool/delay
    burning, and adds fresh ingredients when a round gets sparse.
    """
    global ai_timer, ai_action_timer, ai_last_action, burner_on, burner_power, stir_impulse

    if not ai_controller:
        return

    ai_timer += dt
    ai_action_timer = max(0.0, ai_action_timer - dt)
    state = average_food_state()
    avg_temp = state["temp"]
    avg_moisture = state["moisture"]
    avg_cooked = state["cooked"]
    avg_brown = state["brown"]
    avg_burn = state["burn"]

    if not burner_on:
        burner_on = True
        ai_last_action = "turned burner on"

    # Emergency cooling: too hot or visible burn risk.
    if avg_burn > 0.08 or pan_temp > 235 or (avg_brown > ai_target_brown + 0.12 and avg_moisture < 0.20):
        burner_power = clamp(burner_power - 0.045, 0.05, 1.0)
        if ai_action_timer <= 0:
            apply_water_splash(1.0)
            stir_impulse = 1.0
            ai_action_timer = 2.0
            ai_last_action = "emergency cool + stir to prevent burning"
        return

    # Early round: preheat and start cooking.
    if avg_cooked < 0.35:
        target_power = 0.82 if pan_temp < 145 else 0.66
        burner_power += (target_power - burner_power) * 0.035
        if ai_action_timer <= 0 and pan_temp > 90:
            stir_impulse = 0.88
            ai_action_timer = 1.35
            ai_last_action = "stirring early ingredients for even heat"
        else:
            ai_last_action = "building heat for raw ingredients"
        return

    # Middle round: cook through without drying out.
    if avg_cooked < ai_target_cooked:
        if avg_moisture < ai_target_moisture and pan_temp > 150:
            burner_power = clamp(burner_power - 0.025, 0.05, 1.0)
            if ai_action_timer <= 0:
                apply_water_splash(0.65)
                ai_action_timer = 2.4
                ai_last_action = "adding moisture while cooking through"
        else:
            target_power = 0.58 if pan_temp > 175 else 0.70
            burner_power += (target_power - burner_power) * 0.030
            ai_last_action = "holding steady heat for transformation"
        if ai_action_timer <= 0:
            stir_impulse = 0.72
            ai_action_timer = 1.7
            ai_last_action = "stirring to reduce hot spots"
        return

    # Browning stage: lower moisture and create Maillard color, but avoid burning.
    if avg_brown < ai_target_brown:
        target_power = 0.62 if avg_moisture > 0.28 else 0.48
        if pan_temp < BROWNING_TEMP + 10:
            target_power += 0.10
        burner_power += (target_power - burner_power) * 0.030
        if ai_action_timer <= 0:
            stir_impulse = 0.55
            ai_action_timer = 2.2
            ai_last_action = "controlled browning with periodic stir"
        else:
            ai_last_action = "letting Maillard browning develop"
        return

    # Finish stage: coast, cool, or reset if the food is done enough.
    burner_power = clamp(burner_power - 0.020, 0.05, 1.0)
    if ai_action_timer <= 0:
        stir_impulse = 0.45
        ai_action_timer = 3.0
        ai_last_action = "finishing: low heat and gentle stir"
    if avg_brown > ai_target_brown and avg_cooked > ai_target_cooked and avg_burn < 0.12 and clock_time > 35:
        if random.random() < 0.008:
            add_ingredients(4)
            ai_last_action = "adding a small fresh batch to extend the cook"


def update_heat(dt):
    global pan_temp, water_cooldown
    target = ROOM_TEMP + (burner_power * 235.0 if burner_on else 0.0)
    pan_temp += (target - pan_temp) * (0.035 if burner_on else 0.018) * dt * 10.0
    if water_cooldown > 0:
        pan_temp -= 22.0 * water_cooldown * dt
        water_cooldown = max(0.0, water_cooldown - dt * 0.38)
    pan_temp = clamp(pan_temp, ROOM_TEMP - 2, 270)

    flame_core.visible = burner_on
    flame_outer.visible = burner_on
    flame_core.opacity = 0.18 + 0.48 * burner_power if burner_on else 0
    flame_outer.opacity = 0.08 + 0.30 * burner_power if burner_on else 0
    flame_core.radius = 1.0 + 0.9 * burner_power
    flame_outer.radius = 1.4 + 1.0 * burner_power
    flame_core.color = vector(1, 0.18 + 0.35 * burner_power, 0.02)
    flame_outer.color = vector(1, 0.60 + 0.20 * burner_power, 0.10)

    heat_color = mix_color(vector(0.04, 0.045, 0.05), vector(0.95, 0.22, 0.04), clamp((pan_temp - 40) / 210, 0, 1))
    pan_surface.color = heat_color
    pan_body.color = mix_color(vector(0.12, 0.13, 0.14), vector(0.55, 0.12, 0.05), clamp((pan_temp - 80) / 180, 0, 1))

    # Heat particles from burner into pan.
    if burner_on and len(heat_particles) < MAX_HEAT_PARTICLES and random.random() < burner_power * 0.65:
        heat_particles.append(HeatParticle())


def update_ingredients(dt):
    global stir_impulse, reaction_score
    cooked_sum = 0.0
    brown_sum = 0.0
    burned_sum = 0.0
    moisture_sum = 0.0
    for ing in ingredients:
        ing.update(dt, pan_temp)
        cooked_sum += ing.cooked
        brown_sum += ing.browned
        burned_sum += ing.burned
        moisture_sum += ing.moisture

        # Generate steam from hot moist pieces.
        if ing.temp > 88 and ing.moisture > 0.12 and len(steam_particles) < MAX_STEAM and random.random() < 0.065:
            steam_particles.append(RisingParticle(ing.pos + vector(0, 0.15, 0), "steam"))
        # Aroma particles from browning.
        if ing.browned > 0.20 and len(aroma_particles) < MAX_AROMA and random.random() < 0.045:
            aroma_particles.append(RisingParticle(ing.pos + vector(0, 0.22, 0), "aroma"))

    n = max(1, len(ingredients))
    reaction_score = clamp((cooked_sum / n) * 0.55 + (brown_sum / n) * 0.45 - (burned_sum / n) * 0.55, 0, 1)
    stir_impulse = max(0.0, stir_impulse - dt * 0.55)


def update_particles(dt):
    for arr in [steam_particles, aroma_particles, heat_particles]:
        for p in arr[:]:
            if p.update(dt):
                arr.remove(p)


def cooking_stage():
    if len(ingredients) == 0:
        return "empty pan"
    avg_cooked = sum(i.cooked for i in ingredients) / len(ingredients)
    avg_brown = sum(i.browned for i in ingredients) / len(ingredients)
    avg_burn = sum(i.burned for i in ingredients) / len(ingredients)
    avg_moisture = sum(i.moisture for i in ingredients) / len(ingredients)
    if avg_burn > 0.18:
        return "overcooking / burning"
    if avg_brown > 0.35:
        return "Maillard browning"
    if avg_cooked > 0.65 and avg_moisture > 0.25:
        return "cooked through"
    if pan_temp > BOIL_TEMP:
        return "simmering / evaporation"
    if pan_temp > 50:
        return "warming / heat transfer"
    return "raw preparation"


def update_labels():
    avg_temp = sum(i.temp for i in ingredients) / max(1, len(ingredients))
    avg_moisture = sum(i.moisture for i in ingredients) / max(1, len(ingredients))
    avg_cooked = sum(i.cooked for i in ingredients) / max(1, len(ingredients))
    avg_brown = sum(i.browned for i in ingredients) / max(1, len(ingredients))
    avg_burn = sum(i.burned for i in ingredients) / max(1, len(ingredients))

    status_label.text = (
        f"COOKING PROCESS SIMULATION | round {round_number}\n"
        f"stage: {cooking_stage()} | time: {clock_time:5.1f}s | burner: {'ON' if burner_on else 'OFF'} | power: {burner_power:.2f}\n"
        f"pan {pan_temp:5.1f}°C | ingredient avg {avg_temp:5.1f}°C | reaction quality {reaction_score:.2f}\n"
        f"AI cook: {'ON' if ai_controller else 'OFF'} | {ai_last_action}"
    )
    thermo_label.text = (
        "HEAT TRANSFER\n"
        f"burner → pan: {max(0, burner_power if burner_on else 0):.2f}\n"
        f"pan temp: {pan_temp:5.1f}°C\n"
        f"food temp: {avg_temp:5.1f}°C\n"
        f"water cooling: {water_cooldown:.2f}\n"
        f"steam particles: {len(steam_particles)}"
    )
    chem_label.text = (
        "CHEMISTRY BOARD\n"
        f"moisture / water: {avg_moisture:.2f}\n"
        f"protein/starch/veg cooked: {avg_cooked:.2f}\n"
        f"browning reaction: {avg_brown:.2f}\n"
        f"burn compounds: {avg_burn:.2f}\n\n"
        "visual meaning:\n"
        "green/pink/yellow = raw\n"
        "gold = cooked\n"
        "brown = Maillard browning\n"
        "dark = burning\n"
        "white = steam, orange = aroma"
    )
    control_label.text = "Space pause | r reset | h heat | +/- power | s stir | w water/cool | i ingredients | a AI | c camera | 1-4 views"

    fill_height = clamp((pan_temp - 20) / 240, 0, 1) * 2.55
    heat_meter_fill.size = vector(0.25, max(0.05, fill_height), 0.25)
    heat_meter_fill.pos = vector(-6.1, 0.1 + fill_height / 2, -3.0)
    heat_meter_fill.color = mix_color(vector(0.2, 0.6, 1.0), vector(1.0, 0.12, 0.02), clamp((pan_temp - 35) / 210, 0, 1))

# -----------------------------
# Keyboard controls
# -----------------------------
def on_keydown(evt):
    global paused, burner_on, burner_power, stir_impulse, water_cooldown, cinematic_camera, manual_camera_mode, ai_controller
    k = evt.key
    if k == " ":
        paused = not paused
    elif k == "r":
        reset_simulation()
    elif k == "h":
        burner_on = not burner_on
    elif k in ["+", "="]:
        burner_power = clamp(burner_power + 0.08, 0.05, 1.0)
    elif k == "-":
        burner_power = clamp(burner_power - 0.08, 0.05, 1.0)
    elif k == "s":
        stir_impulse = 1.0
    elif k == "w":
        apply_water_splash(1.0)
    elif k == "i":
        add_ingredients(6)
    elif k == "a":
        ai_controller = not ai_controller
    elif k == "c":
        cinematic_camera = not cinematic_camera
    elif k in ["1", "2", "3", "4"]:
        manual_camera_mode = int(k)
        cinematic_camera = False

scene.bind("keydown", on_keydown)

# -----------------------------
# Camera
# -----------------------------
def set_camera(center, forward, rng):
    scene.center = center
    scene.forward = safe_norm(forward, vector(-0.6, -0.4, -0.6))
    scene.range = rng


def update_camera(t):
    if cinematic_camera:
        segment = int((t / 7.0) % 4)
        local = (t % 7.0) / 7.0
        if segment == 0:
            ang = t * 0.18
            f = vector(-math.cos(ang), -0.38, -math.sin(ang))
            set_camera(vector(0, 1.7, 0), f, 8.8)
        elif segment == 1:
            ang = t * 0.35
            set_camera(vector(0.6 * math.sin(ang), 1.0, 0.4 * math.cos(ang)), vector(-0.75, -0.26, -0.55), 4.7)
        elif segment == 2:
            set_camera(vector(0, 1.05, 0), vector(-0.05, -1.0, -0.03), 4.9)
        else:
            set_camera(vector(4.2, 2.0, -2.1), vector(-0.95, -0.15, -0.15), 6.0)
    else:
        if manual_camera_mode == 1:
            set_camera(vector(0, 1.6, 0), vector(-0.65, -0.38, -0.66), 9)
        elif manual_camera_mode == 2:
            set_camera(vector(0, 1.0, 0), vector(-0.8, -0.25, -0.55), 4.4)
        elif manual_camera_mode == 3:
            set_camera(vector(0, 1.0, 0), vector(-0.05, -1, -0.05), 5)
        elif manual_camera_mode == 4:
            set_camera(vector(6.0, 2.0, -2.6), vector(-1, -0.10, -0.03), 4.8)

# -----------------------------
# Main loop
# -----------------------------
while True:
    rate(60)
    if not paused:
        clock_time += DT
        update_ai_controller(DT)
        update_heat(DT)
        update_ingredients(DT)
        update_particles(DT)

        # Auto-loop: once the food is strongly burned or the round is long, reset as a new cooking cycle.
        avg_burn = sum(i.burned for i in ingredients) / max(1, len(ingredients))
        if clock_time > 95 or avg_burn > 0.72:
            reset_simulation()

    update_labels()
    update_camera(clock_time)
