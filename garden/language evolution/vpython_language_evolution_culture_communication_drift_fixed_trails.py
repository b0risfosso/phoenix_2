from vpython import *
import random
import math

# ------------------------------------------------------------
# Language Evolution — culture, communication drift, new words
# ------------------------------------------------------------
# VPython simulation of several cultural communities exchanging words.
# Words drift through mutation, borrowing, blending, clipping, and prestige.
# No CSV logging.
#
# Controls:
#   Space : pause / resume
#   r     : reset simulation
#   b     : force borrowing wave
#   m     : force mutation wave
#   n     : create new slang / new word burst
#   d     : toggle drift
#   c     : toggle cinematic camera
#   1     : whole map view
#   2     : orbit culture view
#   3     : close follow community
#   4     : lexicon board view
#   +/=   : increase communication rate
#   -     : decrease communication rate

scene = canvas(
    title="Language Evolution — Culture, Communication Drift, New Word Formation",
    width=1280,
    height=760,
    background=vector(0.86, 0.93, 1.0),
)
scene.forward = vector(-0.8, -0.55, -1.0)
scene.range = 22
scene.center = vector(0, 1.5, 0)

random.seed(17)

# ---------- Parameters ----------
WORLD_RADIUS = 13.0
COMMUNITY_COUNT = 5
PEOPLE_PER_COMMUNITY = 15
BASE_COMM_RATE = 0.55
DRIFT_RATE = 0.018
BORROW_CHANCE = 0.055
NEW_WORD_CHANCE = 0.025
PRESTIGE_SHIFT_RATE = 0.003
MAX_FLOATING_LABELS = 36
MAX_MESSAGE_PULSES = 80
MAX_WORD_TOKENS = 90

paused = False
drift_enabled = True
cinematic_camera = True
camera_mode = 1
communication_rate = BASE_COMM_RATE
forced_borrow_wave = 0
forced_mutation_wave = 0
forced_new_word_wave = 0
follow_index = 0
sim_time = 0.0
generation = 1

# ---------- Colors ----------
community_colors = [
    vector(0.1, 0.45, 1.0),
    vector(0.95, 0.25, 0.2),
    vector(0.05, 0.7, 0.35),
    vector(0.85, 0.45, 1.0),
    vector(1.0, 0.62, 0.12),
]

# ---------- Word data ----------
BASE_MEANINGS = [
    "water", "fire", "home", "friend", "food", "sky", "trade", "song", "river", "light"
]
BASE_WORDS = {
    "water": ["mira", "wato", "nari", "aqua", "sula"],
    "fire": ["pyr", "faro", "igni", "tala", "vesta"],
    "home": ["dom", "hama", "noko", "casa", "uru"],
    "friend": ["ami", "pala", "frin", "sabi", "kora"],
    "food": ["nema", "brea", "tavo", "maku", "grain"],
    "sky": ["sora", "aero", "nubo", "kiel", "vela"],
    "trade": ["baro", "merk", "swap", "toka", "daro"],
    "song": ["liri", "sona", "chant", "melo", "voka"],
    "river": ["rivo", "flum", "kana", "brook", "nela"],
    "light": ["luma", "sol", "glow", "sera", "lux"],
}

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnprstvwxyz"
SYLLABLES = ["ba", "ka", "lo", "mi", "ra", "tu", "sa", "ne", "vo", "ki", "zu", "sha", "lum", "tor"]

objects = []
communities = []
message_pulses = []
floating_labels = []
word_tokens = []
lexicon_rows = []

# ---------- Utility ----------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def make_obj(obj):
    objects.append(obj)
    return obj


def clear_all():
    global objects, communities, message_pulses, floating_labels, word_tokens, lexicon_rows
    for obj in objects:
        try:
            obj.visible = False
        except Exception:
            pass
    objects = []
    communities = []
    message_pulses = []
    floating_labels = []
    word_tokens = []
    lexicon_rows = []


def mutate_word(word):
    if len(word) < 2:
        return word + random.choice(VOWELS)
    chars = list(word)
    mode = random.choice(["vowel_shift", "consonant_shift", "drop", "add", "repeat", "soften"])
    if mode == "vowel_shift":
        positions = [i for i, c in enumerate(chars) if c in VOWELS]
        if positions:
            chars[random.choice(positions)] = random.choice(VOWELS)
    elif mode == "consonant_shift":
        positions = [i for i, c in enumerate(chars) if c not in VOWELS]
        if positions:
            chars[random.choice(positions)] = random.choice(CONSONANTS)
    elif mode == "drop" and len(chars) > 3:
        chars.pop(random.randrange(len(chars)))
    elif mode == "add":
        chars.insert(random.randrange(len(chars) + 1), random.choice(VOWELS + CONSONANTS))
    elif mode == "repeat" and len(chars) < 9:
        i = random.randrange(len(chars))
        chars.insert(i, chars[i])
    elif mode == "soften":
        return word.replace("k", "ch").replace("t", "d").replace("p", "b")
    return "".join(chars)


def blend_words(a, b):
    if not a:
        return b
    if not b:
        return a
    cut_a = random.randint(1, max(1, len(a) - 1))
    cut_b = random.randint(1, max(1, len(b) - 1))
    return (a[:cut_a] + b[cut_b:])[:10]


def new_word():
    return "".join(random.choice(SYLLABLES) for _ in range(random.randint(2, 3)))[:10]


def word_similarity(a, b):
    if not a or not b:
        return 0.0
    common = sum(1 for c in set(a) if c in set(b))
    length_penalty = abs(len(a) - len(b)) / max(len(a), len(b))
    return clamp(common / max(len(set(a + b)), 1) - 0.25 * length_penalty, 0, 1)


class Community:
    def __init__(self, idx, angle):
        self.idx = idx
        self.name = ["Harbor", "Highland", "Market", "Forest", "Rivergate"][idx]
        self.color = community_colors[idx]
        self.pos = vector(math.cos(angle) * WORLD_RADIUS, 0, math.sin(angle) * WORLD_RADIUS)
        self.velocity = vector(0, 0, 0)
        self.prestige = random.uniform(0.7, 1.3)
        self.openness = random.uniform(0.55, 1.25)
        self.innovation = random.uniform(0.45, 1.15)
        self.lexicon = {meaning: BASE_WORDS[meaning][idx] for meaning in BASE_MEANINGS}
        self.word_age = {meaning: random.randint(1, 6) for meaning in BASE_MEANINGS}
        self.active_meaning = random.choice(BASE_MEANINGS)
        self.last_change = "stable speech"
        self.people = []
        self.create_visuals()

    def create_visuals(self):
        make_obj(cylinder(pos=self.pos + vector(0, -0.13, 0), axis=vector(0, 0.12, 0), radius=2.55,
                          color=self.color, opacity=0.22))
        make_obj(ring(pos=self.pos + vector(0, 0.04, 0), axis=vector(0, 1, 0), radius=2.7, thickness=0.045,
                      color=self.color, opacity=0.75))
        self.core = make_obj(sphere(pos=self.pos + vector(0, 0.45, 0), radius=0.65, color=self.color, opacity=0.9))
        self.name_label = make_obj(label(pos=self.pos + vector(0, 2.25, 0), text=self.name,
                                         height=14, box=False, color=vector(0.05, 0.05, 0.08)))
        self.word_label = make_obj(label(pos=self.pos + vector(0, 1.45, 0), text="",
                                         height=11, box=True, border=4, color=vector(0.02, 0.02, 0.02),
                                         background=vector(0.96, 0.98, 1.0), opacity=0.9))
        for p in range(PEOPLE_PER_COMMUNITY):
            theta = 2 * math.pi * p / PEOPLE_PER_COMMUNITY + random.uniform(-0.1, 0.1)
            rad = random.uniform(0.95, 2.0)
            dot = make_obj(sphere(pos=self.pos + vector(math.cos(theta) * rad, 0.25, math.sin(theta) * rad),
                                  radius=0.13, color=self.color, opacity=0.78))
            self.people.append({"body": dot, "theta": theta, "rad": rad, "speed": random.uniform(0.5, 1.5)})

    def update_people(self, dt):
        for person in self.people:
            person["theta"] += dt * 0.42 * person["speed"] * (0.5 + self.openness)
            wobble = 0.18 * math.sin(sim_time * person["speed"] + self.idx)
            person["body"].pos = self.pos + vector(
                math.cos(person["theta"]) * (person["rad"] + wobble),
                0.25 + 0.08 * math.sin(sim_time * 2 + person["theta"]),
                math.sin(person["theta"]) * (person["rad"] + wobble),
            )

    def update_label(self):
        self.active_meaning = random.choice(BASE_MEANINGS) if random.random() < 0.01 else self.active_meaning
        word = self.lexicon[self.active_meaning]
        self.word_label.text = f"{self.active_meaning}: {word}\nprestige {self.prestige:.2f} | openness {self.openness:.2f}"

    def internal_drift(self):
        meaning = random.choice(BASE_MEANINGS)
        old = self.lexicon[meaning]
        changed = mutate_word(old)
        if changed != old:
            self.lexicon[meaning] = changed
            self.word_age[meaning] = 0
            self.last_change = f"drift: {old} → {changed}"
            spawn_word_token(self.pos + vector(0, 2.8, 0), changed, self.color, "mutation")

    def invent_word(self):
        meaning = random.choice(BASE_MEANINGS)
        old = self.lexicon[meaning]
        if random.random() < 0.55:
            fresh = blend_words(old, new_word())
        else:
            fresh = new_word()
        self.lexicon[meaning] = fresh
        self.word_age[meaning] = 0
        self.last_change = f"new word for {meaning}: {fresh}"
        spawn_word_token(self.pos + vector(0, 3.1, 0), fresh, self.color, "new")

    def age_words(self):
        for meaning in BASE_MEANINGS:
            self.word_age[meaning] += 1


class MessagePulse:
    def __init__(self, src, dst, meaning, word, color_vec, kind="message"):
        self.src = src
        self.dst = dst
        self.meaning = meaning
        self.word = word
        self.kind = kind
        self.t = 0.0
        self.speed = random.uniform(0.55, 0.95)
        self.arc_height = random.uniform(2.5, 5.5)
        self.body = make_obj(sphere(pos=src.pos + vector(0, 1.0, 0), radius=0.18, color=color_vec, opacity=0.9,
                                    emissive=True))
        # VPython curve objects do not consistently expose a public .pos list
        # across versions, so store trail points in plain Python data and
        # rebuild the curve when the trail needs trimming.
        self.trail_points = [self.body.pos]
        self.trail = make_obj(curve(pos=self.trail_points, color=color_vec, radius=0.025))
        self.label = make_obj(label(pos=self.body.pos + vector(0, 0.45, 0), text=word, height=9, box=False,
                                    color=vector(0.05, 0.05, 0.05)))

    def update(self, dt):
        self.t += dt * self.speed
        t = clamp(self.t, 0, 1)
        start = self.src.pos + vector(0, 1.0, 0)
        end = self.dst.pos + vector(0, 1.0, 0)
        mid = start * (1 - t) + end * t
        mid.y += math.sin(math.pi * t) * self.arc_height
        self.body.pos = mid
        self.label.pos = mid + vector(0, 0.48, 0)
        self.trail_points.append(vector(mid.x, mid.y, mid.z))
        if len(self.trail_points) > 16:
            self.trail_points = self.trail_points[-16:]
            self.trail.clear()
            for p in self.trail_points:
                self.trail.append(pos=p)
        else:
            self.trail.append(pos=mid)
        if self.t >= 1.0:
            return True
        return False

    def destroy(self):
        self.body.visible = False
        self.trail.visible = False
        self.label.visible = False


class WordToken:
    def __init__(self, pos, text, color_vec, kind):
        self.age = 0.0
        self.life = random.uniform(3.2, 5.2)
        self.vel = vector(random.uniform(-0.18, 0.18), random.uniform(0.18, 0.35), random.uniform(-0.18, 0.18))
        self.kind = kind
        self.body = make_obj(sphere(pos=pos, radius=0.22, color=color_vec, opacity=0.65, emissive=True))
        self.label = make_obj(label(pos=pos + vector(0, 0.38, 0), text=text, height=10, box=False,
                                    color=vector(0.02, 0.02, 0.04)))

    def update(self, dt):
        self.age += dt
        self.body.pos += self.vel * dt
        self.label.pos = self.body.pos + vector(0, 0.38, 0)
        fade = clamp(1 - self.age / self.life, 0, 1)
        self.body.opacity = 0.65 * fade
        self.body.radius = 0.08 + 0.18 * fade
        if self.age >= self.life:
            self.body.visible = False
            self.label.visible = False
            return True
        return False


def spawn_word_token(pos, text, color_vec, kind):
    word_tokens.append(WordToken(pos, text, color_vec, kind))
    while len(word_tokens) > MAX_WORD_TOKENS:
        old = word_tokens.pop(0)
        old.body.visible = False
        old.label.visible = False


def spawn_event_label(pos, text, color_vec=vector(0.05, 0.05, 0.05)):
    lab = make_obj(label(pos=pos, text=text, height=12, box=True, border=4,
                         background=vector(1, 1, 1), color=color_vec, opacity=0.82))
    floating_labels.append({"label": lab, "age": 0.0, "life": 3.2})
    while len(floating_labels) > MAX_FLOATING_LABELS:
        old = floating_labels.pop(0)["label"]
        old.visible = False


def setup_world():
    global sim_time, generation, paused, communication_rate, forced_borrow_wave, forced_mutation_wave, forced_new_word_wave
    clear_all()
    sim_time = 0.0
    generation = 1
    paused = False
    communication_rate = BASE_COMM_RATE
    forced_borrow_wave = 0
    forced_mutation_wave = 0
    forced_new_word_wave = 0

    # Ground, geography, and paths.
    make_obj(box(pos=vector(0, -0.22, 0), size=vector(34, 0.12, 34), color=vector(0.78, 0.90, 0.78), opacity=0.95))
    make_obj(ring(pos=vector(0, -0.12, 0), axis=vector(0, 1, 0), radius=WORLD_RADIUS, thickness=0.035,
                  color=vector(0.25, 0.45, 0.3), opacity=0.35))
    make_obj(label(pos=vector(0, 7.2, -14.2), text="Language Evolution: drift, borrowing, slang, prestige, communication",
                   height=16, box=False, color=vector(0.02, 0.03, 0.05)))

    for i in range(COMMUNITY_COUNT):
        angle = 2 * math.pi * i / COMMUNITY_COUNT + 0.16
        communities.append(Community(i, angle))

    # Communication routes between communities.
    for i, src in enumerate(communities):
        for j, dst in enumerate(communities):
            if i < j:
                c = make_obj(curve(pos=[src.pos + vector(0, 0.02, 0), dst.pos + vector(0, 0.02, 0)],
                                   color=vector(0.42, 0.46, 0.5), radius=0.018))
                c.opacity = 0.22

    create_boards()


def create_boards():
    global title_label, stats_label, event_label, lexicon_rows
    title_label = make_obj(label(pos=vector(-16.0, 7.0, 0), text="", height=12, box=True, border=6,
                                 background=vector(0.98, 0.99, 1.0), color=vector(0.02, 0.02, 0.03)))
    stats_label = make_obj(label(pos=vector(15.5, 7.0, 0), text="", height=12, box=True, border=6,
                                 background=vector(0.98, 0.99, 1.0), color=vector(0.02, 0.02, 0.03)))
    event_label = make_obj(label(pos=vector(0, 6.5, 14.5), text="", height=12, box=True, border=6,
                                 background=vector(1.0, 0.98, 0.9), color=vector(0.02, 0.02, 0.03)))
    x0 = -15.5
    y0 = 5.25
    z0 = -15.0
    make_obj(label(pos=vector(x0, y0 + 0.8, z0), text="Shared meanings / current words", height=12, box=False,
                   color=vector(0.04, 0.04, 0.06)))
    for k, meaning in enumerate(BASE_MEANINGS[:8]):
        lab = make_obj(label(pos=vector(x0, y0 - 0.52 * k, z0), text="", height=9, box=False,
                             color=vector(0.04, 0.04, 0.06)))
        lexicon_rows.append((meaning, lab))


def communicate(src, dst, force_borrow=False):
    meaning = random.choice(BASE_MEANINGS)
    src_word = src.lexicon[meaning]
    dst_word = dst.lexicon[meaning]
    pulse = MessagePulse(src, dst, meaning, src_word, src.color, kind="borrow" if force_borrow else "message")
    message_pulses.append(pulse)
    while len(message_pulses) > MAX_MESSAGE_PULSES:
        old = message_pulses.pop(0)
        old.destroy()

    # Borrowing is more likely when source has higher prestige and destination is open.
    prestige_pressure = src.prestige / max(0.1, dst.prestige)
    similarity = word_similarity(src_word, dst_word)
    borrow_prob = BORROW_CHANCE * dst.openness * prestige_pressure * (1.05 - 0.35 * similarity)
    if force_borrow:
        borrow_prob = 1.0
    if random.random() < borrow_prob:
        if random.random() < 0.34:
            borrowed = blend_words(dst_word, src_word)
            change_type = "blend"
        else:
            borrowed = src_word
            change_type = "borrow"
        dst.lexicon[meaning] = borrowed
        dst.word_age[meaning] = 0
        dst.last_change = f"{change_type}: {meaning} = {borrowed} from {src.name}"
        spawn_event_label(dst.pos + vector(0, 3.7, 0), f"{dst.name} adopts {meaning}: {borrowed}", src.color)
        spawn_word_token(dst.pos + vector(0, 3.0, 0), borrowed, src.color, change_type)


def update_pulses(dt):
    done = []
    for p in message_pulses:
        if p.update(dt):
            done.append(p)
    for p in done:
        p.destroy()
        if p in message_pulses:
            message_pulses.remove(p)


def update_tokens(dt):
    dead = []
    for tok in word_tokens:
        if tok.update(dt):
            dead.append(tok)
    for tok in dead:
        if tok in word_tokens:
            word_tokens.remove(tok)


def update_floating_labels(dt):
    dead = []
    for item in floating_labels:
        item["age"] += dt
        lab = item["label"]
        lab.pos.y += dt * 0.35
        fade = clamp(1 - item["age"] / item["life"], 0, 1)
        lab.opacity = 0.82 * fade
        if item["age"] >= item["life"]:
            lab.visible = False
            dead.append(item)
    for item in dead:
        if item in floating_labels:
            floating_labels.remove(item)


def update_language(dt):
    global forced_borrow_wave, forced_mutation_wave, forced_new_word_wave, generation

    # Routine communication events.
    attempts = communication_rate * dt * 6.0
    if random.random() < attempts:
        src, dst = random.sample(communities, 2)
        communicate(src, dst)

    # Drift, innovation, and prestige changes.
    for com in communities:
        if drift_enabled and random.random() < DRIFT_RATE * dt * 8.0 * com.innovation:
            com.internal_drift()
        if random.random() < NEW_WORD_CHANCE * dt * 5.0 * com.innovation:
            com.invent_word()
        if random.random() < PRESTIGE_SHIFT_RATE * dt * 20.0:
            com.prestige = clamp(com.prestige + random.uniform(-0.08, 0.08), 0.35, 1.8)
        com.update_people(dt)
        com.update_label()

    # Forced waves from keyboard.
    if forced_borrow_wave > 0:
        forced_borrow_wave -= 1
        src = max(communities, key=lambda c: c.prestige)
        dst = random.choice([c for c in communities if c is not src])
        communicate(src, dst, force_borrow=True)

    if forced_mutation_wave > 0:
        forced_mutation_wave -= 1
        random.choice(communities).internal_drift()

    if forced_new_word_wave > 0:
        forced_new_word_wave -= 1
        random.choice(communities).invent_word()

    if int(sim_time) % 16 == 0 and abs(sim_time - round(sim_time)) < dt:
        generation += 1
        for com in communities:
            com.age_words()


def update_boards():
    dominant = max(communities, key=lambda c: c.prestige)
    mean_openness = sum(c.openness for c in communities) / len(communities)
    title_label.text = (
        f"GENERATION {generation}\n"
        f"communication rate: {communication_rate:.2f}\n"
        f"drift: {'on' if drift_enabled else 'off'}\n"
        f"camera: {'auto' if cinematic_camera else 'manual'}"
    )
    stats_label.text = (
        f"prestige center: {dominant.name}\n"
        f"mean openness: {mean_openness:.2f}\n"
        f"active messages: {len(message_pulses)}\n"
        f"new/drift tokens: {len(word_tokens)}"
    )
    changes = [f"{c.name}: {c.last_change}" for c in communities]
    event_label.text = "Recent language changes\n" + "\n".join(changes[:5])

    for meaning, lab in lexicon_rows:
        row = [f"{c.name[0]}:{c.lexicon[meaning]}" for c in communities]
        lab.text = f"{meaning:<7}  " + "   ".join(row)


def update_camera(dt):
    if not cinematic_camera:
        return
    cycle = (sim_time % 32.0) / 32.0
    global camera_mode
    if cycle < 0.30:
        camera_mode = 1
    elif cycle < 0.55:
        camera_mode = 2
    elif cycle < 0.78:
        camera_mode = 3
    else:
        camera_mode = 4

    if camera_mode == 1:
        angle = sim_time * 0.12
        scene.center = vector(0, 2.0, 0)
        scene.range = 23
        cam_pos = vector(math.cos(angle) * 28, 18, math.sin(angle) * 28)
        scene.forward = norm(scene.center - cam_pos)
    elif camera_mode == 2:
        target = communities[int(sim_time / 6) % len(communities)]
        angle = sim_time * 0.35
        scene.center = target.pos + vector(0, 1.2, 0)
        scene.range = 7.5
        cam_pos = target.pos + vector(math.cos(angle) * 7, 4.2, math.sin(angle) * 7)
        scene.forward = norm(scene.center - cam_pos)
    elif camera_mode == 3:
        target = communities[follow_index % len(communities)]
        scene.center = target.pos + vector(0, 1.2, 0)
        scene.range = 5.2
        scene.forward = norm(vector(-0.7, -0.35, -0.8))
    elif camera_mode == 4:
        scene.center = vector(-9.5, 3.0, -12.0)
        scene.range = 8.5
        scene.forward = norm(vector(0.15, -0.2, -1.0))


def handle_key(evt):
    global paused, drift_enabled, cinematic_camera, camera_mode, communication_rate
    global forced_borrow_wave, forced_mutation_wave, forced_new_word_wave, follow_index
    key = evt.key
    if key == " ":
        paused = not paused
    elif key == "r":
        setup_world()
    elif key == "b":
        forced_borrow_wave = 12
        spawn_event_label(vector(0, 5.4, 0), "BORROWING WAVE: prestige words spread", vector(0.08, 0.1, 0.2))
    elif key == "m":
        forced_mutation_wave = 14
        spawn_event_label(vector(0, 5.4, 0), "MUTATION WAVE: pronunciation drift", vector(0.08, 0.1, 0.2))
    elif key == "n":
        forced_new_word_wave = 12
        spawn_event_label(vector(0, 5.4, 0), "NEW SLANG BURST: new words form", vector(0.08, 0.1, 0.2))
    elif key == "d":
        drift_enabled = not drift_enabled
    elif key == "c":
        cinematic_camera = not cinematic_camera
    elif key in ["+", "="]:
        communication_rate = clamp(communication_rate + 0.1, 0.05, 2.5)
    elif key == "-":
        communication_rate = clamp(communication_rate - 0.1, 0.05, 2.5)
    elif key == "1":
        cinematic_camera = False
        camera_mode = 1
        scene.center = vector(0, 2.0, 0)
        scene.range = 23
        scene.forward = vector(-0.8, -0.55, -1.0)
    elif key == "2":
        cinematic_camera = False
        camera_mode = 2
        target = communities[follow_index % len(communities)]
        scene.center = target.pos + vector(0, 1.2, 0)
        scene.range = 7
    elif key == "3":
        cinematic_camera = False
        camera_mode = 3
        follow_index = (follow_index + 1) % len(communities)
        target = communities[follow_index]
        scene.center = target.pos + vector(0, 1.2, 0)
        scene.range = 5.2
    elif key == "4":
        cinematic_camera = False
        camera_mode = 4
        scene.center = vector(-9.5, 3.0, -12.0)
        scene.range = 8.5


scene.bind("keydown", handle_key)
setup_world()

# ---------- Main loop ----------
dt = 0.035
while True:
    rate(60)
    if paused:
        continue
    sim_time += dt
    update_language(dt)
    update_pulses(dt)
    update_tokens(dt)
    update_floating_labels(dt)
    update_boards()
    update_camera(dt)
