"""
VPython Simulation: Stock Market Behavior — Economics, Supply, Demand, Volatility

A visual, educational market simulation showing how supply, demand, trader behavior,
volatility, news shocks, and liquidity can move a simplified stock price.

Run:
    python vpython_stock_market_behavior_supply_demand_volatility.py

Controls:
    Space : pause / resume
    r     : reset simulation
    b     : add buyer wave
    s     : add seller wave
    n     : trigger random news shock
    v     : increase volatility
    V     : decrease volatility
    l     : increase liquidity
    L     : decrease liquidity
    a     : toggle auto market cycles
    c     : toggle cinematic camera
    1     : floor view
    2     : price chart view
    3     : whole market overview
    4     : close order-book view

Notes:
- This is not financial advice and does not model real securities.
- The goal is visual intuition: excess demand tends to push price up,
  excess supply tends to push price down, volatility amplifies movement,
  and liquidity dampens movement.
"""

from vpython import *
import random
import math
from collections import deque

# ----------------------------- Scene setup -----------------------------
scene = canvas(
    title="Stock Market Behavior — Supply, Demand, Volatility",
    width=1280,
    height=760,
    background=vector(0.86, 0.92, 1.0),
    center=vector(0, 3, 0),
)
scene.forward = vector(-0.65, -0.42, -0.62)
scene.range = 30
scene.userspin = True
scene.userzoom = True

# ----------------------------- Constants -----------------------------
DT = 0.035
PRICE_START = 100.0
PRICE_MIN = 10.0
PRICE_MAX = 220.0
HISTORY_LEN = 260
MARKET_HALF_WIDTH = 22
FLOOR_Z = 0
CHART_X0 = -24
CHART_Y0 = 12
CHART_W = 48
CHART_H = 16
ORDER_BOOK_X = 29

# ----------------------------- Colors -----------------------------
BUY_COLOR = vector(0.1, 0.55, 1.0)
SELL_COLOR = vector(1.0, 0.33, 0.18)
NEUTRAL_COLOR = vector(0.6, 0.6, 0.65)
GREEN = vector(0.1, 0.7, 0.25)
RED = vector(0.9, 0.15, 0.08)
DARK = vector(0.08, 0.1, 0.14)
LIGHT = vector(0.92, 0.96, 1.0)
YELLOW = vector(1.0, 0.8, 0.15)
PURPLE = vector(0.55, 0.25, 0.9)

# ----------------------------- Simulation state -----------------------------
paused = False
auto_market_cycles = True
cinematic_camera = True
camera_mode = 0
frame_count = 0

price = PRICE_START
last_price = PRICE_START
volatility = 0.75
liquidity = 1.35
sentiment = 0.0
fundamental_anchor = 100.0
news_timer = 0
news_text = "balanced market"

buy_pressure = 0.0
sell_pressure = 0.0
volume = 0.0
spread = 1.2

price_history = deque([PRICE_START] * HISTORY_LEN, maxlen=HISTORY_LEN)
buy_history = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
sell_history = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
volume_history = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

traders = []
orders = []
chart_segments = []
volume_bars = []
pressure_arrows = []
shock_ripples = []

# ----------------------------- Geometry helpers -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def lerp(a, b, t):
    return a + (b - a) * t


def price_to_chart_y(p):
    norm = (p - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)
    return CHART_Y0 + clamp(norm, 0, 1) * CHART_H


def history_index_to_x(i):
    return CHART_X0 + (i / (HISTORY_LEN - 1)) * CHART_W


def safe_delete(obj):
    try:
        obj.visible = False
        del obj
    except Exception:
        pass

# ----------------------------- Static scene -----------------------------
floor = box(pos=vector(0, -0.08, 0), size=vector(58, 0.16, 34), color=vector(0.78, 0.84, 0.9))

# Trading floor zones
buy_zone = box(pos=vector(-13, 0.02, 0), size=vector(19, 0.06, 30), color=vector(0.72, 0.88, 1.0), opacity=0.32)
sell_zone = box(pos=vector(13, 0.025, 0), size=vector(19, 0.06, 30), color=vector(1.0, 0.78, 0.70), opacity=0.32)
midline = box(pos=vector(0, 0.06, 0), size=vector(0.18, 0.1, 31), color=vector(0.35, 0.38, 0.45))

buy_label = label(pos=vector(-13, 0.3, -15.6), text="BUY DEMAND", color=BUY_COLOR, box=False, height=14)
sell_label = label(pos=vector(13, 0.3, -15.6), text="SELL SUPPLY", color=SELL_COLOR, box=False, height=14)

# Central exchange tower
exchange_base = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0.45, 0), radius=2.5, color=vector(0.25, 0.28, 0.35))
exchange_ring = ring(pos=vector(0, 2.9, 0), axis=vector(0, 1, 0), radius=3.1, thickness=0.08, color=YELLOW)
exchange_screen = box(pos=vector(0, 3.0, -2.65), size=vector(8.0, 3.0, 0.18), color=vector(0.02, 0.04, 0.08))
price_label = label(pos=vector(0, 3.38, -2.82), text="$100.00", height=24, color=GREEN, box=False)
market_label = label(pos=vector(0, 1.95, -2.82), text="", height=12, color=color.white, box=False)

# Chart board
chart_back = box(pos=vector(0, CHART_Y0 + CHART_H / 2, -16.7), size=vector(CHART_W + 2, CHART_H + 2, 0.22), color=vector(0.94, 0.97, 1.0))
chart_border_bottom = box(pos=vector(0, CHART_Y0 - 0.1, -16.55), size=vector(CHART_W, 0.08, 0.12), color=DARK)
chart_border_left = box(pos=vector(CHART_X0, CHART_Y0 + CHART_H / 2, -16.55), size=vector(0.08, CHART_H, 0.12), color=DARK)
chart_title = label(pos=vector(0, CHART_Y0 + CHART_H + 1.4, -16.5), text="PRICE HISTORY", height=14, color=DARK, box=False)
chart_min_lab = label(pos=vector(CHART_X0 - 2.2, CHART_Y0, -16.5), text="$10", height=10, color=DARK, box=False)
chart_max_lab = label(pos=vector(CHART_X0 - 2.2, CHART_Y0 + CHART_H, -16.5), text="$220", height=10, color=DARK, box=False)

# Order book board
book_back = box(pos=vector(ORDER_BOOK_X, 8.5, 0), size=vector(0.28, 17, 17), color=vector(0.95, 0.97, 1.0))
book_title = label(pos=vector(ORDER_BOOK_X + 0.35, 17.7, 0), text="ORDER BOOK", height=14, color=DARK, box=False)
bid_stack = []
ask_stack = []
for i in range(8):
    bid_stack.append(box(pos=vector(ORDER_BOOK_X + 0.55, 2.1 + i * 0.88, -3.5), size=vector(0.35, 0.55, 2.2), color=BUY_COLOR, opacity=0.35))
    ask_stack.append(box(pos=vector(ORDER_BOOK_X + 0.55, 2.1 + i * 0.88, 3.5), size=vector(0.35, 0.55, 2.2), color=SELL_COLOR, opacity=0.35))
book_labels = [
    label(pos=vector(ORDER_BOOK_X + 0.5, 1.0, -3.5), text="bids", height=10, color=BUY_COLOR, box=False),
    label(pos=vector(ORDER_BOOK_X + 0.5, 1.0, 3.5), text="asks", height=10, color=SELL_COLOR, box=False),
]

# Supply / demand pressure arrows
buy_arrow = arrow(pos=vector(-17, 2.0, 13), axis=vector(5, 0, 0), shaftwidth=0.55, color=BUY_COLOR)
sell_arrow = arrow(pos=vector(17, 2.0, 13), axis=vector(-5, 0, 0), shaftwidth=0.55, color=SELL_COLOR)
pressure_label = label(pos=vector(0, 4.0, 13), text="", height=13, color=DARK, box=False)

status = label(
    pos=vector(-28, 23.5, 0),
    text="",
    height=12,
    color=DARK,
    box=False,
    align="left",
)

controls = label(
    pos=vector(-28, -1.7, 17),
    text=(
        "Controls: Space pause | r reset | b buyer wave | s seller wave | n news shock | "
        "v/V volatility | l/L liquidity | a auto | c camera | 1-4 views"
    ),
    height=10,
    color=vector(0.12, 0.14, 0.18),
    box=False,
    align="left",
)

# ----------------------------- Dynamic classes -----------------------------
class Trader:
    def __init__(self, kind=None):
        self.kind = kind if kind else random.choice(["buyer", "seller"])
        self.home_x = random.uniform(-21, -7) if self.kind == "buyer" else random.uniform(7, 21)
        self.home_z = random.uniform(-12, 12)
        self.pos = vector(self.home_x, 0.52, self.home_z)
        self.target = vector(random.uniform(-3, 3), 0.52, random.uniform(-3, 3))
        self.speed = random.uniform(0.035, 0.105)
        self.activity = random.uniform(0.3, 1.0)
        self.cooldown = random.uniform(0, 3)
        self.body = sphere(pos=self.pos, radius=0.38, color=BUY_COLOR if self.kind == "buyer" else SELL_COLOR, opacity=0.88)
        self.head = sphere(pos=self.pos + vector(0, 0.52, 0), radius=0.20, color=vector(1.0, 0.86, 0.64), opacity=0.95)
        self.intent = random.uniform(0.2, 1.0)

    def reset_target(self):
        if random.random() < 0.58:
            self.target = vector(random.uniform(-2.4, 2.4), 0.52, random.uniform(-2.4, 2.4))
        else:
            self.target = vector(self.home_x + random.uniform(-2, 2), 0.52, self.home_z + random.uniform(-2, 2))

    def update(self):
        global buy_pressure, sell_pressure
        # Sentiment pulls traders: positive sentiment activates buyers; negative activates sellers.
        mood_boost = sentiment if self.kind == "buyer" else -sentiment
        self.activity = clamp(0.52 + mood_boost * 0.34 + random.uniform(-0.025, 0.025), 0.12, 1.6)
        direction = self.target - self.pos
        dist = mag(direction)
        if dist < 0.35:
            self.reset_target()
        else:
            self.pos += norm(direction) * self.speed * self.activity
        self.body.pos = self.pos
        self.head.pos = self.pos + vector(0, 0.52, 0)

        self.cooldown -= DT
        if self.cooldown <= 0 and random.random() < 0.045 * self.activity:
            qty = random.uniform(0.6, 2.6) * self.intent
            if self.kind == "buyer":
                buy_pressure += qty
            else:
                sell_pressure += qty
            create_order(self.kind, self.pos, qty)
            self.cooldown = random.uniform(0.6, 2.1) / max(0.25, self.activity)

    def set_kind(self, kind):
        self.kind = kind
        self.home_x = random.uniform(-21, -7) if self.kind == "buyer" else random.uniform(7, 21)
        self.home_z = random.uniform(-12, 12)
        self.body.color = BUY_COLOR if kind == "buyer" else SELL_COLOR
        self.reset_target()

    def delete(self):
        safe_delete(self.body)
        safe_delete(self.head)


class OrderPulse:
    def __init__(self, kind, start_pos, qty):
        self.kind = kind
        self.qty = qty
        self.pos = vector(start_pos.x, start_pos.y + 0.6, start_pos.z)
        self.target = vector(0, 2.3, 0)
        self.life = 1.0
        self.obj = sphere(pos=self.pos, radius=0.09 + 0.08 * min(qty, 3), color=BUY_COLOR if kind == "buyer" else SELL_COLOR, opacity=0.85, emissive=True)
        self.trail = []

    def update(self):
        direction = self.target - self.pos
        if mag(direction) > 0.08:
            self.pos += norm(direction) * (0.34 + 0.025 * self.qty)
        self.life -= 0.015
        self.obj.pos = self.pos
        self.obj.opacity = clamp(self.life, 0, 0.85)
        if random.random() < 0.22:
            dot = sphere(pos=vector(self.pos.x, self.pos.y, self.pos.z), radius=0.055, color=self.obj.color, opacity=0.35)
            self.trail.append([dot, 0.35])
        for item in list(self.trail):
            dot, op = item
            op -= 0.025
            dot.opacity = max(0, op)
            dot.radius *= 0.985
            item[1] = op
            if op <= 0.02:
                safe_delete(dot)
                self.trail.remove(item)
        return self.life > 0 and mag(direction) > 0.12

    def delete(self):
        safe_delete(self.obj)
        for dot, _ in self.trail:
            safe_delete(dot)
        self.trail.clear()


def create_order(kind, start_pos, qty):
    if len(orders) < 170:
        orders.append(OrderPulse(kind, start_pos, qty))

# ----------------------------- Simulation functions -----------------------------
def create_traders(n=44):
    for _ in range(n):
        traders.append(Trader())


def clear_dynamic_objects():
    for t in traders:
        t.delete()
    traders.clear()
    for o in orders:
        o.delete()
    orders.clear()
    for seg in chart_segments:
        safe_delete(seg)
    chart_segments.clear()
    for bar in volume_bars:
        safe_delete(bar)
    volume_bars.clear()
    for ar in pressure_arrows:
        safe_delete(ar)
    pressure_arrows.clear()
    for ripple in shock_ripples:
        safe_delete(ripple[0])
    shock_ripples.clear()


def reset_simulation():
    global price, last_price, volatility, liquidity, sentiment, news_timer, news_text
    global buy_pressure, sell_pressure, volume, spread, frame_count
    clear_dynamic_objects()
    price = PRICE_START
    last_price = PRICE_START
    volatility = 0.75
    liquidity = 1.35
    sentiment = 0.0
    news_timer = 0
    news_text = "balanced market"
    buy_pressure = 0.0
    sell_pressure = 0.0
    volume = 0.0
    spread = 1.2
    frame_count = 0
    price_history.clear()
    buy_history.clear()
    sell_history.clear()
    volume_history.clear()
    for _ in range(HISTORY_LEN):
        price_history.append(PRICE_START)
        buy_history.append(0.0)
        sell_history.append(0.0)
        volume_history.append(0.0)
    create_traders(44)


def buyer_wave():
    global sentiment, buy_pressure, news_text, news_timer
    sentiment = clamp(sentiment + 0.38, -1.5, 1.5)
    buy_pressure += random.uniform(18, 32)
    news_text = "buyer wave: demand surge"
    news_timer = 130
    for _ in range(6):
        traders.append(Trader("buyer"))
    create_news_ripple(BUY_COLOR)


def seller_wave():
    global sentiment, sell_pressure, news_text, news_timer
    sentiment = clamp(sentiment - 0.38, -1.5, 1.5)
    sell_pressure += random.uniform(18, 32)
    news_text = "seller wave: supply surge"
    news_timer = 130
    for _ in range(6):
        traders.append(Trader("seller"))
    create_news_ripple(SELL_COLOR)


def random_news_shock():
    global sentiment, buy_pressure, sell_pressure, news_text, news_timer, volatility
    choices = [
        ("earnings surprise", 0.65, BUY_COLOR),
        ("analyst upgrade", 0.42, BUY_COLOR),
        ("supply scare", -0.46, SELL_COLOR),
        ("regulatory worry", -0.68, SELL_COLOR),
        ("sector rotation", random.choice([-0.35, 0.35]), PURPLE),
        ("calm liquidity day", 0.05, YELLOW),
    ]
    text, impact, col = random.choice(choices)
    sentiment = clamp(sentiment + impact, -1.8, 1.8)
    if impact >= 0:
        buy_pressure += abs(impact) * random.uniform(20, 40)
    else:
        sell_pressure += abs(impact) * random.uniform(20, 40)
    volatility = clamp(volatility + abs(impact) * 0.22, 0.15, 3.5)
    news_text = "news: " + text
    news_timer = 160
    create_news_ripple(col)


def create_news_ripple(col):
    for r in [1.2, 2.0, 2.8]:
        obj = ring(pos=vector(0, 3.0, 0), axis=vector(0, 1, 0), radius=r, thickness=0.055, color=col, opacity=0.55)
        shock_ripples.append([obj, 0.55, random.uniform(0.06, 0.11)])


def update_news_ripples():
    for item in list(shock_ripples):
        obj, op, growth = item
        obj.radius += growth
        op -= 0.008
        obj.opacity = max(0, op)
        item[1] = op
        if op <= 0.02 or obj.radius > 11:
            safe_delete(obj)
            shock_ripples.remove(item)


def update_market():
    global price, last_price, buy_pressure, sell_pressure, sentiment, volume, spread, news_timer, news_text
    # Background order flow gives the market life even without manual input.
    base_flow = 8.0 + 2.0 * math.sin(frame_count * 0.018)
    buy_noise = max(0.0, random.gauss(base_flow * (1.0 + max(sentiment, 0) * 0.42), 2.7 * volatility))
    sell_noise = max(0.0, random.gauss(base_flow * (1.0 + max(-sentiment, 0) * 0.42), 2.7 * volatility))
    buy_pressure += buy_noise * DT * 2.4
    sell_pressure += sell_noise * DT * 2.4

    # Fundamental reversion is weak: price drifts toward anchor unless order imbalance dominates.
    imbalance = buy_pressure - sell_pressure
    total_flow = buy_pressure + sell_pressure
    volume = lerp(volume, total_flow, 0.13)
    liquidity_effect = max(0.25, liquidity)
    random_walk = random.gauss(0, 0.055 * volatility)
    price_change = (imbalance / (28.0 * liquidity_effect)) * volatility + random_walk
    price_change += (fundamental_anchor - price) * 0.0015

    last_price = price
    price = clamp(price + price_change, PRICE_MIN, PRICE_MAX)

    # Pressures decay as orders are matched.
    buy_pressure *= 0.87
    sell_pressure *= 0.87
    sentiment *= 0.997
    spread = clamp(0.45 + volatility * 0.75 + abs(imbalance) / 45 - liquidity * 0.14, 0.2, 6.5)

    if news_timer > 0:
        news_timer -= 1
    else:
        news_text = "balanced market" if abs(sentiment) < 0.12 else ("positive sentiment" if sentiment > 0 else "negative sentiment")

    price_history.append(price)
    buy_history.append(buy_pressure)
    sell_history.append(sell_pressure)
    volume_history.append(volume)


def update_traders_and_orders():
    for t in list(traders):
        t.update()
    # Keep population bounded.
    while len(traders) > 78:
        t = traders.pop(0)
        t.delete()
    for o in list(orders):
        alive = o.update()
        if not alive:
            o.delete()
            orders.remove(o)


def draw_price_chart():
    # Redraw lightweight chart every few frames.
    if frame_count % 3 != 0:
        return
    for seg in chart_segments:
        safe_delete(seg)
    chart_segments.clear()
    data = list(price_history)
    for i in range(1, len(data)):
        x1 = history_index_to_x(i - 1)
        x2 = history_index_to_x(i)
        y1 = price_to_chart_y(data[i - 1])
        y2 = price_to_chart_y(data[i])
        col = GREEN if data[i] >= data[i - 1] else RED
        seg = curve(pos=[vector(x1, y1, -16.35), vector(x2, y2, -16.35)], color=col, radius=0.035)
        chart_segments.append(seg)

    # Volume bars below chart, sampled for performance.
    for bar in volume_bars:
        safe_delete(bar)
    volume_bars.clear()
    hist = list(volume_history)
    max_vol = max(10.0, max(hist))
    for i in range(0, HISTORY_LEN, 8):
        x = history_index_to_x(i)
        h = clamp(hist[i] / max_vol, 0, 1) * 2.8
        bar = box(pos=vector(x, CHART_Y0 - 1.75 + h / 2, -16.35), size=vector(0.95, h, 0.12), color=vector(0.42, 0.48, 0.58), opacity=0.55)
        volume_bars.append(bar)


def update_order_book():
    bid_depth = clamp((buy_pressure + 8) / 32, 0.12, 2.4)
    ask_depth = clamp((sell_pressure + 8) / 32, 0.12, 2.4)
    for i, b in enumerate(bid_stack):
        level_factor = 1.0 - i * 0.07
        b.size.z = 0.8 + 2.8 * bid_depth * level_factor
        b.opacity = clamp(0.22 + bid_depth * 0.18, 0.2, 0.82)
    for i, a in enumerate(ask_stack):
        level_factor = 1.0 - i * 0.07
        a.size.z = 0.8 + 2.8 * ask_depth * level_factor
        a.opacity = clamp(0.22 + ask_depth * 0.18, 0.2, 0.82)


def update_visuals():
    change = price - last_price
    price_label.text = f"${price:0.2f}"
    price_label.color = GREEN if change >= 0 else RED
    market_label.text = f"spread {spread:0.2f} | vol {volatility:0.2f} | liquidity {liquidity:0.2f}"

    b_len = clamp(2.2 + buy_pressure / 5.5, 2.0, 12.0)
    s_len = clamp(2.2 + sell_pressure / 5.5, 2.0, 12.0)
    buy_arrow.axis = vector(b_len, 0, 0)
    sell_arrow.axis = vector(-s_len, 0, 0)
    buy_arrow.shaftwidth = clamp(0.25 + buy_pressure / 40, 0.25, 1.15)
    sell_arrow.shaftwidth = clamp(0.25 + sell_pressure / 40, 0.25, 1.15)

    net = buy_pressure - sell_pressure
    pressure_label.text = f"net pressure: {net:+0.1f}   volume: {volume:0.1f}   {news_text}"
    pressure_label.color = GREEN if net > 1 else RED if net < -1 else DARK

    # Exchange ring pulses with volatility.
    exchange_ring.radius = 3.1 + 0.18 * math.sin(frame_count * 0.16) * volatility
    exchange_ring.color = YELLOW if abs(sentiment) < 0.25 else (BUY_COLOR if sentiment > 0 else SELL_COLOR)

    update_order_book()
    draw_price_chart()
    update_news_ripples()

    status.text = (
        "STOCK MARKET BEHAVIOR\n"
        f"price: ${price:0.2f}\n"
        f"change: {price - PRICE_START:+0.2f} from open\n"
        f"buy demand: {buy_pressure:0.1f}\n"
        f"sell supply: {sell_pressure:0.1f}\n"
        f"sentiment: {sentiment:+0.2f}\n"
        f"volatility: {volatility:0.2f}\n"
        f"liquidity: {liquidity:0.2f}\n"
        f"auto cycles: {'on' if auto_market_cycles else 'off'}\n"
        f"camera: {'on' if cinematic_camera else 'manual'}"
    )


def update_auto_cycles():
    if not auto_market_cycles:
        return
    # Periodically inject market regimes.
    if frame_count > 40 and frame_count % 520 == 0:
        random_news_shock()
    if frame_count % 760 == 240:
        buyer_wave()
    if frame_count % 760 == 610:
        seller_wave()


def update_camera():
    if not cinematic_camera:
        return
    t = frame_count * DT
    cycle = (frame_count // 360) % 4
    if camera_mode != 0:
        cycle = camera_mode - 1

    if cycle == 0:  # whole market orbit
        radius = 48
        scene.center = vector(0, 7.5, 0)
        scene.camera.pos = vector(math.cos(t * 0.28) * radius, 28, math.sin(t * 0.28) * radius)
        scene.camera.axis = scene.center - scene.camera.pos
        scene.range = 31
    elif cycle == 1:  # price chart view
        scene.center = vector(0, 18, -16.4)
        scene.camera.pos = vector(0, 19, 26)
        scene.camera.axis = scene.center - scene.camera.pos
        scene.range = 16
    elif cycle == 2:  # trading floor low follow around exchange
        scene.center = vector(0, 2.5, 0)
        scene.camera.pos = vector(math.cos(t * 0.45) * 24, 7.5, math.sin(t * 0.45) * 24)
        scene.camera.axis = scene.center - scene.camera.pos
        scene.range = 19
    else:  # order book close view
        scene.center = vector(ORDER_BOOK_X, 8.3, 0)
        scene.camera.pos = vector(42, 10, 15)
        scene.camera.axis = scene.center - scene.camera.pos
        scene.range = 10


def keydown(evt):
    global paused, volatility, liquidity, auto_market_cycles, cinematic_camera, camera_mode
    key = evt.key
    if key == " ":
        paused = not paused
    elif key == "r":
        reset_simulation()
    elif key == "b":
        buyer_wave()
    elif key == "s":
        seller_wave()
    elif key == "n":
        random_news_shock()
    elif key == "v":
        volatility = clamp(volatility + 0.18, 0.15, 3.5)
    elif key == "V":
        volatility = clamp(volatility - 0.18, 0.15, 3.5)
    elif key == "l":
        liquidity = clamp(liquidity + 0.18, 0.3, 4.0)
    elif key == "L":
        liquidity = clamp(liquidity - 0.18, 0.3, 4.0)
    elif key == "a":
        auto_market_cycles = not auto_market_cycles
    elif key == "c":
        cinematic_camera = not cinematic_camera
    elif key in ["1", "2", "3", "4"]:
        camera_mode = int(key)
        cinematic_camera = True


scene.bind("keydown", keydown)

# ----------------------------- Main loop -----------------------------
reset_simulation()

while True:
    rate(60)
    if paused:
        update_camera()
        continue

    frame_count += 1
    update_auto_cycles()
    update_traders_and_orders()
    update_market()
    update_visuals()
    update_camera()
