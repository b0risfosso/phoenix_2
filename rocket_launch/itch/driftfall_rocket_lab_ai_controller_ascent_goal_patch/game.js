const G = 9.81;
const AIR_DENSITY = 1.225;
const DT = 0.05;
const MAX_GRAPH_POINTS = 180;

// Performance controls.
// The original terminal-style output can become heavy during long parachute descents.
// These settings keep the browser game responsive by limiting DOM log growth and
// allowing stable parachute descent to advance faster.
const MAX_TERMINAL_LINES = 180;
const NORMAL_STEPS_PER_FRAME = 2;
const FAST_DESCENT_STEPS_PER_FRAME = 10;
const FAST_DESCENT_ALTITUDE_THRESHOLD = 900;
const FAST_DESCENT_SPEED_STABILITY = 0.25;

const missions = [
  {
    title: "First Launch",
    description: "Reach 2,000 m and recover the rocket.",
    target: "Altitude ≥ 2000 m, landing speed ≥ -15 m/s",
    test: s => s.maxAltitude >= 2000 && s.landingSpeed >= -15,
    reward: 100
  },
  {
    title: "Maximize Altitude",
    description: "Tune the rocket to climb beyond 3,200 m.",
    target: "Altitude ≥ 3200 m",
    test: s => s.maxAltitude >= 3200,
    reward: 140
  },
  {
    title: "Soft Recovery",
    description: "Use parachute design and deploy altitude to reduce landing speed.",
    target: "Landing speed ≥ -10 m/s",
    test: s => s.landingSpeed >= -10,
    reward: 160
  },
  {
    title: "Drift Limit",
    description: "Reach a good altitude without letting wind carry the rocket too far.",
    target: "Altitude ≥ 2500 m, drift ≤ 220 m",
    test: s => s.maxAltitude >= 2500 && Math.abs(s.finalDrift) <= 220,
    reward: 180
  },
  {
    title: "Efficient Climb",
    description: "Reach altitude with a limited fuel load.",
    target: "Altitude ≥ 3000 m, starting fuel ≤ 95 kg",
    test: s => s.maxAltitude >= 3000 && s.startFuel <= 95,
    reward: 220
  },
  {
    title: "High Speed Research",
    description: "Push the rocket into a fast ascent while still recovering it.",
    target: "Max speed ≥ 230 m/s, landing speed ≥ -16 m/s",
    test: s => s.maxSpeed >= 230 && s.landingSpeed >= -16,
    reward: 250
  },
  {
    title: "Precision Recovery",
    description: "Land softly and keep drift under control.",
    target: "Landing speed ≥ -9.5 m/s, drift ≤ 260 m",
    test: s => s.landingSpeed >= -9.5 && Math.abs(s.finalDrift) <= 260,
    reward: 300
  }
];

const controls = {
  thrust: document.getElementById("thrust"),
  fuel: document.getElementById("fuel"),
  dryMass: document.getElementById("dryMass"),
  burnRate: document.getElementById("burnRate"),
  drag: document.getElementById("drag"),
  area: document.getElementById("area"),
  parachuteArea: document.getElementById("parachuteArea"),
  deployAltitude: document.getElementById("deployAltitude"),
  wind: document.getElementById("wind")
};

const valueSpans = {
  thrust: document.getElementById("thrustValue"),
  fuel: document.getElementById("fuelValue"),
  dryMass: document.getElementById("dryMassValue"),
  burnRate: document.getElementById("burnRateValue"),
  drag: document.getElementById("dragValue"),
  area: document.getElementById("areaValue"),
  parachuteArea: document.getElementById("parachuteAreaValue"),
  deployAltitude: document.getElementById("deployAltitudeValue"),
  wind: document.getElementById("windValue")
};

const ai = {
  enabled: document.getElementById("aiEnabled"),
  goal: document.getElementById("aiGoal"),
  aggression: document.getElementById("aiAggression"),
  delay: document.getElementById("aiDelay"),
  aggressionValue: document.getElementById("aiAggressionValue"),
  delayValue: document.getElementById("aiDelayValue"),
  tuneButton: document.getElementById("aiTuneButton"),
  stopButton: document.getElementById("aiStopButton"),
  status: document.getElementById("aiStatus"),
  pendingTimer: null,
  launchCount: 0,
  lastSummary: null,
  bestByGoal: null,
  goalChangePending: false,
  suppressAutoSchedule: false
};

const ui = {
  missionNumber: document.getElementById("missionNumber"),
  credits: document.getElementById("credits"),
  bestAltitude: document.getElementById("bestAltitude"),
  missionTitle: document.getElementById("missionTitle"),
  missionDescription: document.getElementById("missionDescription"),
  missionTarget: document.getElementById("missionTarget"),
  launchButton: document.getElementById("launchButton"),
  resetButton: document.getElementById("resetButton"),
  rocket: document.getElementById("rocket"),
  flame: document.getElementById("flame"),
  chute: document.getElementById("chute"),
  sky: document.getElementById("sky"),
  time: document.getElementById("time"),
  altitude: document.getElementById("altitude"),
  velocity: document.getElementById("velocity"),
  acceleration: document.getElementById("acceleration"),
  fuelReadout: document.getElementById("fuelReadout"),
  mass: document.getElementById("mass"),
  drift: document.getElementById("drift"),
  phase: document.getElementById("phase"),
  terminal: document.getElementById("terminal"),
  summary: document.getElementById("summary")
};

const altitudeCanvas = document.getElementById("altitudeGraph");
const velocityCanvas = document.getElementById("velocityGraph");
const altitudeCtx = altitudeCanvas.getContext("2d");
const velocityCtx = velocityCanvas.getContext("2d");

let missionIndex = 0;
let credits = 0;
let bestAltitude = 0;
let running = false;
let paused = false;
let launched = false;
let successReadyForNext = false;
let altitudeHistory = [];
let velocityHistory = [];
let terminalLines = [];
let frameDrawCounter = 0;

let state = null;
let startParams = null;

function defaultControls() {
  controls.thrust.value = 3600;
  controls.fuel.value = 95;
  controls.dryMass.value = 55;
  controls.burnRate.value = 5.8;
  controls.drag.value = 0.55;
  controls.area.value = 0.22;
  controls.parachuteArea.value = 5;
  controls.deployAltitude.value = 650;
  controls.wind.value = 1.2;
}

function readParams() {
  return {
    thrust: Number(controls.thrust.value),
    fuel: Number(controls.fuel.value),
    dryMass: Number(controls.dryMass.value),
    burnRate: Number(controls.burnRate.value),
    cd: Number(controls.drag.value),
    area: Number(controls.area.value),
    parachuteArea: Number(controls.parachuteArea.value),
    deployAltitude: Number(controls.deployAltitude.value),
    wind: Number(controls.wind.value)
  };
}

function updateControlLabels() {
  valueSpans.thrust.textContent = `${Number(controls.thrust.value).toFixed(0)} N`;
  valueSpans.fuel.textContent = `${Number(controls.fuel.value).toFixed(0)} kg`;
  valueSpans.dryMass.textContent = `${Number(controls.dryMass.value).toFixed(0)} kg`;
  valueSpans.burnRate.textContent = `${Number(controls.burnRate.value).toFixed(1)} kg/s`;
  valueSpans.drag.textContent = `${Number(controls.drag.value).toFixed(2)}`;
  valueSpans.area.textContent = `${Number(controls.area.value).toFixed(2)} m²`;
  valueSpans.parachuteArea.textContent = `${Number(controls.parachuteArea.value).toFixed(1)} m²`;
  valueSpans.deployAltitude.textContent = `${Number(controls.deployAltitude.value).toFixed(0)} m`;
  valueSpans.wind.textContent = `${Number(controls.wind.value).toFixed(1)} m/s`;
}

function loadMission() {
  const m = missions[missionIndex];
  ui.missionNumber.textContent = String(missionIndex + 1);
  ui.credits.textContent = String(credits);
  ui.bestAltitude.textContent = `${bestAltitude.toFixed(0)} m`;
  ui.missionTitle.textContent = m.title;
  ui.missionDescription.textContent = m.description;
  ui.missionTarget.textContent = m.target;
  successReadyForNext = false;
}

function resetState() {
  const params = readParams();
  startParams = params;
  state = {
    t: 0,
    altitude: 0,
    velocity: 0,
    acceleration: 0,
    fuel: params.fuel,
    mass: params.dryMass + params.fuel,
    drift: 0,
    phase: "on pad",
    engineOn: true,
    parachuteDeployed: false,
    launched: false,
    burnoutTime: null,
    maxAltitude: 0,
    maxSpeed: 0,
    maxAccel: 0,
    landingSpeed: 0,
    finalDrift: 0,
    landed: false,
    startFuel: params.fuel
  };
  running = false;
  paused = false;
  launched = false;
  altitudeHistory = [];
  velocityHistory = [];
  terminalLines = [];
  ui.terminal.textContent = "";
  ui.summary.textContent = "No launch yet.";
  ui.launchButton.textContent = "LAUNCH";
  updateTelemetry();
  drawGraphs();
  positionRocket();
}

function signDragForce(velocity, dragMagnitude) {
  if (velocity > 0) return -dragMagnitude;
  if (velocity < 0) return dragMagnitude;
  return 0;
}

function determinePhase() {
  if (state.altitude <= 0.01 && state.t < 0.2) return "on pad";
  if (state.engineOn && state.velocity >= 0) return "powered ascent";
  if (!state.engineOn && state.velocity > 2) return "coasting upward";
  if (Math.abs(state.velocity) <= 2 && state.altitude > 1) return "near apogee";
  if (state.parachuteDeployed && state.velocity < 0) return "parachute descent";
  if (state.velocity < 0) return "ballistic descent";
  return "flight";
}

function stepSimulation() {
  if (!running || paused || state.landed) return;

  const p = startParams;

  state.mass = p.dryMass + state.fuel;

  let thrust = 0;
  if (state.fuel > 0) {
    const fuelBurn = Math.min(p.burnRate * DT, state.fuel);
    state.fuel -= fuelBurn;
    thrust = p.thrust;
    state.engineOn = true;
  } else {
    thrust = 0;
    state.engineOn = false;
    if (state.burnoutTime === null) state.burnoutTime = state.t;
  }

  if (state.altitude > 0.1) state.launched = true;

  if (!state.parachuteDeployed && state.launched && state.velocity < 0 && state.altitude <= p.deployAltitude) {
    state.parachuteDeployed = true;
    addTerminalLine(">>> PARACHUTE DEPLOYED");
  }

  let activeArea = p.area;
  let activeCd = p.cd;
  if (state.parachuteDeployed) {
    activeArea += p.parachuteArea;
    activeCd = Math.max(activeCd, 1.3);
  }

  const dragMagnitude = 0.5 * AIR_DENSITY * activeCd * activeArea * state.velocity * state.velocity;
  const dragForce = signDragForce(state.velocity, dragMagnitude);
  const gravityForce = state.mass * G;
  const netForce = thrust + dragForce - gravityForce;

  state.acceleration = netForce / Math.max(state.mass, 0.001);
  state.velocity += state.acceleration * DT;
  state.altitude += state.velocity * DT;

  const driftMultiplier = state.parachuteDeployed ? 2.5 : 1.0;
  state.drift += p.wind * driftMultiplier * DT;

  state.t += DT;

  state.maxAltitude = Math.max(state.maxAltitude, state.altitude);
  state.maxSpeed = Math.max(state.maxSpeed, Math.abs(state.velocity));
  state.maxAccel = Math.max(state.maxAccel, Math.abs(state.acceleration));
  bestAltitude = Math.max(bestAltitude, state.maxAltitude);

  state.phase = determinePhase();

  altitudeHistory.push(state.altitude);
  velocityHistory.push(state.velocity);
  if (altitudeHistory.length > MAX_GRAPH_POINTS) altitudeHistory.shift();
  if (velocityHistory.length > MAX_GRAPH_POINTS) velocityHistory.shift();

  maybeLogTelemetry();

  if (state.launched && state.altitude <= 0) {
    state.altitude = 0;
    state.landingSpeed = state.velocity;
    state.finalDrift = state.drift;
    state.landed = true;
    running = false;
    finishLaunch();
  }

  updateTelemetry();
  positionRocket();

  frameDrawCounter += 1;
  if (frameDrawCounter % 3 === 0 || state.landed) {
    drawGraphs();
  }
}

let lastLoggedSecond = -1;
function maybeLogTelemetry() {
  // Normal flight logs twice per second. Stable parachute descent logs less often
  // because it can last a long time with nearly identical values.
  const stableParachute = state.parachuteDeployed && Math.abs(state.acceleration) < FAST_DESCENT_SPEED_STABILITY;
  const divisor = stableParachute ? 1 : 2;
  const logSlot = Math.floor(state.t * divisor);
  if (logSlot === lastLoggedSecond) return;
  lastLoggedSecond = logSlot;

  const line =
    `t=${state.t.toFixed(2).padStart(6, "0")}s ` +
    `alt=${state.altitude.toFixed(1).padStart(8)} m ` +
    `vel=${state.velocity.toFixed(1).padStart(7)} m/s ` +
    `acc=${state.acceleration.toFixed(1).padStart(7)} m/s² ` +
    `fuel=${state.fuel.toFixed(1).padStart(6)} kg ` +
    `drift=${state.drift.toFixed(1).padStart(7)} m ` +
    `phase=${state.phase}`;
  addTerminalLine(line);
}

function addTerminalLine(line) {
  terminalLines.push(line);
  if (terminalLines.length > MAX_TERMINAL_LINES) {
    terminalLines = terminalLines.slice(terminalLines.length - MAX_TERMINAL_LINES);
  }
  ui.terminal.textContent = terminalLines.join("\n") + "\n";
  ui.terminal.scrollTop = ui.terminal.scrollHeight;
}

function finishLaunch() {
  const m = missions[missionIndex];
  const summary = {
    maxAltitude: state.maxAltitude,
    maxSpeed: state.maxSpeed,
    maxAccel: state.maxAccel,
    burnoutTime: state.burnoutTime ?? state.t,
    flightTime: state.t,
    landingSpeed: state.landingSpeed,
    finalDrift: state.finalDrift,
    startFuel: state.startFuel
  };

  const passed = m.test(summary);
  let result = "hard landing";
  if (summary.landingSpeed >= -9) result = "soft recovery";
  else if (summary.landingSpeed >= -16) result = "rough parachute landing";
  if (!state.launched) result = "failed to lift off";

  if (passed) {
    credits += m.reward;
    successReadyForNext = true;
  }

  ui.summary.innerHTML =
    `<div class="${passed ? "pass" : "fail"}">${passed ? "MISSION PASSED" : "MISSION FAILED"}</div>` +
    `<div>Result: ${result}</div>` +
    `<div>Max altitude: ${summary.maxAltitude.toFixed(1)} m</div>` +
    `<div>Max speed: ${summary.maxSpeed.toFixed(1)} m/s</div>` +
    `<div>Max acceleration: ${summary.maxAccel.toFixed(1)} m/s²</div>` +
    `<div>Burnout time: ${summary.burnoutTime.toFixed(1)} s</div>` +
    `<div>Flight time: ${summary.flightTime.toFixed(1)} s</div>` +
    `<div>Landing speed: ${summary.landingSpeed.toFixed(2)} m/s</div>` +
    `<div>Final drift: ${summary.finalDrift.toFixed(1)} m</div>` +
    `<div>Reward: ${passed ? m.reward : 0} credits</div>`;

  addTerminalLine("------------------------------------------------------------");
  addTerminalLine(
    `summary: max_altitude=${summary.maxAltitude.toFixed(2)} m, ` +
    `max_speed=${summary.maxSpeed.toFixed(2)} m/s, ` +
    `max_acceleration=${summary.maxAccel.toFixed(2)} m/s², ` +
    `burnout_time=${summary.burnoutTime.toFixed(2)} s, ` +
    `flight_time=${summary.flightTime.toFixed(2)} s, ` +
    `landing_speed=${summary.landingSpeed.toFixed(2)} m/s, ` +
    `drift=${summary.finalDrift.toFixed(2)} m, result=${result}`
  );

  if (passed && missionIndex < missions.length - 1) {
    addTerminalLine(">>> Press N for next mission.");
  }

  ai.lastSummary = {
    ...summary,
    passed,
    result,
    missionIndex,
    params: {...startParams}
  };
  ai.launchCount += 1;
  updateAiStatusAfterLaunch(ai.lastSummary);

  ui.launchButton.textContent = "LAUNCH AGAIN";
  loadMission();

  if (ai.enabled.checked) {
    ai.goalChangePending = false;
    scheduleAiNextLaunch();
  }
}


function setControlValue(name, value) {
  const input = controls[name];
  const min = Number(input.min);
  const max = Number(input.max);
  const step = Number(input.step || 1);
  const clamped = Math.max(min, Math.min(max, value));
  const rounded = Math.round(clamped / step) * step;
  input.value = String(rounded);
}

function clearAiTimer() {
  if (ai.pendingTimer) {
    clearTimeout(ai.pendingTimer);
    ai.pendingTimer = null;
  }
}

function selectedAiGoal() {
  return ai.goal.value;
}

function scoreSummaryForGoal(summary, goal) {
  if (!summary) return 0;

  if (goal === "mission") {
    const mission = missions[summary.missionIndex] || missions[missionIndex];
    let score = 0;
    score += Math.min(summary.maxAltitude / 7000, 1.25) * 35;
    score += Math.max(0, 1 - Math.abs(summary.landingSpeed) / 25) * 35;
    score += Math.max(0, 1 - Math.abs(summary.finalDrift) / 700) * 20;
    score += summary.passed ? 30 : 0;
    return score;
  }

  if (goal === "altitude") {
    return summary.maxAltitude / 70 - Math.abs(summary.landingSpeed) * 0.8;
  }

  if (goal === "soft") {
    return 120 - Math.abs(summary.landingSpeed) * 8 + Math.min(summary.maxAltitude / 3000, 1) * 12;
  }

  if (goal === "drift") {
    return 100 - Math.abs(summary.finalDrift) * 0.25 + Math.min(summary.maxAltitude / 3500, 1) * 20;
  }

  if (goal === "efficient") {
    return summary.maxAltitude / Math.max(1, summary.startFuel) - Math.abs(summary.landingSpeed) * 0.35;
  }

  if (goal === "speed") {
    return summary.maxSpeed - Math.abs(summary.landingSpeed) * 1.2;
  }

  // balanced
  return (
    Math.min(summary.maxAltitude / 4000, 1.3) * 45 +
    Math.max(0, 1 - Math.abs(summary.landingSpeed) / 18) * 35 +
    Math.max(0, 1 - Math.abs(summary.finalDrift) / 500) * 20
  );
}

function randomSigned(amount) {
  return (Math.random() * 2 - 1) * amount;
}

function tuneParametersWithAI() {
  const goal = selectedAiGoal();
  const aggression = Number(ai.aggression.value);
  const p = readParams();
  const s = ai.lastSummary;
  const score = scoreSummaryForGoal(s, goal);

  if (!ai.bestByGoal || ai.bestByGoal.goal !== goal || score > ai.bestByGoal.score) {
    if (s) ai.bestByGoal = {goal, score, params: {...s.params}, summary: {...s}};
  }

  let next = {...p};

  // Goal-driven tuning.
  if (goal === "mission") {
    const mission = missions[missionIndex];

    if (mission.title.includes("Altitude") || mission.title.includes("Climb") || mission.title.includes("Research")) {
      next.thrust += 500 * aggression;
      next.fuel += 10 * aggression;
      next.drag -= 0.035 * aggression;
      next.area -= 0.02 * aggression;
    }

    if (mission.title.includes("Recovery") || mission.title.includes("Soft") || mission.title.includes("Precision")) {
      next.parachuteArea += 1.3 * aggression;
      next.deployAltitude += 120 * aggression;
      next.wind *= 0.95;
    }

    if (mission.title.includes("Drift")) {
      next.parachuteArea -= 0.6 * aggression;
      next.deployAltitude -= 130 * aggression;
      next.drag -= 0.02 * aggression;
      next.area -= 0.015 * aggression;
    }

    if (mission.title.includes("Efficient")) {
      next.fuel -= 8 * aggression;
      next.drag -= 0.04 * aggression;
      next.area -= 0.025 * aggression;
      next.burnRate -= 0.25 * aggression;
      next.thrust += 180 * aggression;
    }
  }

  if (goal === "altitude") {
    next.thrust += 700 * aggression;
    next.fuel += 14 * aggression;
    next.burnRate += 0.25 * aggression;
    next.drag -= 0.045 * aggression;
    next.area -= 0.025 * aggression;
    next.dryMass -= 2.5 * aggression;
  } else if (goal === "soft") {
    next.parachuteArea += 1.8 * aggression;
    next.deployAltitude += 180 * aggression;
    next.thrust *= 0.99;
    next.fuel *= 0.99;
  } else if (goal === "drift") {
    next.parachuteArea -= 0.9 * aggression;
    next.deployAltitude -= 170 * aggression;
    next.drag -= 0.025 * aggression;
    next.area -= 0.015 * aggression;
    next.fuel -= 4 * aggression;
  } else if (goal === "efficient") {
    next.fuel -= 11 * aggression;
    next.burnRate -= 0.35 * aggression;
    next.drag -= 0.05 * aggression;
    next.area -= 0.025 * aggression;
    next.thrust += 120 * aggression;
    next.dryMass -= 2.0 * aggression;
  } else if (goal === "speed") {
    next.thrust += 900 * aggression;
    next.burnRate += 0.75 * aggression;
    next.fuel += 5 * aggression;
    next.drag -= 0.035 * aggression;
  } else if (goal === "balanced") {
    next.thrust += 250 * aggression;
    next.fuel += 3 * aggression;
    next.drag -= 0.02 * aggression;
    next.area -= 0.01 * aggression;
    next.parachuteArea += 0.8 * aggression;
    next.deployAltitude += 80 * aggression;
  }

  // Reactive corrections based on last flight.
  if (s) {
    if (s.maxAltitude < 2000) {
      next.thrust += 650 * aggression;
      next.fuel += 12 * aggression;
    }

    if (s.landingSpeed < -14) {
      next.parachuteArea += 1.2 * aggression;
      next.deployAltitude += 130 * aggression;
    }

    if (Math.abs(s.finalDrift) > 350) {
      next.parachuteArea -= 0.7 * aggression;
      next.deployAltitude -= 110 * aggression;
      next.wind *= 0.9;
    }

    if (s.maxSpeed < 160 && (goal === "speed" || goal === "altitude")) {
      next.thrust += 450 * aggression;
      next.burnRate += 0.35 * aggression;
    }

    // If the last attempt passed, slightly explore instead of overcorrecting.
    if (s.passed) {
      next.thrust += randomSigned(120 * aggression);
      next.fuel += randomSigned(4 * aggression);
      next.parachuteArea += randomSigned(0.4 * aggression);
      next.deployAltitude += randomSigned(45 * aggression);
    }
  }

  // Exploration noise. This prevents the AI from repeating identical launches.
  next.thrust += randomSigned(220 * aggression);
  next.fuel += randomSigned(6 * aggression);
  next.burnRate += randomSigned(0.22 * aggression);
  next.drag += randomSigned(0.018 * aggression);
  next.area += randomSigned(0.01 * aggression);
  next.parachuteArea += randomSigned(0.35 * aggression);
  next.deployAltitude += randomSigned(40 * aggression);

  setControlValue("thrust", next.thrust);
  setControlValue("fuel", next.fuel);
  setControlValue("dryMass", next.dryMass);
  setControlValue("burnRate", next.burnRate);
  setControlValue("drag", next.drag);
  setControlValue("area", next.area);
  setControlValue("parachuteArea", next.parachuteArea);
  setControlValue("deployAltitude", next.deployAltitude);
  setControlValue("wind", next.wind);

  updateControlLabels();
  if (!running) resetState();

  ai.status.textContent =
    `AI tuned for "${ai.goal.options[ai.goal.selectedIndex].text}". ` +
    `Aggression ${Number(ai.aggression.value).toFixed(2)}. ` +
    (s ? `Last score ${score.toFixed(1)}.` : "No prior launch yet.");
  addTerminalLine(">>> AI tuned parameters for goal: " + ai.goal.options[ai.goal.selectedIndex].text);
}

function scheduleAiNextLaunch() {
  clearAiTimer();

  if (ai.suppressAutoSchedule) return;

  // Never schedule a new launch while one is already flying or paused in flight.
  // This prevents goal changes from stacking timers during ascent.
  if (running || (state && !state.landed && state.launched)) {
    ai.status.textContent = "AI goal queued. Current launch will finish before retuning.";
    return;
  }

  // If mission passed, advance to next mission automatically when possible.
  if (successReadyForNext && missionIndex < missions.length - 1 && selectedAiGoal() === "mission") {
    nextMission();
  }

  const delayMs = Number(ai.delay.value) * 1000;
  ai.status.textContent = `AI waiting ${Number(ai.delay.value).toFixed(1)} s, then tuning and launching.`;

  ai.pendingTimer = setTimeout(() => {
    ai.pendingTimer = null;
    if (!ai.enabled.checked) return;
    if (running || (state && !state.landed && state.launched)) return;
    tuneParametersWithAI();
    if (!running) launchOrPause();
  }, delayMs);
}

function updateAiStatusAfterLaunch(summary) {
  const goal = selectedAiGoal();
  const score = scoreSummaryForGoal(summary, goal);
  ai.status.textContent =
    `Launch ${ai.launchCount} complete. Goal score ${score.toFixed(1)}. ` +
    `${summary.passed ? "Mission passed." : "Mission not yet passed."}`;
}

function stopAI() {
  ai.enabled.checked = false;
  clearAiTimer();
  ai.goalChangePending = false;
  ai.suppressAutoSchedule = false;
  ai.status.textContent = "AI stopped.";
  addTerminalLine(">>> AI STOPPED");
}

function updateTelemetry() {
  if (!state) return;
  ui.time.textContent = `${state.t.toFixed(2)} s`;
  ui.altitude.textContent = `${Math.max(0, state.altitude).toFixed(2)} m`;
  ui.velocity.textContent = `${state.velocity.toFixed(2)} m/s`;
  ui.acceleration.textContent = `${state.acceleration.toFixed(2)} m/s²`;
  ui.fuelReadout.textContent = `${Math.max(0, state.fuel).toFixed(2)} kg`;
  ui.mass.textContent = `${state.mass.toFixed(2)} kg`;
  ui.drift.textContent = `${state.drift.toFixed(2)} m`;
  ui.phase.textContent = state.phase;

  ui.flame.style.opacity = state.engineOn && running && !paused ? "1" : "0";
  ui.chute.style.display = state.parachuteDeployed ? "block" : "none";
}

function positionRocket() {
  if (!state) return;

  const skyHeight = ui.sky.clientHeight - 32;
  const altitudeCap = 7000;
  const y = Math.min(state.altitude / altitudeCap, 1) * (skyHeight - 92);
  const x = Math.max(-45, Math.min(45, state.drift / 8));
  const tilt = Math.max(-18, Math.min(18, state.velocity < 0 ? -x / 4 : x / 5));

  ui.rocket.style.transform = `translateX(calc(-50% + ${x}px)) translateY(${-y}px) rotate(${tilt}deg)`;
}

function drawGraph(ctx, data, label, color, symmetric=false) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "rgba(0,0,0,0.18)";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(220,238,255,0.18)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = i * h / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(220,238,255,0.75)";
  ctx.font = "13px monospace";
  ctx.fillText(label, 10, 18);

  if (data.length < 2) return;

  let min = Math.min(...data);
  let max = Math.max(...data);
  if (symmetric) {
    const m = Math.max(Math.abs(min), Math.abs(max), 1);
    min = -m;
    max = m;
  } else {
    min = Math.min(0, min);
    max = Math.max(1, max);
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();

  data.forEach((v, i) => {
    const x = i / (MAX_GRAPH_POINTS - 1) * w;
    const y = h - ((v - min) / (max - min)) * (h - 26) - 8;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

function drawGraphs() {
  drawGraph(altitudeCtx, altitudeHistory, "ALTITUDE", "#65ffb8", false);
  drawGraph(velocityCtx, velocityHistory, "VELOCITY", "#ffd166", true);
}

function launchOrPause() {
  if (!running && state && state.landed) {
    resetState();
  }

  if (!running) {
    clearAiTimer();
    startParams = readParams();
    resetState();
    startParams = readParams();
    state.mass = startParams.dryMass + startParams.fuel;
    running = true;
    paused = false;
    launched = true;
    lastLoggedSecond = -1;
    ui.launchButton.textContent = "PAUSE";
    addTerminalLine("============================================================");
    addTerminalLine(`MISSION ${missionIndex + 1}: ${missions[missionIndex].title}`);
    addTerminalLine(
      `parameters: thrust=${startParams.thrust.toFixed(1)} N, fuel=${startParams.fuel.toFixed(1)} kg, ` +
      `dry_mass=${startParams.dryMass.toFixed(1)} kg, burn_rate=${startParams.burnRate.toFixed(2)} kg/s, ` +
      `Cd=${startParams.cd.toFixed(3)}, area=${startParams.area.toFixed(3)} m², ` +
      `parachute_area=${startParams.parachuteArea.toFixed(2)} m², deploy_altitude=${startParams.deployAltitude.toFixed(1)} m`
    );
  } else {
    paused = !paused;
    ui.launchButton.textContent = paused ? "RESUME" : "PAUSE";
    addTerminalLine(paused ? ">>> PAUSED" : ">>> RESUMED");
  }
}

function nextMission() {
  if (successReadyForNext && missionIndex < missions.length - 1) {
    clearAiTimer();
    missionIndex += 1;
    loadMission();
    resetState();
    addTerminalLine(`>>> Next mission loaded: ${missions[missionIndex].title}`);
  }
}

function currentStepsPerFrame() {
  if (!state || !running || paused || state.landed) return NORMAL_STEPS_PER_FRAME;

  const stableParachuteDescent =
    state.parachuteDeployed &&
    state.altitude < FAST_DESCENT_ALTITUDE_THRESHOLD &&
    Math.abs(state.acceleration) < FAST_DESCENT_SPEED_STABILITY &&
    Math.abs(state.velocity) < 18;

  return stableParachuteDescent ? FAST_DESCENT_STEPS_PER_FRAME : NORMAL_STEPS_PER_FRAME;
}

function gameLoop() {
  const steps = currentStepsPerFrame();
  for (let i = 0; i < steps; i++) {
    stepSimulation();
    if (!running || paused || (state && state.landed)) break;
  }
  requestAnimationFrame(gameLoop);
}

Object.values(controls).forEach(input => {
  input.addEventListener("input", () => {
    updateControlLabels();
    if (!running) resetState();
  });
});

ui.launchButton.addEventListener("click", launchOrPause);
ui.resetButton.addEventListener("click", () => {
  clearAiTimer();
  resetState();
  addTerminalLine(">>> RESET");
  if (ai.enabled.checked) scheduleAiNextLaunch();
});

document.addEventListener("keydown", e => {
  if (e.code === "Space") {
    e.preventDefault();
    launchOrPause();
  } else if (e.key.toLowerCase() === "r") {
    clearAiTimer();
    resetState();
    addTerminalLine(">>> RESET");
    if (ai.enabled.checked) scheduleAiNextLaunch();
  } else if (e.key.toLowerCase() === "n") {
    nextMission();
  }
});

ai.aggression.addEventListener("input", () => {
  ai.aggressionValue.textContent = Number(ai.aggression.value).toFixed(2);
});

ai.delay.addEventListener("input", () => {
  ai.delayValue.textContent = `${Number(ai.delay.value).toFixed(1)} s`;
});

ai.enabled.addEventListener("change", () => {
  if (ai.enabled.checked) {
    ai.status.textContent = "AI enabled. It will tune and launch automatically when safe.";
    addTerminalLine(">>> AI ENABLED");
    if (!running && !(state && state.launched && !state.landed)) {
      scheduleAiNextLaunch();
    } else {
      ai.status.textContent = "AI enabled. Current launch will finish before automation resumes.";
    }
  } else {
    stopAI();
  }
});

ai.goal.addEventListener("change", () => {
  ai.bestByGoal = null;
  ai.goalChangePending = true;
  clearAiTimer();

  const goalName = ai.goal.options[ai.goal.selectedIndex].text;

  if (running || (state && state.launched && !state.landed)) {
    ai.status.textContent = `AI goal changed to "${goalName}". Change will apply after this launch lands.`;
    addTerminalLine(`>>> AI goal queued after landing: ${goalName}`);
    return;
  }

  ai.status.textContent = `AI goal changed to "${goalName}".`;

  if (ai.enabled.checked) {
    scheduleAiNextLaunch();
  }
});

ai.tuneButton.addEventListener("click", () => {
  if (running || (state && state.launched && !state.landed)) {
    ai.status.textContent = "AI Tune Once is disabled during flight. Wait for landing or reset.";
    addTerminalLine(">>> AI tune skipped: rocket is in flight");
    return;
  }
  tuneParametersWithAI();
});

ai.stopButton.addEventListener("click", stopAI);

defaultControls();
updateControlLabels();
loadMission();
resetState();
gameLoop();
