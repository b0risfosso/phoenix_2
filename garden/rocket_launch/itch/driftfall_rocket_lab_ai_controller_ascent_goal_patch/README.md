# Driftfall Rocket Lab

A small browser game for itch.io based on a terminal rocket launch simulation.

## Tagline

Tune. Launch. Read the telemetry. Survive the landing.

## How to play locally

Open `index.html` in a browser.

No build step is required.

## How to upload to itch.io

1. Create a new itch.io project.
2. Set the project kind to **HTML**.
3. Upload `driftfall_rocket_lab.zip`.
4. Enable **This file will be played in the browser**.
5. Recommended viewport: 1280 × 720 or larger.

## Controls

- Space: launch / pause / resume
- R: reset
- N: next mission after passing a mission
- Mouse: tune sliders

## Gameplay

Adjust the rocket parameters, launch, read telemetry, and complete mission goals.

You can tune:

- thrust
- fuel
- dry mass
- burn rate
- drag coefficient
- body area
- parachute area
- parachute deploy altitude
- wind

The simulation models:

- thrust
- gravity
- changing mass from fuel burn
- drag
- wind drift
- burnout
- coasting upward
- apogee
- ballistic descent
- parachute descent
- landing result

## Files

- `index.html`
- `style.css`
- `game.js`
- `README.md`


## Optional AI Controller

This version includes an optional AI controller.

The AI can:

- auto-tune rocket parameters
- auto-launch after a configurable delay
- optimize for the current mission
- optimize for altitude, soft landing, low drift, fuel efficiency, speed, or balanced recovery
- react to previous flight summaries
- advance through missions automatically when using the current mission goal

Controls:

- Check **Enable auto-tune + auto-launch** to let the AI run launches.
- Use **AI Tune Once** to let the AI adjust sliders without launching.
- Use **STOP AI** to disable automation.


## Performance update

This build includes a descent performance patch:

- terminal output is capped to the most recent telemetry lines
- graph history is shorter
- graph drawing is throttled
- long stable parachute descents advance faster
- repeated DOM text appends were replaced with a capped telemetry buffer

These changes reduce freezing or lag during descent, especially when the rocket spends a long time under parachute.


## AI ascent freeze patch

This build fixes a freeze that could occur when selecting a new AI goal during ascent.

Changes:

- AI goal changes during flight are queued until landing
- old auto-launch timers are cleared before new timers are scheduled
- AI Tune Once is disabled while the rocket is in flight
- manual launch/reset clears pending AI timers
- auto-launch cannot start while a rocket is already ascending, descending, paused, or waiting to land
