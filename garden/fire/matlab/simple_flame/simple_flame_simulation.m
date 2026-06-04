% simple_flame_simulation.m
% -------------------------------------------------------------------------
% Simple Flame Scientific Simulation
%
% Run in MATLAB:
%     simple_flame_simulation
%
% This is not a visual flame animation. It is a scientific/math simulation
% of a small well-mixed flame cell. It tracks:
%   - fuel concentration
%   - oxygen concentration
%   - temperature
%   - reaction rate
%   - heat release
%   - convective/radiative/mixing heat losses
%   - smoke and soot formation
%   - burn efficiency
%   - extinction/end state
%
% The model is intentionally simplified. It is useful for experimenting with
% flame properties, not for high-accuracy combustion engineering.
% -------------------------------------------------------------------------

clear; clc;

fprintf("Simple Flame Scientific Simulation\n");
fprintf("Well-mixed toy combustion model: fuel + oxygen -> heat + products\n\n");

% -------------------------------------------------------------------------
% Global simulation settings
% -------------------------------------------------------------------------

rng(23);

rounds = 8;
duration_s = 6.0;
dt = 0.01;
print_every_s = 0.20;

% -------------------------------------------------------------------------
% Baseline flame parameters
% -------------------------------------------------------------------------

params.fuel_name = "baseline methane-like gas";

params.initial_fuel = 1.00;          % normalized fuel amount
params.initial_oxygen = 1.20;        % normalized oxygen amount
params.fuel_supply_rate = 0.10;      % normalized units / s
params.oxygen_supply_rate = 0.55;    % normalized units / s
params.max_fuel = 1.80;
params.max_oxygen = 1.80;

params.initial_temperature_K = 300.0;
params.spark_energy_J = 650.0;
params.ignition_temperature_K = 720.0;
params.extinction_temperature_K = 620.0;
params.reaction_strength = 2.4;
params.activation_temperature_K = 900.0;
params.oxygen_per_fuel = 2.0;

params.heat_of_combustion_J_per_unit = 520.0;
params.heat_capacity_J_per_K = 1.20;
params.ambient_temperature_K = 300.0;

params.convective_loss_coeff = 0.75;
params.radiative_loss_coeff = 2.6e-8;
params.mixing_loss_coeff = 0.12;

params.soot_rate = 0.020;
params.smoke_decay_rate = 0.18;
params.humidity = 0.20;
params.chamber_volume = 1.00;

summaries = struct([]);

% -------------------------------------------------------------------------
% Main round loop
% -------------------------------------------------------------------------

previous_summary = [];

for round_number = 1:rounds
    params = choose_next_flame_params(params, previous_summary, round_number);

    summary = run_flame_round(round_number, params, duration_s, dt, print_every_s);
    summaries = [summaries, summary]; %#ok<AGROW>
    previous_summary = summary;
end

print_final_summary(summaries);


% =========================================================================
% Local functions
% =========================================================================

function summary = run_flame_round(round_number, p, duration_s, dt, print_every_s)
    state.t_s = 0.0;
    state.temperature_K = p.initial_temperature_K;
    state.fuel = p.initial_fuel;
    state.oxygen = p.initial_oxygen;
    state.burned_gas = 0.0;
    state.smoke = 0.0;
    state.soot = 0.0;
    state.total_energy_released_J = 0.0;
    state.total_heat_lost_J = 0.0;
    state.total_fuel_burned = 0.0;
    state.state_name = "unignited";

    fprintf("\n%s\n", repmat("=", 1, 120));
    fprintf("ROUND %d: %s\n", round_number, p.fuel_name);
    fprintf("%s\n", repmat("=", 1, 120));
    fprintf("initial fuel=%.3f, initial O2=%.3f, fuel_supply=%.3f/s, O2_supply=%.3f/s\n", ...
        p.initial_fuel, p.initial_oxygen, p.fuel_supply_rate, p.oxygen_supply_rate);
    fprintf("spark=%.2f J, ignition=%.1f K, extinction=%.1f K, reaction_strength=%.2f\n", ...
        p.spark_energy_J, p.ignition_temperature_K, p.extinction_temperature_K, p.reaction_strength);
    fprintf("cooling: conv=%.3f, rad=%.2e, mix=%.3f, humidity=%.2f, volume=%.2f\n", ...
        p.convective_loss_coeff, p.radiative_loss_coeff, p.mixing_loss_coeff, p.humidity, p.chamber_volume);
    fprintf("%s\n", repmat("-", 1, 120));

    fprintf("%6s | %15s | %8s | %7s | %7s | %8s | %9s | %8s | %7s | %7s | %6s\n", ...
        "t(s)", "state", "T(K)", "fuel", "O2", "rate", "power(W)", "loss(W)", "smoke", "soot", "h(m)");
    fprintf("%s\n", repmat("-", 1, 120));

    peak_temperature = state.temperature_K;
    peak_rate = 0.0;
    burn_time = 0.0;
    efficiency_sum = 0.0;
    efficiency_samples = 0;
    oxygen_initial_plus_supply = p.initial_oxygen;
    next_print = 0.0;

    steps = floor(duration_s / dt);

    diagnostics.reaction_rate = 0.0;
    diagnostics.power_W = 0.0;
    diagnostics.total_loss_W = 0.0;
    diagnostics.efficiency = 0.0;

    for step = 0:steps
        is_first_step = (step == 0);
        [state, diagnostics] = step_flame(state, p, dt, is_first_step);

        peak_temperature = max(peak_temperature, state.temperature_K);
        peak_rate = max(peak_rate, diagnostics.reaction_rate);

        if diagnostics.reaction_rate > 0.02
            burn_time = burn_time + dt;
        end

        efficiency_sum = efficiency_sum + diagnostics.efficiency;
        efficiency_samples = efficiency_samples + 1;
        oxygen_initial_plus_supply = oxygen_initial_plus_supply + p.oxygen_supply_rate * dt;

        if state.t_s + 1e-9 >= next_print || step == steps
            h = flame_height_estimate_m(state, diagnostics, p);
            fprintf("%6.2f | %15s | %8.1f | %7.3f | %7.3f | %8.4f | %9.2f | %8.2f | %7.4f | %7.4f | %6.3f\n", ...
                state.t_s, state.state_name, state.temperature_K, state.fuel, state.oxygen, ...
                diagnostics.reaction_rate, diagnostics.power_W, diagnostics.total_loss_W, ...
                state.smoke, state.soot, h);
            next_print = next_print + print_every_s;
        end
    end

    oxygen_used = max(0.0, oxygen_initial_plus_supply - state.oxygen);
    avg_efficiency = efficiency_sum / max(1, efficiency_samples);
    extinction_cause = determine_extinction_cause(state, p, diagnostics);

    summary.round_number = round_number;
    summary.fuel_name = p.fuel_name;
    summary.peak_temperature_K = peak_temperature;
    summary.final_temperature_K = state.temperature_K;
    summary.burn_time_s = burn_time;
    summary.total_energy_released_J = state.total_energy_released_J;
    summary.total_heat_lost_J = state.total_heat_lost_J;
    summary.fuel_burned = state.total_fuel_burned;
    summary.oxygen_used = oxygen_used;
    summary.peak_reaction_rate = peak_rate;
    summary.average_efficiency = avg_efficiency;
    summary.final_smoke = state.smoke;
    summary.final_soot = state.soot;
    summary.extinction_cause = extinction_cause;

    fprintf("%s\n", repmat("-", 1, 120));
    fprintf("Round result:\n");
    fprintf("  peak temperature: %.1f K\n", summary.peak_temperature_K);
    fprintf("  final temperature: %.1f K\n", summary.final_temperature_K);
    fprintf("  burn time: %.2f s\n", summary.burn_time_s);
    fprintf("  total energy released: %.2f J\n", summary.total_energy_released_J);
    fprintf("  total heat lost: %.2f J\n", summary.total_heat_lost_J);
    fprintf("  fuel burned: %.4f normalized units\n", summary.fuel_burned);
    fprintf("  oxygen used: %.4f normalized units\n", summary.oxygen_used);
    fprintf("  peak reaction rate: %.4f units/s\n", summary.peak_reaction_rate);
    fprintf("  average heat efficiency: %.3f\n", summary.average_efficiency);
    fprintf("  final smoke: %.4f\n", summary.final_smoke);
    fprintf("  final soot: %.4f\n", summary.final_soot);
    fprintf("  extinction/end state: %s\n", summary.extinction_cause);
end


function [s, diagnostics] = step_flame(state, p, dt, is_first_step)
    s = state;

    if is_first_step
        s.temperature_K = s.temperature_K + p.spark_energy_J / max(0.001, p.heat_capacity_J_per_K);
    end

    % Supply reactants.
    s.fuel = clamp(s.fuel + p.fuel_supply_rate * dt, 0.0, p.max_fuel);
    s.oxygen = clamp(s.oxygen + p.oxygen_supply_rate * dt, 0.0, p.max_oxygen);

    % Reaction.
    rate = reaction_rate(s, p);
    requested_fuel_burn = rate * dt;
    possible_fuel_burn = min(s.fuel, s.oxygen / max(0.001, p.oxygen_per_fuel));
    fuel_burned = min(requested_fuel_burn, possible_fuel_burn);

    oxygen_used = fuel_burned * p.oxygen_per_fuel;
    heat_generated_J = fuel_burned * p.heat_of_combustion_J_per_unit;

    s.fuel = s.fuel - fuel_burned;
    s.oxygen = s.oxygen - oxygen_used;
    s.burned_gas = s.burned_gas + fuel_burned + oxygen_used;
    s.total_fuel_burned = s.total_fuel_burned + fuel_burned;
    s.total_energy_released_J = s.total_energy_released_J + heat_generated_J;

    % Smoke/soot formation.
    equivalence_indicator = s.fuel / max(0.001, s.oxygen);
    oxygen_starvation = clamp((equivalence_indicator - 0.45) * 0.6, 0.0, 2.0);
    cool_penalty = clamp((p.ignition_temperature_K + 350.0 - s.temperature_K) / 900.0, 0.0, 1.0);
    soot_created = fuel_burned * p.soot_rate * (1.0 + oxygen_starvation + cool_penalty + p.humidity);
    smoke_created = soot_created * 5.0 + fuel_burned * 0.02 * oxygen_starvation;

    s.soot = s.soot + soot_created;
    s.smoke = s.smoke + smoke_created;
    s.smoke = s.smoke * exp(-p.smoke_decay_rate * dt);

    losses = heat_losses_W(s, p);
    heat_lost_J = losses.total * dt;
    s.total_heat_lost_J = s.total_heat_lost_J + heat_lost_J;

    net_heat_J = heat_generated_J - heat_lost_J;
    s.temperature_K = s.temperature_K + net_heat_J / max(0.001, p.heat_capacity_J_per_K);

    if s.temperature_K < p.ambient_temperature_K
        s.temperature_K = p.ambient_temperature_K;
    end

    s.t_s = s.t_s + dt;
    s.state_name = classify_state(s, p, rate);

    diagnostics.reaction_rate = rate;
    diagnostics.fuel_burned = fuel_burned;
    diagnostics.oxygen_used = oxygen_used;
    diagnostics.heat_generated_J = heat_generated_J;
    diagnostics.heat_lost_J = heat_lost_J;
    diagnostics.convective_loss_W = losses.convective;
    diagnostics.radiative_loss_W = losses.radiative;
    diagnostics.mixing_loss_W = losses.mixing;
    diagnostics.total_loss_W = losses.total;
    diagnostics.power_W = heat_generated_J / max(0.001, dt);
    diagnostics.efficiency = heat_generated_J / max(0.001, heat_generated_J + heat_lost_J);
end


function r = reaction_rate(s, p)
    if s.temperature_K < p.extinction_temperature_K
        r = 0.0;
        return;
    end

    if s.fuel <= 0.0001 || s.oxygen <= 0.0001
        r = 0.0;
        return;
    end

    tf = temperature_factor(s.temperature_K, p);
    fuel_factor = s.fuel / max(0.001, p.chamber_volume);
    oxygen_factor = s.oxygen / max(0.001, p.chamber_volume);
    humidity_penalty = clamp(1.0 - 0.45 * p.humidity, 0.25, 1.0);

    r = p.reaction_strength * fuel_factor * oxygen_factor * tf * humidity_penalty;
end


function tf = temperature_factor(T, p)
    if T < p.ignition_temperature_K
        tf = 0.04 * T / max(1.0, p.ignition_temperature_K);
    else
        excess = T - p.ignition_temperature_K;
        tf = clamp(1.0 - exp(-excess / max(1.0, p.activation_temperature_K)), 0.0, 1.0);
    end
end


function losses = heat_losses_W(s, p)
    delta_T = max(0.0, s.temperature_K - p.ambient_temperature_K);

    losses.convective = p.convective_loss_coeff * delta_T;
    losses.radiative = p.radiative_loss_coeff * max(0.0, s.temperature_K^4 - p.ambient_temperature_K^4);
    losses.mixing = p.mixing_loss_coeff * delta_T * (p.oxygen_supply_rate + p.fuel_supply_rate);
    losses.total = losses.convective + losses.radiative + losses.mixing;
end


function name = classify_state(s, p, rate)
    if s.temperature_K < p.ignition_temperature_K && s.total_energy_released_J <= 0.0
        name = "pre-ignition";
    elseif s.fuel <= 0.005
        name = "fuel-limited";
    elseif s.oxygen <= 0.005
        name = "oxygen-limited";
    elseif s.temperature_K < p.extinction_temperature_K && s.total_energy_released_J > 0.0
        name = "extinguishing";
    elseif rate > 0.40
        name = "active burning";
    elseif rate > 0.05
        name = "weak burning";
    else
        name = "hot but slow";
    end
end


function h = flame_height_estimate_m(s, diagnostics, p)
    power = diagnostics.power_W;
    oxygen_factor = clamp(s.oxygen / max(0.001, p.max_oxygen), 0.0, 1.0);
    fuel_factor = clamp(s.fuel / max(0.001, p.max_fuel), 0.0, 1.0);

    h = clamp(0.04 + 0.010 * sqrt(max(0.0, power)) * ...
        (0.4 + oxygen_factor) * (0.3 + fuel_factor), 0.0, 1.5);
end


function cause = determine_extinction_cause(s, p, diagnostics)
    if s.fuel <= 0.01
        cause = "fuel exhausted";
    elseif s.oxygen <= 0.01
        cause = "oxygen starved";
    elseif s.temperature_K < p.extinction_temperature_K
        cause = "thermal extinction";
    elseif diagnostics.reaction_rate < 0.03
        cause = "weak reaction";
    elseif s.total_energy_released_J > 0 && s.temperature_K >= p.extinction_temperature_K
        cause = "still burning at end";
    else
        cause = "not ignited";
    end
end


function p = choose_next_flame_params(p, previous, round_number)
    goals = [
        "baseline balanced flame"
        "fuel rich flame"
        "oxygen poor flame"
        "strong airflow flame"
        "humid weak ignition"
        "high cooling test"
        "hot efficient flame"
        "small chamber flash"
    ];

    goal = goals(round_number);

    switch goal
        case "baseline balanced flame"
            p.fuel_name = "baseline methane-like gas";
            p.initial_fuel = 1.00;
            p.initial_oxygen = 1.20;
            p.fuel_supply_rate = 0.10;
            p.oxygen_supply_rate = 0.55;
            p.spark_energy_J = 650.0;
            p.reaction_strength = 2.4;
            p.convective_loss_coeff = 0.75;
            p.mixing_loss_coeff = 0.12;
            p.humidity = 0.20;
            p.chamber_volume = 1.00;
            p.soot_rate = 0.020;

        case "fuel rich flame"
            p.fuel_name = "fuel-rich yellow flame";
            p.initial_fuel = 1.35;
            p.initial_oxygen = 0.85;
            p.fuel_supply_rate = 0.18;
            p.oxygen_supply_rate = 0.33;
            p.spark_energy_J = 700.0;
            p.reaction_strength = 2.1;
            p.convective_loss_coeff = 0.68;
            p.mixing_loss_coeff = 0.10;
            p.humidity = 0.18;
            p.chamber_volume = 1.05;
            p.soot_rate = 0.040;

        case "oxygen poor flame"
            p.fuel_name = "oxygen-starved flame";
            p.initial_fuel = 1.05;
            p.initial_oxygen = 0.42;
            p.fuel_supply_rate = 0.12;
            p.oxygen_supply_rate = 0.14;
            p.spark_energy_J = 640.0;
            p.reaction_strength = 1.9;
            p.convective_loss_coeff = 0.72;
            p.mixing_loss_coeff = 0.07;
            p.humidity = 0.24;
            p.chamber_volume = 1.00;
            p.soot_rate = 0.055;

        case "strong airflow flame"
            p.fuel_name = "airflow-stretched flame";
            p.initial_fuel = 0.95;
            p.initial_oxygen = 1.35;
            p.fuel_supply_rate = 0.11;
            p.oxygen_supply_rate = 0.85;
            p.spark_energy_J = 680.0;
            p.reaction_strength = 2.8;
            p.convective_loss_coeff = 1.05;
            p.mixing_loss_coeff = 0.28;
            p.humidity = 0.16;
            p.chamber_volume = 1.10;
            p.soot_rate = 0.018;

        case "humid weak ignition"
            p.fuel_name = "humid weak ignition";
            p.initial_fuel = 0.90;
            p.initial_oxygen = 1.00;
            p.fuel_supply_rate = 0.08;
            p.oxygen_supply_rate = 0.45;
            p.spark_energy_J = 430.0;
            p.reaction_strength = 1.85;
            p.convective_loss_coeff = 0.82;
            p.mixing_loss_coeff = 0.15;
            p.humidity = 0.70;
            p.chamber_volume = 1.00;
            p.soot_rate = 0.026;

        case "high cooling test"
            p.fuel_name = "high cooling flame";
            p.initial_fuel = 1.10;
            p.initial_oxygen = 1.20;
            p.fuel_supply_rate = 0.10;
            p.oxygen_supply_rate = 0.55;
            p.spark_energy_J = 710.0;
            p.reaction_strength = 2.3;
            p.convective_loss_coeff = 1.45;
            p.mixing_loss_coeff = 0.35;
            p.humidity = 0.25;
            p.chamber_volume = 1.15;
            p.soot_rate = 0.022;

        case "hot efficient flame"
            p.fuel_name = "hot blue efficient flame";
            p.initial_fuel = 1.05;
            p.initial_oxygen = 1.60;
            p.fuel_supply_rate = 0.13;
            p.oxygen_supply_rate = 0.95;
            p.spark_energy_J = 760.0;
            p.reaction_strength = 3.2;
            p.convective_loss_coeff = 0.70;
            p.mixing_loss_coeff = 0.16;
            p.humidity = 0.08;
            p.chamber_volume = 1.00;
            p.soot_rate = 0.010;

        case "small chamber flash"
            p.fuel_name = "small chamber flash";
            p.initial_fuel = 1.25;
            p.initial_oxygen = 1.25;
            p.fuel_supply_rate = 0.02;
            p.oxygen_supply_rate = 0.03;
            p.spark_energy_J = 820.0;
            p.reaction_strength = 4.2;
            p.convective_loss_coeff = 0.95;
            p.mixing_loss_coeff = 0.05;
            p.humidity = 0.10;
            p.chamber_volume = 0.55;
            p.soot_rate = 0.028;
    end

    if ~isempty(previous)
        if previous.peak_temperature_K < p.ignition_temperature_K + 100
            p.spark_energy_J = p.spark_energy_J + 120.0;
            p.oxygen_supply_rate = p.oxygen_supply_rate + 0.08;
        end

        if previous.extinction_cause == "oxygen starved"
            p.oxygen_supply_rate = p.oxygen_supply_rate + 0.15;
            p.initial_oxygen = p.initial_oxygen + 0.20;
        end

        if previous.final_soot > 0.02
            p.oxygen_supply_rate = p.oxygen_supply_rate + 0.10;
            p.soot_rate = p.soot_rate * 0.85;
        end

        if previous.peak_temperature_K > 2600
            p.reaction_strength = p.reaction_strength * 0.9;
            p.convective_loss_coeff = p.convective_loss_coeff + 0.1;
        end
    end

    p.initial_fuel = clamp(p.initial_fuel, 0.05, 2.0);
    p.initial_oxygen = clamp(p.initial_oxygen, 0.05, 2.0);
    p.fuel_supply_rate = clamp(p.fuel_supply_rate, 0.0, 0.5);
    p.oxygen_supply_rate = clamp(p.oxygen_supply_rate, 0.0, 1.4);
    p.spark_energy_J = clamp(p.spark_energy_J, 0.0, 1200.0);
    p.reaction_strength = clamp(p.reaction_strength, 0.1, 6.0);
    p.convective_loss_coeff = clamp(p.convective_loss_coeff, 0.05, 3.0);
    p.mixing_loss_coeff = clamp(p.mixing_loss_coeff, 0.0, 0.8);
    p.humidity = clamp(p.humidity, 0.0, 1.0);
    p.chamber_volume = clamp(p.chamber_volume, 0.3, 3.0);
    p.soot_rate = clamp(p.soot_rate, 0.0, 0.12);

    fprintf("\nAI controller selected next experiment: %s\n", goal);
end


function print_final_summary(summaries)
    fprintf("\n%s\n", repmat("=", 1, 132));
    fprintf("FINAL SIMPLE FLAME COMBUSTION COMPARISON\n");
    fprintf("%s\n", repmat("=", 1, 132));

    fprintf("%5s | %24s | %8s | %8s | %7s | %9s | %9s | %7s | %8s | %8s | %6s | %7s | %s\n", ...
        "Round", "Fuel model", "Peak K", "Final K", "Burn s", "Energy J", "Lost J", ...
        "Fuel", "O2 used", "PeakRt", "Eff", "Soot", "End state");
    fprintf("%s\n", repmat("-", 1, 132));

    for i = 1:numel(summaries)
        s = summaries(i);
        name = char(s.fuel_name);
        if strlength(name) > 24
            name = extractBefore(string(name), 25);
        end

        fprintf("%5d | %24s | %8.1f | %8.1f | %7.2f | %9.2f | %9.2f | %7.4f | %8.4f | %8.4f | %6.3f | %7.4f | %s\n", ...
            s.round_number, name, s.peak_temperature_K, s.final_temperature_K, ...
            s.burn_time_s, s.total_energy_released_J, s.total_heat_lost_J, ...
            s.fuel_burned, s.oxygen_used, s.peak_reaction_rate, ...
            s.average_efficiency, s.final_soot, s.extinction_cause);
    end

    peak_values = [summaries.peak_temperature_K];
    energy_values = [summaries.total_energy_released_J];
    soot_values = [summaries.final_soot];
    burn_values = [summaries.burn_time_s];

    [~, hottest_idx] = max(peak_values);
    [~, energy_idx] = max(energy_values);
    [~, clean_idx] = min(soot_values);
    [~, long_idx] = max(burn_values);

    fprintf("\nBest scientific observations:\n");
    fprintf("  Hottest flame: Round %d, %s, peak %.1f K\n", ...
        summaries(hottest_idx).round_number, summaries(hottest_idx).fuel_name, summaries(hottest_idx).peak_temperature_K);
    fprintf("  Most energy released: Round %d, %.2f J\n", ...
        summaries(energy_idx).round_number, summaries(energy_idx).total_energy_released_J);
    fprintf("  Cleanest burn: Round %d, soot %.4f\n", ...
        summaries(clean_idx).round_number, summaries(clean_idx).final_soot);
    fprintf("  Longest active burn: Round %d, %.2f s\n", ...
        summaries(long_idx).round_number, summaries(long_idx).burn_time_s);
end


function y = clamp(x, low, high)
    y = max(low, min(high, x));
end
