% water_tank_drain_simulation.m
% -------------------------------------------------------------------------
% Simulation 1: Water Tank Drain / Fill Scientific Model
%
% Run in MATLAB:
%     water_tank_drain_simulation
%
% This simulation models the height of water in a vertical tank as water
% enters through an inlet and leaves through a small drain hole. It uses
% Torricelli-style outflow:
%
%     outflow = Cd * A_hole * sqrt(2*g*h)
%
% It tracks water height, volume, inflow, outflow, and overflow over time.
% -------------------------------------------------------------------------

clear; clc; close all;

fprintf("Water Tank Drain / Fill Simulation\n");
fprintf("Model: tank water height with inflow, drain outflow, and overflow\n\n");

% Physical constants and tank geometry
g = 9.81;                 % m/s^2
tank_radius = 0.35;       % m
tank_area = pi*tank_radius^2;
tank_height = 1.20;       % m

hole_radius = 0.012;      % m
hole_area = pi*hole_radius^2;
Cd = 0.62;                % discharge coefficient

% Simulation settings
dt = 0.02;
duration = 80.0;
steps = floor(duration / dt);

% Initial state
h = 0.20;                 % initial water height, m
total_overflow = 0.0;
total_in = 0.0;
total_out = 0.0;

% Storage arrays
t_arr = zeros(steps+1,1);
h_arr = zeros(steps+1,1);
volume_arr = zeros(steps+1,1);
inflow_arr = zeros(steps+1,1);
outflow_arr = zeros(steps+1,1);
overflow_arr = zeros(steps+1,1);

fprintf("%8s | %9s | %10s | %10s | %10s | %10s\n", ...
    "time(s)", "height(m)", "volume(L)", "in(L/s)", "out(L/s)", "overflow(L)");
fprintf("%s\n", repmat("-",1,72));

for k = 1:steps+1
    t = (k-1)*dt;

    % Inflow schedule: faucet pulses between weak and strong flow.
    if t < 20
        inflow = 0.0015;              % m^3/s
    elseif t < 45
        inflow = 0.0032;
    elseif t < 62
        inflow = 0.0005;
    else
        inflow = 0.0000;
    end

    % Outflow through bottom hole.
    outflow = Cd * hole_area * sqrt(2*g*max(h,0));

    % Height update from volume balance.
    dV = (inflow - outflow) * dt;
    h = h + dV / tank_area;

    overflow = 0.0;
    if h > tank_height
        overflow_volume = (h - tank_height) * tank_area;
        overflow = overflow_volume;
        h = tank_height;
        total_overflow = total_overflow + overflow_volume;
    end

    if h < 0
        h = 0;
    end

    total_in = total_in + inflow*dt;
    total_out = total_out + outflow*dt;

    % Store
    t_arr(k) = t;
    h_arr(k) = h;
    volume_arr(k) = h*tank_area;
    inflow_arr(k) = inflow;
    outflow_arr(k) = outflow;
    overflow_arr(k) = total_overflow;

    if mod(k-1, round(2/dt)) == 0 || k == steps+1
        fprintf("%8.2f | %9.3f | %10.2f | %10.2f | %10.2f | %10.2f\n", ...
            t, h, volume_arr(k)*1000, inflow*1000, outflow*1000, total_overflow*1000);
    end
end

fprintf("\nFinal summary:\n");
fprintf("  final height: %.3f m\n", h_arr(end));
fprintf("  final volume: %.2f L\n", volume_arr(end)*1000);
fprintf("  total inflow: %.2f L\n", total_in*1000);
fprintf("  total drained: %.2f L\n", total_out*1000);
fprintf("  total overflow: %.2f L\n", total_overflow*1000);

% Plots
figure("Name","Water Tank Drain / Fill Simulation");

subplot(3,1,1);
plot(t_arr, h_arr, "LineWidth", 1.5);
ylabel("Height (m)");
title("Tank Water Height");
grid on;

subplot(3,1,2);
plot(t_arr, inflow_arr*1000, "LineWidth", 1.5); hold on;
plot(t_arr, outflow_arr*1000, "LineWidth", 1.5);
ylabel("Flow (L/s)");
legend("Inflow","Outflow");
title("Inflow and Drain Outflow");
grid on;

subplot(3,1,3);
plot(t_arr, overflow_arr*1000, "LineWidth", 1.5);
xlabel("Time (s)");
ylabel("Overflow (L)");
title("Cumulative Overflow");
grid on;
