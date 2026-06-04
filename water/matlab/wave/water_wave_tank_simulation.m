% water_wave_tank_simulation.m
% -------------------------------------------------------------------------
% Simulation 2: One-Dimensional Water Wave Tank
%
% Run in MATLAB:
%     water_wave_tank_simulation
%
% This simulation models a shallow 1D water surface using a simplified wave
% equation with damping:
%
%     eta_tt = c^2 * eta_xx - damping * eta_t + forcing
%
% eta(x,t) is water surface displacement. The script creates a wave pulse,
% applies wall reflections, and records wave energy over time.
% -------------------------------------------------------------------------

clear; clc; close all;

fprintf("One-Dimensional Water Wave Tank Simulation\n");
fprintf("Model: damped shallow-water surface displacement\n\n");

% Tank and numerical settings
L = 10.0;              % m
N = 180;               % grid points
dx = L/(N-1);
x = linspace(0,L,N);

g = 9.81;
depth = 0.40;          % shallow water depth, m
c = sqrt(g*depth);     % wave speed approximation

dt = 0.008;
duration = 25.0;
steps = floor(duration/dt);
damping = 0.035;

% Stability check
courant = c*dt/dx;
fprintf("Wave speed c = %.3f m/s, Courant number = %.3f\n", c, courant);
if courant > 1
    warning("Courant number > 1; simulation may be unstable.");
end

% State variables
eta = zeros(1,N);      % surface displacement, m
v = zeros(1,N);        % time derivative of eta

% Initial disturbance: Gaussian bump
eta = 0.12 * exp(-((x-2.2).^2)/(2*0.22^2));

% Storage for diagnostics
sample_count = floor(steps/20) + 2;
time_log = zeros(sample_count,1);
max_eta_log = zeros(sample_count,1);
energy_log = zeros(sample_count,1);
sample_idx = 1;

fprintf("%8s | %12s | %12s | %12s\n", "time(s)", "max_eta(m)", "min_eta(m)", "energy");
fprintf("%s\n", repmat("-",1,56));

figure("Name","1D Water Wave Tank");

for k = 1:steps+1
    t = (k-1)*dt;

    % Second spatial derivative.
    eta_xx = zeros(1,N);
    eta_xx(2:N-1) = (eta(3:N) - 2*eta(2:N-1) + eta(1:N-2)) / dx^2;

    % Occasional paddle forcing on left side.
    forcing = zeros(1,N);
    if t > 6 && t < 9
        forcing(8:14) = 0.45 * sin(2*pi*1.3*t);
    end
    if t > 14 && t < 17
        forcing(8:14) = 0.25 * sin(2*pi*2.0*t);
    end

    % Update velocity and surface.
    a = c^2 * eta_xx - damping*v + forcing;
    v = v + a*dt;
    eta = eta + v*dt;

    % Reflective wall boundary conditions.
    eta(1) = eta(2);
    eta(end) = eta(end-1);
    v(1) = 0;
    v(end) = 0;

    % Diagnostics.
    slope = gradient(eta, dx);
    energy = sum(0.5*v.^2 + 0.5*c^2*slope.^2) * dx;

    if mod(k-1, round(0.5/dt)) == 0 || k == steps+1
        fprintf("%8.2f | %12.4f | %12.4f | %12.6f\n", ...
            t, max(eta), min(eta), energy);

        if sample_idx <= sample_count
            time_log(sample_idx) = t;
            max_eta_log(sample_idx) = max(abs(eta));
            energy_log(sample_idx) = energy;
            sample_idx = sample_idx + 1;
        end
    end

    % Plot every few frames.
    if mod(k-1, round(0.08/dt)) == 0 || k == 1
        subplot(2,1,1);
        plot(x, eta, "LineWidth", 1.5);
        ylim([-0.22, 0.22]);
        xlabel("Position x (m)");
        ylabel("Surface eta (m)");
        title(sprintf("Water Wave Tank, t = %.2f s", t));
        grid on;

        subplot(2,1,2);
        plot(time_log(1:max(1,sample_idx-1)), energy_log(1:max(1,sample_idx-1)), "LineWidth", 1.5);
        xlabel("Time (s)");
        ylabel("Wave energy");
        title("Wave Energy Over Time");
        grid on;

        drawnow;
    end
end

fprintf("\nFinal summary:\n");
fprintf("  final max surface displacement: %.4f m\n", max(abs(eta)));
fprintf("  final energy: %.6f\n", energy);
fprintf("  wave speed estimate: %.3f m/s\n", c);
