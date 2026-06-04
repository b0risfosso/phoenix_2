% water_particle_flow_simulation.m
% -------------------------------------------------------------------------
% Simulation 3: 2D Water Particle Flow Around Obstacles
%
% Run in MATLAB:
%     water_particle_flow_simulation
%
% This simulation models water as many simplified particles moving through a
% 2D box. Gravity pulls particles down, random spreading mimics turbulence,
% walls and circular obstacles deflect particles, and bottom collection tracks
% pooling/retention.
%
% This is not full SPH fluid dynamics. It is a lightweight particle-based
% teaching model of flow, splash, obstacle deflection, and retention.
% -------------------------------------------------------------------------

clear; clc; close all;

fprintf("2D Water Particle Flow Around Obstacles Simulation\n");
fprintf("Model: simplified particles with gravity, damping, walls, and obstacles\n\n");

rng(31);

% Domain
W = 10.0;
H = 6.0;

% Particle settings
num_particles = 260;
x = 1.0 + 0.35*randn(num_particles,1);
y = H - 0.4 + 0.20*randn(num_particles,1);
vx = 0.65 + 0.20*randn(num_particles,1);
vy = -0.10 + 0.10*randn(num_particles,1);

active = true(num_particles,1);
settled = false(num_particles,1);

% Physics
g = -9.81;
dt = 0.015;
duration = 12.0;
steps = floor(duration/dt);
drag = 0.985;
wall_bounce = 0.45;
floor_bounce = 0.22;
turbulence = 0.08;

% Obstacles: [cx, cy, radius]
obstacles = [
    4.0, 4.2, 0.55;
    6.2, 3.0, 0.70;
    3.2, 2.0, 0.45;
    7.8, 1.7, 0.50
];

% Diagnostics
fprintf("%8s | %8s | %10s | %10s | %10s | %10s\n", ...
    "time(s)", "active", "avg_y", "avg_speed", "settled", "escaped");
fprintf("%s\n", repmat("-",1,72));

figure("Name","2D Water Particle Flow Around Obstacles");

for k = 1:steps+1
    t = (k-1)*dt;

    % Refill source gradually for inactive escaped particles early in run.
    if t < 7
        refill_count = min(4, sum(~active));
        if refill_count > 0
            idx = find(~active, refill_count);
            x(idx) = 1.0 + 0.25*randn(refill_count,1);
            y(idx) = H - 0.25 + 0.10*randn(refill_count,1);
            vx(idx) = 0.85 + 0.20*randn(refill_count,1);
            vy(idx) = -0.15 + 0.10*randn(refill_count,1);
            active(idx) = true;
            settled(idx) = false;
        end
    end

    idx = find(active);
    n_active = numel(idx);

    % Gravity, turbulence, and drag.
    vy(idx) = vy(idx) + g*dt;
    vx(idx) = vx(idx) + turbulence*randn(n_active,1)*sqrt(dt);
    vy(idx) = vy(idx) + turbulence*randn(n_active,1)*sqrt(dt);
    vx(idx) = vx(idx)*drag;
    vy(idx) = vy(idx)*drag;

    % Update positions.
    x(idx) = x(idx) + vx(idx)*dt;
    y(idx) = y(idx) + vy(idx)*dt;

    % Wall collisions.
    left = active & x < 0;
    x(left) = 0;
    vx(left) = abs(vx(left))*wall_bounce;

    right = active & x > W;
    % Particles escaping on the right are counted inactive.
    active(right) = false;

    top = active & y > H;
    y(top) = H;
    vy(top) = -abs(vy(top))*wall_bounce;

    bottom = active & y < 0;
    y(bottom) = 0;
    vy(bottom) = abs(vy(bottom))*floor_bounce;
    vx(bottom) = vx(bottom)*0.75;

    % Settling: slow particles near floor become pooled.
    slow_near_floor = active & y < 0.08 & hypot(vx,vy) < 0.28;
    settled(slow_near_floor) = true;
    vx(slow_near_floor) = 0;
    vy(slow_near_floor) = 0;

    % Obstacle collisions.
    for j = 1:size(obstacles,1)
        cx = obstacles(j,1);
        cy = obstacles(j,2);
        r = obstacles(j,3);

        dx = x - cx;
        dy = y - cy;
        dist = hypot(dx,dy);
        hit = active & dist < r & dist > 0;

        if any(hit)
            nx = dx(hit)./dist(hit);
            ny = dy(hit)./dist(hit);

            % Move particles to obstacle surface.
            x(hit) = cx + nx*r;
            y(hit) = cy + ny*r;

            % Reflect velocity across obstacle normal.
            vn = vx(hit).*nx + vy(hit).*ny;
            vx(hit) = (vx(hit) - 1.65*vn.*nx)*0.72;
            vy(hit) = (vy(hit) - 1.65*vn.*ny)*0.72;
        end
    end

    % Diagnostics.
    active_idx = find(active);
    if isempty(active_idx)
        avg_y = 0;
        avg_speed = 0;
    else
        avg_y = mean(y(active_idx));
        avg_speed = mean(hypot(vx(active_idx),vy(active_idx)));
    end

    escaped = sum(~active);
    settled_count = sum(settled);

    if mod(k-1, round(0.5/dt)) == 0 || k == steps+1
        fprintf("%8.2f | %8d | %10.3f | %10.3f | %10d | %10d\n", ...
            t, sum(active), avg_y, avg_speed, settled_count, escaped);
    end

    % Plot.
    if mod(k-1, round(0.05/dt)) == 0 || k == 1
        clf;
        hold on;

        % Draw obstacles.
        th = linspace(0,2*pi,80);
        for j = 1:size(obstacles,1)
            cx = obstacles(j,1);
            cy = obstacles(j,2);
            r = obstacles(j,3);
            fill(cx + r*cos(th), cy + r*sin(th), [0.55 0.55 0.55], "EdgeColor", "k");
        end

        scatter(x(active), y(active), 20, "filled");
        if any(settled)
            scatter(x(settled), y(settled), 25, "filled");
        end

        rectangle("Position",[0,0,W,H], "EdgeColor","k", "LineWidth",1.5);
        xlim([-0.2 W+0.5]);
        ylim([-0.2 H+0.3]);
        xlabel("x position");
        ylabel("y position");
        title(sprintf("Water Particle Flow, t = %.2f s | active=%d settled=%d escaped=%d", ...
            t, sum(active), settled_count, escaped));
        grid on;
        drawnow;
    end
end

fprintf("\nFinal summary:\n");
fprintf("  particles active in box: %d\n", sum(active));
fprintf("  particles settled/pooling: %d\n", sum(settled));
fprintf("  particles escaped right outlet: %d\n", sum(~active));
fprintf("  final average speed of active particles: %.3f\n", avg_speed);
