/*
 * neuron_network_simulation.c
 * ------------------------------------------------------------
 * Terminal C simulation of a small spiking neuron network.
 *
 * Build:
 *     gcc neuron_network_simulation.c -o neuron_network_simulation -lm
 *
 * Run:
 *     ./neuron_network_simulation
 *
 * Model:
 *     - 6 leaky integrate-and-fire neurons
 *     - 4 excitatory neurons and 2 inhibitory neurons
 *     - delayed synaptic events
 *     - adaptation current after each spike
 *     - deterministic input pulses and waves
 *     - round-based controller that tunes excitation/inhibition
 *
 * This is a simplified scientific/educational model, not a full biological
 * neural simulator.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define TOTAL_ROUNDS 8
#define NETWORK_SIZE 6
#define EXCITATORY_COUNT 4
#define INHIBITORY_COUNT (NETWORK_SIZE - EXCITATORY_COUNT)

#define ROUND_DURATION_MS 600.0
#define DT_MS 0.25
#define PRINT_EVERY_MS 25.0

#define DISPLAY_WIDTH 34
#define MAX_EVENTS 20000
#define MAX_SPIKE_TIMES 5000

typedef struct {
    double v_rest_mv;
    double v_reset_mv;
    double v_threshold_mv;
    double membrane_resistance_mohm;
    double membrane_tau_ms;
    double refractory_ms;

    double base_current_na;
    double pulse_current_na;
    double pulse_start_ms;
    double pulse_duration_ms;
    double pulse_interval_ms;

    double excitatory_weight_na;
    double inhibitory_weight_na;
    double exc_to_inh_scale;
    double inh_to_exc_scale;
    double synaptic_delay_ms;
    double delay_spread_ms;
    double synaptic_decay_ms;

    double adaptation_decay_ms;
    double spike_adaptation_add_na;

    double input_wave_strength_na;
    double input_wave_period_ms;
    double neuron_bias_step_na;
} NetworkParams;

typedef struct {
    double t_ms;
    double voltage_mv;
    double refractory_left_ms;
    double adaptation_na;
    double synaptic_current_na;
    int spikes;
    double last_spike_ms;
    double max_voltage_mv;
    double min_voltage_mv;
} NeuronState;

typedef struct {
    double delivery_ms;
    int target_index;
    double current_na;
    int source_index;
    int active;
} SynapticEvent;

typedef struct {
    int round;
    int total_spikes;
    int exc_spikes;
    int inh_spikes;
    double network_rate_hz;
    double exc_rate_hz;
    double inh_rate_hz;
    double balance_ratio;
    double mean_voltage_mv;
    double max_voltage_mv;
    double min_voltage_mv;
    int delivered_events;
    int pending_events_end;

    double base_current_na;
    double pulse_current_na;
    double excitatory_weight_na;
    double inhibitory_weight_na;
    double synaptic_delay_ms;
    double delay_spread_ms;
    double threshold_mv;
} RoundResult;

double clamp_double(double value, double lo, double hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

const char *neuron_type(int index) {
    return (index < EXCITATORY_COUNT) ? "E" : "I";
}

double triangular_wave(double t_ms, double period_ms) {
    if (period_ms <= 0.0) return 0.0;

    double phase = fmod(t_ms, period_ms) / period_ms;
    if (phase < 0.25) {
        return phase * 4.0;
    }
    if (phase < 0.75) {
        return 2.0 - phase * 4.0;
    }
    return phase * 4.0 - 4.0;
}

NetworkParams default_params(void) {
    NetworkParams p;

    p.v_rest_mv = -70.0;
    p.v_reset_mv = -75.0;
    p.v_threshold_mv = -54.0;
    p.membrane_resistance_mohm = 10.0;
    p.membrane_tau_ms = 20.0;
    p.refractory_ms = 4.0;

    p.base_current_na = 1.05;
    p.pulse_current_na = 0.70;
    p.pulse_start_ms = 70.0;
    p.pulse_duration_ms = 38.0;
    p.pulse_interval_ms = 90.0;

    p.excitatory_weight_na = 0.34;
    p.inhibitory_weight_na = 0.42;
    p.exc_to_inh_scale = 1.15;
    p.inh_to_exc_scale = 1.10;
    p.synaptic_delay_ms = 3.0;
    p.delay_spread_ms = 1.0;
    p.synaptic_decay_ms = 8.0;

    p.adaptation_decay_ms = 130.0;
    p.spike_adaptation_add_na = 0.12;

    p.input_wave_strength_na = 0.07;
    p.input_wave_period_ms = 64.0;
    p.neuron_bias_step_na = 0.045;

    return p;
}

double stimulus_current_na(double t_ms, int neuron_index, const NetworkParams *p) {
    double center = (NETWORK_SIZE - 1) / 2.0;
    double bias = (neuron_index - center) * p->neuron_bias_step_na;
    double current = p->base_current_na + bias;

    double local_t = t_ms - neuron_index * 8.0;
    if (local_t >= p->pulse_start_ms && p->pulse_interval_ms > 0.0) {
        double phase = fmod(local_t - p->pulse_start_ms, p->pulse_interval_ms);
        if (phase <= p->pulse_duration_ms) {
            current += p->pulse_current_na;
        }
    }

    double wave_period = p->input_wave_period_ms + neuron_index * 5.0;
    current += p->input_wave_strength_na * triangular_wave(t_ms + neuron_index * 11.0, wave_period);

    return current;
}

double connection_weight_na(int source_index, int target_index, const NetworkParams *p) {
    if (source_index == target_index) return 0.0;

    int source_is_exc = source_index < EXCITATORY_COUNT;
    int target_is_exc = target_index < EXCITATORY_COUNT;

    double weight;
    if (source_is_exc) {
        weight = p->excitatory_weight_na;
        if (!target_is_exc) {
            weight *= p->exc_to_inh_scale;
        }
    } else {
        weight = -p->inhibitory_weight_na;
        if (target_is_exc) {
            weight *= p->inh_to_exc_scale;
        }
    }

    int distance = abs(source_index - target_index);
    double distance_scale = 1.0 - 0.09 * distance;
    if (distance_scale < 0.55) distance_scale = 0.55;

    return weight * distance_scale;
}

double connection_delay_ms(int source_index, int target_index, const NetworkParams *p) {
    int distance = abs(source_index - target_index);
    return p->synaptic_delay_ms + p->delay_spread_ms * distance;
}

int schedule_event(
    SynapticEvent events[],
    int *event_count,
    double delivery_ms,
    int target_index,
    double current_na,
    int source_index
) {
    if (*event_count >= MAX_EVENTS) {
        return 0;
    }

    events[*event_count].delivery_ms = delivery_ms;
    events[*event_count].target_index = target_index;
    events[*event_count].current_na = current_na;
    events[*event_count].source_index = source_index;
    events[*event_count].active = 1;
    (*event_count)++;

    return 1;
}

int deliver_synaptic_events(
    NeuronState states[],
    SynapticEvent events[],
    int *event_count,
    double t_ms
) {
    int delivered = 0;
    int write_index = 0;

    for (int i = 0; i < *event_count; i++) {
        if (events[i].active && events[i].delivery_ms <= t_ms + 1e-9) {
            int target = events[i].target_index;
            if (target >= 0 && target < NETWORK_SIZE) {
                states[target].synaptic_current_na += events[i].current_na;
                delivered++;
            }
        } else {
            events[write_index] = events[i];
            write_index++;
        }
    }

    *event_count = write_index;
    return delivered;
}

int step_network(
    NeuronState states[],
    const NetworkParams *p,
    double dt_ms,
    SynapticEvent events[],
    int *event_count,
    int spiking_indices[],
    double external_currents[]
) {
    double t_ms = states[0].t_ms;
    deliver_synaptic_events(states, events, event_count, t_ms);

    int spiking_count = 0;

    for (int i = 0; i < NETWORK_SIZE; i++) {
        NeuronState *s = &states[i];

        double external_current = stimulus_current_na(s->t_ms, i, p);
        external_currents[i] = external_current;

        double effective_current = external_current + s->synaptic_current_na - s->adaptation_na;

        if (s->refractory_left_ms > 0.0) {
            s->voltage_mv = p->v_reset_mv;
            s->refractory_left_ms -= dt_ms;
            if (s->refractory_left_ms < 0.0) {
                s->refractory_left_ms = 0.0;
            }
        } else {
            double dv_dt = (
                -(s->voltage_mv - p->v_rest_mv)
                + p->membrane_resistance_mohm * effective_current
            ) / p->membrane_tau_ms;

            s->voltage_mv += dv_dt * dt_ms;

            if (s->voltage_mv >= p->v_threshold_mv) {
                s->spikes += 1;
                s->last_spike_ms = s->t_ms;
                s->voltage_mv = p->v_reset_mv;
                s->refractory_left_ms = p->refractory_ms;
                s->adaptation_na += p->spike_adaptation_add_na;
                spiking_indices[spiking_count++] = i;
            }
        }

        if (p->adaptation_decay_ms > 0.0) {
            s->adaptation_na *= exp(-dt_ms / p->adaptation_decay_ms);
        }

        if (p->synaptic_decay_ms > 0.0) {
            s->synaptic_current_na *= exp(-dt_ms / p->synaptic_decay_ms);
        }

        s->t_ms += dt_ms;

        if (s->voltage_mv > s->max_voltage_mv) s->max_voltage_mv = s->voltage_mv;
        if (s->voltage_mv < s->min_voltage_mv) s->min_voltage_mv = s->voltage_mv;
    }

    for (int i = 0; i < spiking_count; i++) {
        int source_index = spiking_indices[i];

        for (int target_index = 0; target_index < NETWORK_SIZE; target_index++) {
            double weight = connection_weight_na(source_index, target_index, p);
            if (weight == 0.0) continue;

            double delivery_ms = t_ms + connection_delay_ms(source_index, target_index, p);
            schedule_event(events, event_count, delivery_ms, target_index, weight, source_index);
        }
    }

    return spiking_count;
}

void voltage_bar(double voltage_mv, char buffer[], int width) {
    double v_min = -80.0;
    double v_max = -45.0;
    int x = (int)((voltage_mv - v_min) / (v_max - v_min) * (width - 1));

    if (x < 0) x = 0;
    if (x >= width) x = width - 1;

    for (int i = 0; i < width; i++) {
        buffer[i] = '-';
    }
    buffer[x] = '|';
    buffer[width] = '\0';
}

void print_network_snapshot(NeuronState states[], int spiking_indices[], int spiking_count) {
    char bar[DISPLAY_WIDTH + 1];

    for (int i = 0; i < NETWORK_SIZE; i++) {
        int spiked = 0;
        for (int j = 0; j < spiking_count; j++) {
            if (spiking_indices[j] == i) {
                spiked = 1;
                break;
            }
        }

        voltage_bar(states[i].voltage_mv, bar, DISPLAY_WIDTH);

        printf(
            "  %s%d%c:%6.1fmV[%s]",
            neuron_type(i),
            i,
            spiked ? '*' : ' ',
            states[i].voltage_mv,
            bar
        );

        if (i != NETWORK_SIZE - 1) {
            printf("\n");
        }
    }
    printf("\n");
}

RoundResult run_round(int round_index, const NetworkParams *p) {
    printf("\n============================================================================================\n");
    printf("ROUND %d\n", round_index);
    printf("============================================================================================\n");
    printf("Network:\n");
    printf("  neurons=%d (%d excitatory, %d inhibitory)\n", NETWORK_SIZE, EXCITATORY_COUNT, INHIBITORY_COUNT);
    printf("Parameters:\n");
    printf("  V_rest=%.1f mV, V_threshold=%.1f mV, V_reset=%.1f mV\n",
           p->v_rest_mv, p->v_threshold_mv, p->v_reset_mv);
    printf("  Rm=%.2f Mohm, tau=%.2f ms, refractory=%.2f ms\n",
           p->membrane_resistance_mohm, p->membrane_tau_ms, p->refractory_ms);
    printf("  external base=%.3f nA, pulse=%.3f nA\n",
           p->base_current_na, p->pulse_current_na);
    printf("  synapse E=%.3f nA, I=%.3f nA, delay=%.2f ms, spread=%.2f ms\n",
           p->excitatory_weight_na, p->inhibitory_weight_na,
           p->synaptic_delay_ms, p->delay_spread_ms);
    printf("  adaptation_add=%.3f nA, synaptic_decay=%.1f ms\n",
           p->spike_adaptation_add_na, p->synaptic_decay_ms);
    printf("--------------------------------------------------------------------------------------------\n");

    NeuronState states[NETWORK_SIZE];
    for (int i = 0; i < NETWORK_SIZE; i++) {
        double start_v = p->v_rest_mv - 0.4 * i;
        states[i].t_ms = 0.0;
        states[i].voltage_mv = start_v;
        states[i].refractory_left_ms = 0.0;
        states[i].adaptation_na = 0.0;
        states[i].synaptic_current_na = 0.0;
        states[i].spikes = 0;
        states[i].last_spike_ms = -999999.0;
        states[i].max_voltage_mv = start_v;
        states[i].min_voltage_mv = start_v;
    }

    SynapticEvent events[MAX_EVENTS];
    int event_count = 0;

    int spiking_indices[NETWORK_SIZE];
    double external_currents[NETWORK_SIZE];
    double voltage_sum[NETWORK_SIZE];

    for (int i = 0; i < NETWORK_SIZE; i++) {
        voltage_sum[i] = 0.0;
    }

    int samples = 0;
    int total_delivered_events = 0;
    double next_print = 0.0;

    int steps = (int)(ROUND_DURATION_MS / DT_MS);

    for (int step = 0; step <= steps; step++) {
        double t_before = states[0].t_ms;
        int delivered = deliver_synaptic_events(states, events, &event_count, t_before);
        total_delivered_events += delivered;

        int spiking_count = step_network(states, p, DT_MS, events, &event_count, spiking_indices, external_currents);

        samples++;
        for (int i = 0; i < NETWORK_SIZE; i++) {
            voltage_sum[i] += states[i].voltage_mv;
        }

        if (states[0].t_ms + 1e-9 >= next_print) {
            printf("t=%7.2f ms | spikes now=", states[0].t_ms);

            if (spiking_count == 0) {
                printf("%-10s", "none");
            } else {
                char spike_labels[80] = "";
                for (int i = 0; i < spiking_count; i++) {
                    char label[16];
                    snprintf(label, sizeof(label), "%s%d%s",
                             neuron_type(spiking_indices[i]),
                             spiking_indices[i],
                             (i == spiking_count - 1) ? "" : ",");
                    strncat(spike_labels, label, sizeof(spike_labels) - strlen(spike_labels) - 1);
                }
                printf("%-10s", spike_labels);
            }

            int exc_now = 0;
            for (int i = 0; i < spiking_count; i++) {
                if (spiking_indices[i] < EXCITATORY_COUNT) exc_now++;
            }
            int inh_now = spiking_count - exc_now;

            double avg_syn = 0.0;
            for (int i = 0; i < NETWORK_SIZE; i++) {
                avg_syn += states[i].synaptic_current_na;
            }
            avg_syn /= NETWORK_SIZE;

            printf(" | E_now=%d I_now=%d | pending_events=%3d | avg_syn=%7.3f nA\n",
                   exc_now, inh_now, event_count, avg_syn);

            print_network_snapshot(states, spiking_indices, spiking_count);
            next_print += PRINT_EVERY_MS;
        }
    }

    double duration_s = ROUND_DURATION_MS / 1000.0;
    int spike_counts[NETWORK_SIZE];
    int exc_spikes = 0;
    int inh_spikes = 0;

    for (int i = 0; i < NETWORK_SIZE; i++) {
        spike_counts[i] = states[i].spikes;
        if (i < EXCITATORY_COUNT) {
            exc_spikes += spike_counts[i];
        } else {
            inh_spikes += spike_counts[i];
        }
    }

    int total_spikes = exc_spikes + inh_spikes;
    double exc_rate_hz = exc_spikes / (double)EXCITATORY_COUNT / duration_s;
    double inh_rate_hz = inh_spikes / (double)INHIBITORY_COUNT / duration_s;
    double network_rate_hz = total_spikes / (double)NETWORK_SIZE / duration_s;
    double balance_ratio = exc_spikes / (double)((inh_spikes > 0) ? inh_spikes : 1);

    double mean_voltage = 0.0;
    double max_voltage = states[0].max_voltage_mv;
    double min_voltage = states[0].min_voltage_mv;

    for (int i = 0; i < NETWORK_SIZE; i++) {
        mean_voltage += voltage_sum[i];
        if (states[i].max_voltage_mv > max_voltage) max_voltage = states[i].max_voltage_mv;
        if (states[i].min_voltage_mv < min_voltage) min_voltage = states[i].min_voltage_mv;
    }
    mean_voltage /= (samples * NETWORK_SIZE);

    RoundResult result;
    result.round = round_index;
    result.total_spikes = total_spikes;
    result.exc_spikes = exc_spikes;
    result.inh_spikes = inh_spikes;
    result.network_rate_hz = network_rate_hz;
    result.exc_rate_hz = exc_rate_hz;
    result.inh_rate_hz = inh_rate_hz;
    result.balance_ratio = balance_ratio;
    result.mean_voltage_mv = mean_voltage;
    result.max_voltage_mv = max_voltage;
    result.min_voltage_mv = min_voltage;
    result.delivered_events = total_delivered_events;
    result.pending_events_end = event_count;

    result.base_current_na = p->base_current_na;
    result.pulse_current_na = p->pulse_current_na;
    result.excitatory_weight_na = p->excitatory_weight_na;
    result.inhibitory_weight_na = p->inhibitory_weight_na;
    result.synaptic_delay_ms = p->synaptic_delay_ms;
    result.delay_spread_ms = p->delay_spread_ms;
    result.threshold_mv = p->v_threshold_mv;

    printf("--------------------------------------------------------------------------------------------\n");
    printf("Round result:\n");
    printf("  total spikes: %d\n", total_spikes);
    printf("  excitatory spikes: %d, inhibitory spikes: %d, E/I spike ratio: %.2f\n",
           exc_spikes, inh_spikes, balance_ratio);
    printf("  mean network firing rate: %.2f Hz per neuron\n", network_rate_hz);
    printf("  E rate: %.2f Hz per E neuron, I rate: %.2f Hz per I neuron\n", exc_rate_hz, inh_rate_hz);
    printf("  average voltage: %.2f mV\n", mean_voltage);
    printf("  voltage range: %.2f to %.2f mV\n", min_voltage, max_voltage);
    printf("  synaptic events delivered: %d\n", total_delivered_events);
    printf("  spike counts by neuron:\n");
    for (int i = 0; i < NETWORK_SIZE; i++) {
        printf("    %s%d: %d\n", neuron_type(i), i, spike_counts[i]);
    }

    return result;
}

NetworkParams ai_controller(NetworkParams p, RoundResult result) {
    NetworkParams new_p = p;

    int total_spikes = result.total_spikes;
    double balance_ratio = result.balance_ratio;
    int target_total_spikes = 28;
    double target_balance_ratio = 2.0;

    char decisions[4][180];
    int decision_count = 0;

    if (total_spikes < target_total_spikes - 8) {
        snprintf(decisions[decision_count++], 180,
                 "Network was underactive; increase drive and strengthen excitatory connections.");
        new_p.base_current_na += 0.10;
        new_p.pulse_current_na += 0.08;
        new_p.excitatory_weight_na += 0.035;
        new_p.v_threshold_mv -= 0.4;
    } else if (total_spikes > target_total_spikes + 10) {
        snprintf(decisions[decision_count++], 180,
                 "Network was overactive; reduce drive and strengthen inhibitory control.");
        new_p.base_current_na -= 0.08;
        new_p.pulse_current_na -= 0.05;
        new_p.inhibitory_weight_na += 0.045;
        new_p.spike_adaptation_add_na += 0.012;
        new_p.v_threshold_mv += 0.35;
    } else {
        snprintf(decisions[decision_count++], 180,
                 "Network activity was near target; tune timing and recovery.");
        new_p.synaptic_delay_ms += 0.35;
        new_p.delay_spread_ms += 0.08;
        new_p.synaptic_decay_ms += 0.5;
        new_p.input_wave_strength_na += 0.006;
    }

    if (balance_ratio > target_balance_ratio + 0.75) {
        snprintf(decisions[decision_count++], 180,
                 "Excitation dominated inhibition; strengthen inhibitory output and E-to-I recruitment.");
        new_p.inhibitory_weight_na += 0.035;
        new_p.exc_to_inh_scale += 0.035;
        new_p.inh_to_exc_scale += 0.025;
    } else if (balance_ratio < target_balance_ratio - 0.75) {
        snprintf(decisions[decision_count++], 180,
                 "Inhibition dominated or matched excitation too strongly; strengthen excitatory spread.");
        new_p.excitatory_weight_na += 0.03;
        new_p.inhibitory_weight_na -= 0.025;
        new_p.inh_to_exc_scale -= 0.02;
    } else {
        snprintf(decisions[decision_count++], 180,
                 "E/I balance was acceptable; make a small delay exploration step.");
        if (result.network_rate_hz > 10.0) {
            new_p.synaptic_delay_ms += 0.2;
        } else {
            new_p.synaptic_delay_ms -= 0.1;
        }
    }

    new_p.base_current_na = clamp_double(new_p.base_current_na, 0.1, 3.0);
    new_p.pulse_current_na = clamp_double(new_p.pulse_current_na, 0.0, 3.0);
    new_p.v_threshold_mv = clamp_double(new_p.v_threshold_mv, -62.0, -45.0);
    new_p.membrane_tau_ms = clamp_double(new_p.membrane_tau_ms, 5.0, 45.0);
    new_p.refractory_ms = clamp_double(new_p.refractory_ms, 1.0, 12.0);
    new_p.excitatory_weight_na = clamp_double(new_p.excitatory_weight_na, 0.02, 1.4);
    new_p.inhibitory_weight_na = clamp_double(new_p.inhibitory_weight_na, 0.02, 1.6);
    new_p.exc_to_inh_scale = clamp_double(new_p.exc_to_inh_scale, 0.4, 2.5);
    new_p.inh_to_exc_scale = clamp_double(new_p.inh_to_exc_scale, 0.4, 2.5);
    new_p.synaptic_delay_ms = clamp_double(new_p.synaptic_delay_ms, 0.5, 12.0);
    new_p.delay_spread_ms = clamp_double(new_p.delay_spread_ms, 0.0, 4.0);
    new_p.synaptic_decay_ms = clamp_double(new_p.synaptic_decay_ms, 2.0, 30.0);
    new_p.spike_adaptation_add_na = clamp_double(new_p.spike_adaptation_add_na, 0.0, 0.7);
    new_p.adaptation_decay_ms = clamp_double(new_p.adaptation_decay_ms, 30.0, 300.0);
    new_p.input_wave_strength_na = clamp_double(new_p.input_wave_strength_na, 0.0, 0.25);

    printf("\nAI controller decision:\n");
    printf("  Target total spikes: %d\n", target_total_spikes);
    printf("  Observed total spikes: %d\n", total_spikes);
    printf("  Target E/I ratio: %.2f\n", target_balance_ratio);
    printf("  Observed E/I ratio: %.2f\n", balance_ratio);
    for (int i = 0; i < decision_count; i++) {
        printf("  %s\n", decisions[i]);
    }
    printf("  Next round network parameters selected.\n");

    return new_p;
}

void print_final_summary(RoundResult results[], int count) {
    printf("\n================================================================================================================\n");
    printf("FINAL NETWORK COMPARISON SUMMARY\n");
    printf("================================================================================================================\n");

    printf("%5s | %6s | %5s | %5s | %8s | %7s | %7s | %7s | %7s | %8s | %8s\n",
           "Round", "Total", "E", "I", "Hz/N", "E/I", "E_w", "I_w", "Delay", "Events", "Thresh");
    printf("----------------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < count; i++) {
        RoundResult r = results[i];

        printf("%5d | %6d | %5d | %5d | %8.2f | %7.2f | %7.3f | %7.3f | %7.2f | %8d | %8.2f\n",
               r.round,
               r.total_spikes,
               r.exc_spikes,
               r.inh_spikes,
               r.network_rate_hz,
               r.balance_ratio,
               r.excitatory_weight_na,
               r.inhibitory_weight_na,
               r.synaptic_delay_ms,
               r.delivered_events,
               r.threshold_mv);
    }

    int target_total_spikes = 28;
    double target_balance_ratio = 2.0;
    int best_index = 0;
    double best_score = fabs(results[0].total_spikes - target_total_spikes)
                      + 4.0 * fabs(results[0].balance_ratio - target_balance_ratio);

    for (int i = 1; i < count; i++) {
        double score = fabs(results[i].total_spikes - target_total_spikes)
                     + 4.0 * fabs(results[i].balance_ratio - target_balance_ratio);

        if (score < best_score) {
            best_score = score;
            best_index = i;
        }
    }

    RoundResult best = results[best_index];

    printf("\nBest network target match:\n");
    printf("  Round %d: %d total spikes, E/I ratio %.2f, mean rate %.2f Hz per neuron\n",
           best.round, best.total_spikes, best.balance_ratio, best.network_rate_hz);
}

int main(void) {
    printf("C Neuron AI Network Rounds Simulation\n");
    printf("Terminal-only simulation. No external libraries except libm.\n");
    printf("Model: small excitatory/inhibitory leaky integrate-and-fire network with delayed synapses.\n");

    NetworkParams params = default_params();
    RoundResult results[TOTAL_ROUNDS];

    for (int round_index = 1; round_index <= TOTAL_ROUNDS; round_index++) {
        RoundResult result = run_round(round_index, &params);
        results[round_index - 1] = result;

        if (round_index < TOTAL_ROUNDS) {
            params = ai_controller(params, result);
        }
    }

    print_final_summary(results, TOTAL_ROUNDS);

    return 0;
}
