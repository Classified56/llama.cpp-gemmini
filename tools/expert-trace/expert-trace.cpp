#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-ext.h"
#include "llama.h"
#include "log.h"

#include <cerrno>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct expert_loads {
    int32_t layer = -1;
    int64_t top_k = 0;
    int64_t n_tokens = 0;
    std::vector<int64_t> totals;
    std::vector<int64_t> positions;
};

struct cpu_expert_measurement {
    char     tensor_name[GGML_MAX_NAME] = {};
    int64_t expert = -1;
    int64_t load = 0;
    uint64_t cycle_before = 0;
    uint64_t cycle_after = 0;
};

struct expert_trace_state {
    int32_t n_expert = 0;
    int32_t n_layer = 0;
    bool callback_error = false;
    std::vector<expert_loads> routes;
    std::vector<cpu_expert_measurement> cpu_measurements;
};

static bool parse_layer_suffix(const char * name, int32_t & layer, std::string * operation = nullptr) {
    if (name == nullptr) {
        return false;
    }

    const char * suffix = std::strrchr(name, '-');
    if (suffix == nullptr || suffix == name) {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(suffix + 1, &end, 10);
    if (errno != 0 || end == suffix + 1 || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
        return false;
    }

    layer = static_cast<int32_t>(parsed);
    if (operation != nullptr) {
        operation->assign(name, static_cast<size_t>(suffix - name));
    }
    return true;
}

static bool parse_topk_layer(const char * name, int32_t & layer) {
    static const char prefix[] = "ffn_moe_topk-";
    return name != nullptr &&
        std::strncmp(name, prefix, sizeof(prefix) - 1) == 0 &&
        parse_layer_suffix(name, layer);
}

static bool collect_expert_loads(
        int32_t layer,
        int32_t n_expert,
        int64_t top_k,
        int64_t n_tokens,
        const int32_t * ids,
        size_t id_stride,
        size_t token_stride,
        expert_loads & result) {
    if (n_expert <= 0 || top_k <= 0 || n_tokens <= 0) {
        return false;
    }

    result.layer = layer;
    result.top_k = top_k;
    result.n_tokens = n_tokens;
    result.totals.assign(n_expert, 0);
    result.positions.assign(static_cast<size_t>(n_expert) * static_cast<size_t>(top_k), 0);

    const auto * bytes = reinterpret_cast<const uint8_t *>(ids);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t position = 0; position < top_k; ++position) {
            int32_t expert;
            std::memcpy(&expert, bytes + token * token_stride + position * id_stride, sizeof(expert));
            if (expert < 0 || expert >= n_expert) {
                LOG_ERR("expert trace: invalid expert id %d at layer %d, token %lld, position %lld\n",
                        expert, layer, static_cast<long long>(token), static_cast<long long>(position));
                return false;
            }

            result.totals[expert]++;
            result.positions[static_cast<size_t>(expert) * static_cast<size_t>(top_k) + position]++;
        }
    }

    return true;
}

static bool expert_trace_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<expert_trace_state *>(user_data);
    int32_t layer = -1;
    const bool selected = parse_topk_layer(tensor->name, layer);

    if (ask) {
        return selected;
    }
    if (!selected) {
        return true;
    }

    if (tensor->type != GGML_TYPE_I32 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        LOG_ERR("expert trace: unsupported top-k tensor layout for %s: type=%s, shape=[%lld,%lld,%lld,%lld]\n",
                tensor->name, ggml_type_name(tensor->type),
                static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
                static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]));
        state.callback_error = true;
        return false;
    }

    std::vector<uint8_t> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());

    expert_loads loads;
    if (!collect_expert_loads(layer, state.n_expert, tensor->ne[0], tensor->ne[1],
                reinterpret_cast<const int32_t *>(data.data()), tensor->nb[0], tensor->nb[1], loads)) {
        state.callback_error = true;
        return false;
    }

    state.routes.push_back(std::move(loads));
    return true;
}

static void cpu_measurement_callback(
        const ggml_cpu_moe_measurement * measurement,
        void * user_data) {
    auto & state = *static_cast<expert_trace_state *>(user_data);
    cpu_expert_measurement copy;
    std::snprintf(copy.tensor_name, sizeof(copy.tensor_name), "%s", measurement->tensor_name);
    copy.expert = measurement->expert;
    copy.load = measurement->load;
    copy.cycle_before = measurement->cycle_before;
    copy.cycle_after = measurement->cycle_after;
    state.cpu_measurements.push_back(copy);
}

static void emit_routes(const expert_trace_state & state, std::vector<int64_t> & route_loads) {
    for (const expert_loads & loads : state.routes) {
        for (int32_t expert = 0; expert < state.n_expert; ++expert) {
            route_loads[static_cast<size_t>(loads.layer) * state.n_expert + expert] += loads.totals[expert];

            std::printf("expert_route phase=prefill layer=%d expert=%d load=%lld positions=",
                    loads.layer, expert, static_cast<long long>(loads.totals[expert]));
            for (int64_t position = 0; position < loads.top_k; ++position) {
                if (position != 0) {
                    std::putchar('|');
                }
                const int64_t count = loads.positions[
                        static_cast<size_t>(expert) * static_cast<size_t>(loads.top_k) + position];
                std::printf("%lld:%lld", static_cast<long long>(position), static_cast<long long>(count));
            }
            std::putchar('\n');
        }
    }
}

static bool emit_cpu_measurements(const expert_trace_state & state, const std::vector<int64_t> & route_loads) {
    const char * cycle_source = ggml_cpu_moe_cycle_counter_source();
    const size_t n_slots = static_cast<size_t>(state.n_layer) * state.n_expert;
    std::vector<uint64_t> total_cycles(n_slots, 0);
    std::vector<int32_t> matrix_ops(n_slots, 0);
    bool ok = true;

    for (const cpu_expert_measurement & measurement : state.cpu_measurements) {
        int32_t layer = -1;
        std::string operation;
        if (!parse_layer_suffix(measurement.tensor_name, layer, &operation) || layer >= state.n_layer ||
                measurement.expert < 0 || measurement.expert >= state.n_expert) {
            LOG_ERR("expert trace: cannot identify CPU measurement tensor '%s' expert %lld\n",
                    measurement.tensor_name, static_cast<long long>(measurement.expert));
            ok = false;
            continue;
        }

        const uint64_t cycle_delta = measurement.cycle_after - measurement.cycle_before;
        const size_t slot = static_cast<size_t>(layer) * state.n_expert + measurement.expert;
        total_cycles[slot] += cycle_delta;
        matrix_ops[slot]++;

        std::printf("expert_compute phase=prefill layer=%d expert=%lld operation=%s load=%lld cycle_source=%s "
                    "cycle_before=%llu cycle_after=%llu cycle_delta=%llu\n",
                layer,
                static_cast<long long>(measurement.expert),
                operation.c_str(),
                static_cast<long long>(measurement.load),
                cycle_source,
                static_cast<unsigned long long>(measurement.cycle_before),
                static_cast<unsigned long long>(measurement.cycle_after),
                static_cast<unsigned long long>(cycle_delta));
    }

    for (int32_t layer = 0; layer < state.n_layer; ++layer) {
        for (int32_t expert = 0; expert < state.n_expert; ++expert) {
            const size_t slot = static_cast<size_t>(layer) * state.n_expert + expert;
            std::printf("expert_compute_total phase=prefill layer=%d expert=%d load=%lld matrix_ops=%d "
                        "cycle_source=%s cycle_delta=%llu\n",
                    layer,
                    expert,
                    static_cast<long long>(route_loads[slot]),
                    matrix_ops[slot],
                    cycle_source,
                    static_cast<unsigned long long>(total_cycles[slot]));
        }
    }

    return ok;
}

static bool emit_trace(const expert_trace_state & state) {
    if (state.routes.empty()) {
        LOG_ERR("expert trace: no MoE routing tensors were observed\n");
        return false;
    }
    if (state.cpu_measurements.empty()) {
        LOG_ERR("expert trace: no CPU MUL_MAT_ID measurements were observed; "
                "the operation may have run on another backend or with more than one thread\n");
        return false;
    }

    std::vector<int64_t> route_loads(static_cast<size_t>(state.n_layer) * state.n_expert, 0);
    emit_routes(state, route_loads);
    const bool ok = emit_cpu_measurements(state, route_loads);
    std::fflush(stdout);
    return ok;
}

static bool run_decode(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("expert trace: no input tokens; provide a prompt with -p\n");
        return false;
    }
    if (tokens.size() > static_cast<size_t>(params.n_batch)) {
        LOG_ERR("expert trace: prompt has %zu tokens but n_batch is %u; increase --batch-size\n",
                tokens.size(), params.n_batch);
        return false;
    }

    LOG_INF("expert trace: decoding %zu input tokens with one CPU thread\n", tokens.size());
    return llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) == 0;
}

static int run_self_test() {
    static const int32_t ids[] = {
        0, 1,
        1, 2,
        2, 3,
        0, 2,
    };

    expert_loads loads;
    const bool ok = collect_expert_loads(3, 4, 2, 4, ids,
            sizeof(ids[0]), 2 * sizeof(ids[0]), loads);
    if (!ok || loads.totals != std::vector<int64_t>({2, 2, 3, 1})) {
        LOG_ERR("expert trace self-test: load aggregation failed\n");
        return 1;
    }

    std::printf("expert_trace_self_test status=PASS\n");
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        return run_self_test();
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.cpuparams.n_threads = 1;
    params.cpuparams_batch.n_threads = 1;
    params.warmup = false;

    expert_trace_state trace_state;
    params.cb_eval = expert_trace_callback;
    params.cb_eval_user_data = &trace_state;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("expert trace: failed to initialize model or context\n");
        return 1;
    }

    trace_state.n_expert = llama_model_n_expert(model);
    trace_state.n_layer = llama_model_n_layer(model);
    if (trace_state.n_expert <= 0) {
        LOG_ERR("expert trace: model is not an MoE model\n");
        return 1;
    }

    trace_state.routes.reserve(trace_state.n_layer);
    trace_state.cpu_measurements.reserve(
            static_cast<size_t>(trace_state.n_layer) * trace_state.n_expert * 3);

    std::printf("expert_trace_start phase=prefill experts=%d layers=%d threads=1 cycle_source=%s "
                "measurement=cpu_mul_mat_id\n",
            trace_state.n_expert, trace_state.n_layer, ggml_cpu_moe_cycle_counter_source());
    std::fflush(stdout);

    ggml_cpu_set_moe_measurement_callback(cpu_measurement_callback, &trace_state);
    const bool decode_ok = run_decode(ctx, params);
    ggml_cpu_set_moe_measurement_callback(nullptr, nullptr);

    const bool trace_ok = emit_trace(trace_state);
    if (!decode_ok || !trace_ok || trace_state.callback_error) {
        LOG_ERR("expert trace: decode or measurement failed\n");
        return 1;
    }

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
