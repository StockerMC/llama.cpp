//
// test-llama-context - exercises the state handling of llama_context: KV and recurrent cache
// save, load, copy and rollback
//
// the test runs the same set of cases against every model it is given and prints a matrix -
// one row per model, one column per case - so that an architecture-specific regression is
// visible at a glance. the harness of the suite (tests/testing.h) drives the cases, so the
// verdicts, the counters and the exit code follow the same convention as the other tests.
//
// layout:
//   tests/llama-context/test-case.h     the interface a case implements, and the registry
//   tests/llama-context/test-runner.*   how cases are selected, executed and reported
//   tests/llama-context/test-log.*      the per-model log capture and its dump on failure
//   tests/llama-context/test-util.*     shared decode / state / comparison helpers
//   tests/llama-context/cases-kv.cpp    the KV state cases
//   tests/llama-context/cases-rs.cpp    the recurrent state cases
//
// two modes:
//   -m model.gguf   run the cases against a single model, one line per case, then a summary
//   --models DIR    run them against every .gguf in DIR and print the result matrix
//
// the log of llama, ggml and the backends is captured per model, so that a failure of one model
// in --models mode still shows its details without flooding the matrix
//

#include "llama-context/test-case.h"
#include "llama-context/test-log.h"
#include "llama-context/test-runner.h"
#include "llama-context/test-util.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "testing.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <regex>
#include <string>
#include <vector>

// tokenize the prompt, or generate random tokens for models without a tokenizer
static llama_tokens prepare_tokens(llama_model * model, const common_params & params) {
    llama_tokens tokens;

    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        TLOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto   n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        TLOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    TLOG_INF("%s: the input prompt is %d tokens\n", __func__, (int) tokens.size());

    return tokens;
}

static std::vector<std::string> collect_models(const std::string & models_dir) {
    std::vector<std::string> models;

    for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            models.push_back(entry.path().string());
        }
    }
    std::sort(models.begin(), models.end());

    return models;
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n  %s --models tests/test-models --test 'rollback|replay'\n", argv[0]);
    LOG("\n  %s -m your_model.gguf -lv 5\n", argv[0]);
    LOG("\nspecial flags:\n");
    LOG("\n  --models DIR              run every .gguf in DIR and print a result table\n");
    LOG("\n  --test REGEX              run only the cases whose name matches REGEX\n");
    LOG("\n  --list                    list the registered cases and exit\n");
    LOG("\n  --logs DIR                keep the log of each model in DIR instead of discarding it\n");
    LOG("\n  --logs-on-fail always     after the table, print the log of every failed model (default)\n");
    LOG("\n  --logs-on-fail never      never print the logs of the failed models\n");
    LOG("\n");
}

static void print_cases() {
    printf("\ncases:\n");

    for (const auto & tc : test_cases()) {
        printf("  %-11s %-5s %s\n", tc->name(), tc->group() == TEST_GROUP_RS ? "[rs]" : "[kv]", tc->description());
    }

    printf("\nthe [rs] cases are also run against a cache pre-filled with a known pattern\n\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    // extract our own options before handing the rest to the common parser
    std::string models_dir;
    std::string logs_dir;
    std::string test_filter;
    bool          list_cases      = false;
    bool          dump_logs_on_fail = true;

    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string value;

        const int res_models = test_opt(argv, i, argc, "--models", value);
        if (res_models != 0) {
            if (res_models < 0) {
                fprintf(stderr, "--models requires a directory argument\n");
                return 1;
            }
            models_dir = value;
            continue;
        }

        const int res_logs = test_opt(argv, i, argc, "--logs", value);
        if (res_logs != 0) {
            if (res_logs < 0) {
                fprintf(stderr, "--logs requires a directory argument\n");
                return 1;
            }
            logs_dir = value;
            continue;
        }

        const int res_test = test_opt(argv, i, argc, "--test", value);
        if (res_test != 0) {
            if (res_test < 0) {
                fprintf(stderr, "--test requires a regular expression argument\n");
                return 1;
            }
            test_filter = value;
            continue;
        }

        if (test_opt(argv, i, argc, "--logs-on-fail", value) != 0) {
            if (value == "always") {
                dump_logs_on_fail = true;
            } else if (value == "never") {
                dump_logs_on_fail = false;
            } else {
                fprintf(stderr, "--logs-on-fail: expected 'always' or 'never', got '%s'\n", value.c_str());
                return 1;
            }
            continue;
        }

        if (std::strcmp(argv[i], "--list") == 0) {
            list_cases = true;
            continue;
        }

        filtered_argv.push_back(argv[i]);
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int) filtered_argv.size() - 1;

    if (list_cases) {
        print_cases();
        return 0;
    }

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    // -lv applies to the test's own logging too
    test_log().set_threshold(params.verbosity);

    const std::vector<const ctx_test *> cases = select_cases(test_filter);
    if (cases.empty()) {
        fprintf(stderr, "--test '%s' selected no cases, see --list\n", test_filter.c_str());
        return 1;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    // capture the logs of each model to a file, so that they can be dumped when a model fails:
    // the table stays clean, but the details of a failure stay close at hand
    if (!logs_dir.empty() && !std::filesystem::create_directories(logs_dir)) {
        fprintf(stderr, "failed to create the log directory '%s'\n", logs_dir.c_str());
        return 1;
    }

    if (!models_dir.empty() || !logs_dir.empty()) {
        test_log_capture_start(params.verbosity);
    }

    llama_backend_init();

    // the harness reports the cases through the test logger, so that its verdicts end up in the
    // per-model capture in --models mode and on stderr otherwise
    testing t(test_log_stream());

    // run the full suite for a single model
    if (models_dir.empty()) {
        model_run mr;
        mr.params          = params;
        mr.name            = std::filesystem::path(params.model.path).filename().string();
        mr.params.out_file = test_state_path(mr.name);

        if (!logs_dir.empty()) {
            test_log().open(test_log_path(logs_dir, mr.name));
        }

        auto llama_init = common_init_from_params(params, true);

        GGML_ASSERT(llama_init->context() == nullptr);
        if (llama_init->model() == nullptr) {
            TLOG_ERR("%s: failed to init model '%s'\n", __func__, params.model.path.c_str());
            test_log().close();
            return 1;
        }

        mr.model  = llama_init->model();
        mr.tokens = prepare_tokens(mr.model, mr.params);

        const auto runs = run_cases(mr, t, cases);
        test_log().close();

        std::remove(mr.params.out_file.c_str()); // the cases leave the session file behind

        bool failed = false;
        for (const auto & run : runs) {
            failed = failed || run.status == test_status::FAIL;
        }

        if (!failed) {
            LOG("All applicable tests passed.\n");
        }

        return t.summary();
    }

    // table mode: run the suite for every model in the directory
    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
        TLOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
        return 1;
    }

    const auto models = collect_models(models_dir);
    if (models.empty()) {
        TLOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
        return 1;
    }

    auto col_width = [](const char * name) { return (int) std::max(strlen(name), (size_t) 4); };

    size_t name_width = 5; // "Model"
    for (const auto & model_path : models) {
        name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
    }

    // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
    common_log_set_verbosity_thold(0);

    if (!test_filter.empty()) {
        LOG("filter: %s\n", test_filter.c_str());
        common_log_flush(common_log_main());
    }

    LOG("%-*s", (int) name_width, "Model");
    for (const auto & tc : cases) {
        LOG("  %-*s", col_width(tc->name()), tc->name());
    }
    LOG("\n");
    common_log_flush(common_log_main());

    std::vector<size_t> n_pass(cases.size(), 0);
    std::vector<size_t> n_skip(cases.size(), 0);
    std::vector<size_t> n_fail(cases.size(), 0);

    size_t n_model_fail = 0;

    for (const auto & model_path : models) {
        const auto name = std::filesystem::path(model_path).filename().string();

        LOG("%-*s", (int) name_width, name.c_str());
        common_log_flush(common_log_main());

        std::vector<case_run> runs;

        // capture this model from the moment it is loaded, so that even a load failure has a log
        const std::string log_path = test_log_path(logs_dir, name);
        test_log().open(log_path);

        {
            struct common_params model_params = params;
            model_params.model.path = model_path;

            // a model that cannot be loaded is a failure, not a skip
            auto llama_init = common_init_from_params(model_params, true);
            if (llama_init->model() == nullptr) {
                TLOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
                for (const auto & tc : cases) {
                    case_run run;
                    run.test   = tc;
                    run.name   = tc->name();
                    run.status = test_status::FAIL;
                    run.reason = "the model failed to load";
                    runs.push_back(std::move(run));
                }
            } else {
                GGML_ASSERT(llama_init->context() == nullptr);

                model_run mr;
                mr.params          = std::move(model_params);
                mr.name            = name;
                mr.model           = llama_init->model();
                // the cases share a working directory, so state files are named after the model
                mr.params.out_file = test_state_path(name);
                mr.tokens          = prepare_tokens(mr.model, mr.params);

                runs = run_cases(mr, t, cases);
            }
        }

        for (size_t i = 0; i < runs.size(); i++) {
            LOG("  %s%*s", test_status_str(runs[i].status), col_width(cases[i]->name()) - 4, "");

            switch (runs[i].status) {
                case test_status::PASS: n_pass[i]++; break;
                case test_status::FAIL: n_fail[i]++; break;
                case test_status::SKIP: n_skip[i]++; break;
            }
        }
        LOG("\n");
        common_log_flush(common_log_main());

        const bool failed = std::any_of(runs.begin(), runs.end(),
                [](const case_run & run) { return run.status == test_status::FAIL; });
        n_model_fail += failed;

        test_log().close();
        std::remove(test_state_path(name).c_str()); // the cases leave the session file behind

        // dump the log of a failed model once the table is printed - interleaving it with the
        //   rows would make the table unreadable; keep the file when the dump was truncated
        if (failed && dump_logs_on_fail) {
            LOG("\n=== logs: %s (%s) ===\n", name.c_str(), log_path.c_str());
            common_log_flush(common_log_main());

            // keep the file when the dump was truncated, or when a log directory was requested
            if (!test_log_dump(log_path, LOG_DUMP_MAX_BYTES) && logs_dir.empty()) {
                std::remove(log_path.c_str());
            }
        } else if (logs_dir.empty()) {
            std::remove(log_path.c_str());
        }
    }

    common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
    common_log_flush(common_log_main());

    for (size_t i = 0; i < cases.size(); i++) {
        if (n_fail[i] == 0 && n_skip[i] == 0) {
            continue;
        }
        LOG_INF("%s: %-13s %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, cases[i]->name(), n_pass[i], n_skip[i], n_fail[i], models.size());
    }
    LOG_INF("%s: models: %zu passed, %zu failed (of %zu)\n", __func__, models.size() - n_model_fail, n_model_fail, models.size());

    return n_model_fail == 0 ? 0 : 1;
}
