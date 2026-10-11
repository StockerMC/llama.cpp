#include "test-runner.h"

#include "test-log.h"

#include <regex>
#include <utility>

std::vector<const ctx_test *> select_cases(const std::string & filter) {
    std::vector<const ctx_test *> selected;

    if (filter.empty()) {
        selected.reserve(test_cases().size());
        for (const auto & tc : test_cases()) {
            selected.push_back(tc.get());
        }

        return selected;
    }

    const std::regex re(filter);

    for (const auto & tc : test_cases()) {
        if (std::regex_match(tc->name(), re)) {
            selected.push_back(tc.get());
        }
    }

    return selected;
}

std::vector<case_run> run_cases(model_run & mr, testing & t, const std::vector<const ctx_test *> & cases) {
    // the state cases keep the historical default of a unified KV cache when running a single
    // sequence, the recurrent cases use whatever was passed on the command line
    const bool unified_base  = mr.params.kv_unified;
    const bool unified_state = mr.params.n_parallel == 1 ? true : unified_base;

    std::vector<case_run> runs;
    runs.reserve(cases.size());

    for (const ctx_test * tc : cases) {
        mr.params.kv_unified = tc->group() == TEST_GROUP_STATE ? unified_state : unified_base;

        case_run run;
        run.test = tc;
        run.name = tc->name();

        t.test(tc->name(), [&](testing & tt) {
            const int failures_before = tt.failures;

            if (const std::string why = tc->skip_reason(mr); !why.empty()) {
                run.status = test_status::SKIP;
                run.reason = why;
            } else if (tc->needs_baseline() && !mr.have_baseline) {
                run.status = test_status::SKIP;
                run.reason = "no baseline";
            } else if (!tc->uses_fill()) {
                mr.fill = 0;

                const case_result res = tc->run(mr, tt);

                run.status = res.status;
                run.reason = res.reason;
            } else {
                // the recurrent cases are also run against a cache pre-filled with a known
                // pattern, so that a restore reading cells it never wrote cannot pass
                run.detail = "fill 0, 0x3e";

                for (const uint8_t fill : { 0, 0x3e }) {
                    TLOGV(LOG_LEVEL_INFO, "(fill 0x%02x)\n", fill);

                    mr.fill = fill;

                    const case_result res = tc->run(mr, tt);

                    run.status = merge_status(run.status, res.status);
                    if (res.status == test_status::SKIP && run.reason.empty()) {
                        run.reason = res.reason;
                    }
                    if (res.status == test_status::FAIL) {
                        break;
                    }
                }
            }

            // hand the verdict to the harness: a failure must be counted by it, a skip must
            // carry its reason - a body that asserted on its own is already counted
            if (run.status == test_status::FAIL && tt.failures == failures_before) {
                tt.assert_true(std::string(tc->name()) + " failed, see the log", false);
            } else if (run.status == test_status::SKIP) {
                tt.skip(run.reason);
            }
        });

        runs.push_back(std::move(run));

        // keep the captured log usable even if the next case crashes the process
        test_log().flush();
    }

    return runs;
}
