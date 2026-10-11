#pragma once

#include "test-case.h"

#include "../testing.h"

#include <string>
#include <vector>

//
// the runner: the single place that decides how a case is executed
//
//   - which cases run (the --test filter selects them, in table order)
//   - which models they run for (ctx_test::skip_reason)
//   - the KV cache mode a case inherits (the state cases keep the historical default of a
//     unified cache for a single sequence, the recurrent cases use the command line)
//   - whether it is repeated against a dirty cache (ctx_test::uses_fill)
//   - how the verdict reaches the harness (testing), so that the per-test verdict lines, the
//     counters and the exit code are the same as in every other test of the suite
//

struct case_run {
    const ctx_test * test   = nullptr;
    std::string      name;
    test_status      status = test_status::SKIP;
    std::string      reason; // why it was skipped
    std::string      detail; // the variants it was run with, e.g. the fill passes
};

// the cases selected by the filter, all of them when it is empty, in table order
std::vector<const ctx_test *> select_cases(const std::string & filter);

// run the selected cases for one model; the harness records one test per case, the returned
// verdicts are what the result table prints
std::vector<case_run> run_cases(model_run & mr, testing & t, const std::vector<const ctx_test *> & cases);
