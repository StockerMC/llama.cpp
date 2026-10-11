#pragma once

#include "test-util.h"

#include "../testing.h"

#include <memory>
#include <string>
#include <vector>

//
// the interface of a single context state case
//
// a case is a struct with a name, a short description, a group and a run() body. everything
// shared between cases belongs to the runner (test-runner.h), not to the case:
//
//   - which models the case applies to        -> skip_reason()
//   - whether it is also run against a dirty cache
//   - whether it needs the logits of the baseline case
//   - the log of the run, the table cell, the counters
//
// the body expresses only the scenario and the checks. it returns a verdict, it does not print
// the verdict:
//
//   return case_result::fail();                          a check did not hold
//   return case_result::skip("why");                     this model cannot exercise the case
//   return case_result::pass();                          everything held
//
// a FAIL is fatal for the run, a SKIP is not - see merge_status()
//

enum class test_status {
    PASS,
    FAIL,
    SKIP,
};

const char * test_status_str(test_status status);

// a skipped case does not fail the run, a failed case always does
test_status merge_status(test_status a, test_status b);

enum test_group {
    TEST_GROUP_STATE, // KV state save / load / copy
    TEST_GROUP_RS,    // recurrent state, also run against a dirty cache
};

struct case_result {
    test_status status = test_status::PASS;
    std::string reason; // why the case was skipped, empty otherwise

    static case_result pass();
    static case_result fail();
    static case_result skip(const std::string & why);
};

struct ctx_test {
    virtual ~ctx_test() = default;

    // the column header of the --models table, keep it short
    virtual const char * name() const = 0;

    // one line of what the case verifies, printed by --list
    virtual const char * description() const { return ""; }

    virtual test_group group() const { return TEST_GROUP_STATE; }

    // the case compares against the logits produced by the baseline case
    virtual bool needs_baseline() const { return false; }

    // the case is also run against a cache pre-filled with a known pattern, so that a restore
    // reading cells it never wrote cannot pass
    virtual bool uses_fill() const { return group() == TEST_GROUP_RS; }

    // non-empty means the case does not apply to this model, the text is the reported reason
    virtual std::string skip_reason(model_run & mr) const;

    // the scenario; `t` is the harness, use it for extra assertions or to refine a skip
    virtual case_result run(model_run & mr, testing & t) const = 0;
};

// record an expectation that did not hold and give up on the case - the harness counts the
// failure and prints it, the message ends up in the log of the current model
#define CTX_REQUIRE(t, cond, msg) \
    do { \
        if (!(t).assert_true((msg), (cond))) { \
            return case_result::fail(); \
        } \
    } while (0)

using ctx_test_ptr = std::unique_ptr<const ctx_test>;

// the cases of one source file, in table order
std::vector<ctx_test_ptr> make_kv_cases();  // cases-kv.cpp
std::vector<ctx_test_ptr> make_rs_cases();  // cases-rs.cpp

// all cases, in table order
const std::vector<ctx_test_ptr> & test_cases();
