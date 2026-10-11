#pragma once

#include "log.h"

#include <cstdarg>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <ostream>
#include <streambuf>
#include <string>

// test logging
//
// table mode silences the common logger to keep the result table readable, which drops every
// message the cases log with the common macros - a failing case would then print nothing but
// FAIL. the cases log here instead:
//
//   no capture (single model)   to stderr, gated by -lv
//   capture (every --models)    to a per-model file, dumped to stdout for each model that
//                               failed once the table has been printed in full
//
// a model that aborts in the middle of a run dies before the dump, so the capture is flushed on
// every error and between cases and an abort handler prints the tail of the open file - the
// handler only uses open/lseek/read/write, the lock is deliberately not taken
//
// llama, ggml and the backends are captured by replacing the callback installed by common_init
// with llama_log_set (which forwards to ggml_log_set), so common/log is left untouched
//
// the testing harness (tests/testing.h) writes to a std::ostream - test_log_stream() returns a
// stream that funnels into this logger, so that its [PASS] / [SKIP] / ASSERTION FAILED lines
// end up in the same place as the TLOG_* output
//

// how much of a captured log to print when a model fails, and how much of it to print when the
// process aborts
extern const size_t LOG_DUMP_MAX_BYTES;

struct test_logger {
    std::mutex    mtx;
    std::ofstream file;
    std::string   path;      // the file an abort dumps the tail of
    std::string   pending;   // logged before the first model opened a file
    bool          capturing = false;
    int           thold = LOG_LEVEL_INFO;

    void set_threshold(int thold_);

    void open(const std::string & path_);
    void close();

    // between cases, so that an abort keeps a usable file behind
    void flush();

    // GGML_LOG_LEVEL_NONE writes the text as is, used for streams such as the generated tokens
    void write(enum ggml_log_level level, const char * fmt, va_list args);

    // print the tail of the capture, used when the process is going away
    void dump_tail(size_t max_bytes);
};

// the single logger of the run
test_logger & test_log();

int test_log_threshold();

void test_log_add(enum ggml_log_level level, const char * fmt, ...);

// captures the logs of llama, ggml and the backends
void test_log_callback(enum ggml_log_level level, const char * text, void * user_data);

// print a captured log to stdout, keeping its tail when it is too long
// returns true when the log was truncated and the file should be kept on disk
bool test_log_dump(const std::string & path, size_t max_bytes);

// a stream that funnels into the logger, to be handed to the testing harness
std::ostream & test_log_stream();

// the capture and the abort handler are only needed when the logs go to a file
void test_log_capture_start(int verbosity);
void test_log_capture_stop();

// the cases share a working directory, so the session file is named after the model
std::string test_state_path(const std::string & model_name);

std::string test_log_path(const std::string & logs_dir, const std::string & model_name);

// the options of this test come in both --opt VALUE and --opt=VALUE forms and are removed from
// argv before the common parser sees them
// returns 0 when the argument is not this option, -1 when it is but the value is missing
int test_opt(char ** argv, int & i, int argc, const char * name, std::string & value);

#define TLOG_TMPL(level, verbosity, ...) \
    do { \
        if ((verbosity) <= test_log_threshold()) { \
            test_log_add((level), __VA_ARGS__); \
        } \
    } while (0)

#define TLOGV(verbosity, ...) TLOG_TMPL(GGML_LOG_LEVEL_NONE,  verbosity,       __VA_ARGS__)
#define TLOG_DBG(...)         TLOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_LEVEL_DEBUG, __VA_ARGS__)
#define TLOG_TRC(...)         TLOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_TRACE, __VA_ARGS__)
#define TLOG_INF(...)         TLOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_INFO,  __VA_ARGS__)
#define TLOG_WRN(...)         TLOG_TMPL(GGML_LOG_LEVEL_WARN,  LOG_LEVEL_WARN,  __VA_ARGS__)
#define TLOG_ERR(...)         TLOG_TMPL(GGML_LOG_LEVEL_ERROR, LOG_LEVEL_ERROR, __VA_ARGS__)
