#pragma once

#include "log.h"

#include <cstddef>
#include <ostream>
#include <string>

// test logging
//
// table mode runs one model per thread and prints a result table, so the log of a model goes to
// a file of its own, and the log of a model that failed is printed to stdout once the table has
// been printed in full. single model mode has a file only with --logs, its messages go to
// stderr as usual
//
// the log callbacks of llama, ggml and the backends are process wide, so the test installs one
// callback that hands every message to the capture of the calling thread - one worker, one
// model, one file. the capture is a common_log that writes only to its file
//
// the harness (tests/testing.h) writes to a std::ostream: stderr in single model mode, a
// discard stream in table mode - the table itself carries the verdicts there, and a case that
// did not pass logs why with the macros below, so the file of the model still explains the
// verdict
//

// how much of a model log to print when the model fails
extern const size_t LOG_DUMP_MAX_BYTES;

// replace the log callback of common_init with the one that routes a message to the capture of
// the thread that produced it - call once, before any model is loaded
void test_log_init();

// routes the log of the calling thread to the given file; an empty path ends the capture and
// returns the messages to stderr - the file holds everything the run logged, up to the
// verbosity of the process
void test_log_capture(const std::string & path);

// true while the log of this thread goes to a file
bool test_log_capturing();

// the verbosity the TLOG_* macros compare against
int test_log_threshold();

void test_log_set_threshold(int verbosity);

// write to the capture of this thread, or to stderr when there is none
void test_log_add(enum ggml_log_level level, const char * fmt, ...) LOG_ATTRIBUTE_FORMAT(2, 3);

// captures the logs of llama, ggml and the backends, install it with llama_log_set
void test_log_callback(enum ggml_log_level level, const char * text, void * user_data);

// in table mode the table itself carries the verdicts, the harness writes here - in single model
// mode the harness writes to stderr
std::ostream & test_null_stream();

// print a captured log to stdout, keeping its tail when it is too long
// returns true when the log was truncated and the file should be kept on disk
bool test_log_dump(const std::string & path, size_t max_bytes);

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
