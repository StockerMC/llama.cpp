#include "test-log.h"

#include "llama.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ostream>
#include <streambuf>
#include <vector>

const size_t LOG_DUMP_MAX_BYTES = 64 * 1024;

namespace {

// one worker thread runs one model, and the log callbacks fire on the thread that produced the
// message - every worker captures the log of its own model into its own file
thread_local struct common_log * t_log;

thread_local int t_thold = LOG_LEVEL_INFO;

// the harness writes here in table mode, where the table itself carries the verdicts
struct test_null_streambuf : std::streambuf {
    int overflow(int c) override {
        return traits_type::not_eof(c);
    }

    std::streamsize xsputn(const char *, std::streamsize n) override {
        return n;
    }
};

} // namespace

void test_log_init() {
    llama_log_set(test_log_callback, nullptr);
}

void test_log_capture(const std::string & path) {
    if (t_log) {
        common_log_free(t_log); // drains the queue, then closes the file
        t_log = nullptr;
    }

    if (path.empty()) {
        return;
    }

    t_log = common_log_init();

    common_log_set_file(t_log, path.c_str(), true /* file_only */);
    common_log_set_prefix(t_log, true);
    common_log_set_timestamps(t_log, true);

    // the file is where the details of a failing model are looked for, never log less than info
    t_thold = std::max(t_thold, LOG_LEVEL_INFO);
}

bool test_log_capturing() {
    return t_log != nullptr;
}

int test_log_threshold() {
    return t_thold;
}

void test_log_set_threshold(int verbosity) {
    t_thold = verbosity;
}

void test_log_add(enum ggml_log_level level, const char * fmt, ...) {
    char stack[1024];

    va_list args;
    va_start(args, fmt);
    const int n = vsnprintf(stack, sizeof(stack), fmt, args);
    va_end(args);

    if (n < 0) {
        return;
    }

    // the message can be a list of tokens, format it again when it did not fit the stack buffer
    std::vector<char> heap;

    if (n >= (int) sizeof(stack)) {
        heap.resize(n + 1);

        va_start(args, fmt);
        vsnprintf(heap.data(), heap.size(), fmt, args);
        va_end(args);
    }

    const char * text = heap.empty() ? stack : heap.data();

    if (test_log_capturing()) {
        common_log_add(t_log, level, "%s", text);
        return;
    }

    // without a capture the harness writes the verdicts to stderr as it goes, so the details
    // have to reach the same stream, in the same order
    fputs(text, stderr);
    fflush(stderr);
}

void test_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    if (test_log_capturing()) {
        // gate the capture the way the console would be - the debug chatter of the graph
        //   builder is orders of magnitude more than everything else
        if (common_log_get_verbosity(level) <= t_thold) {
            common_log_add(t_log, level, "%s", text);
        }

        return;
    }

    common_log_default_callback(level, text, user_data);
}

std::ostream & test_null_stream() {
    // per thread, the workers of the table print at the same time
    thread_local test_null_streambuf sb;
    thread_local std::ostream        null_out(&sb);

    return null_out;
}

bool test_log_dump(const std::string & path, size_t max_bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "failed to read the captured log '%s'\n", path.c_str());
        return false;
    }

    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    size_t skip = 0;
    if (text.size() > max_bytes) {
        skip = text.find('\n', text.size() - max_bytes);
        skip = skip == std::string::npos ? 0 : skip + 1;
    }

    if (skip > 0) {
        const size_t n_all   = std::count(text.begin(), text.end(), '\n');
        const size_t n_shown = std::count(text.begin() + skip, text.end(), '\n');

        fprintf(stdout, "... %zu of %zu lines omitted, full log: %s\n", n_all - n_shown, n_all, path.c_str());
    }

    fwrite(text.data() + skip, 1, text.size() - skip, stdout);
    fflush(stdout);

    return skip > 0;
}

std::string test_state_path(const std::string & model_name) {
    return "dump_state." + model_name + ".bin";
}

std::string test_log_path(const std::string & logs_dir, const std::string & model_name) {
    const std::string file = "test-llama-context." + model_name + ".log";

    return logs_dir.empty() ? file : (std::filesystem::path(logs_dir) / file).string();
}

int test_opt(char ** argv, int & i, int argc, const char * name, std::string & value) {
    const size_t len = std::strlen(name);

    if (std::strncmp(argv[i], name, len) != 0) {
        return 0;
    }

    if (argv[i][len] == '=') {
        value = argv[i] + len + 1;
        return value.empty() ? -1 : 1;
    }

    if (argv[i][len] != '\0') {
        return 0;
    }

    if (i + 1 >= argc) {
        return -1;
    }

    value = argv[++i];
    return value.empty() ? -1 : 1;
}
