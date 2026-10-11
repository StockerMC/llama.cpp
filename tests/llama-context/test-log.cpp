#include "test-log.h"

#include "llama.h"

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <mutex>

#if defined(_WIN32)
#  define TEST_LOG_ABORT_DUMP 0
#else
#  include <fcntl.h>
#  include <unistd.h>
#  define TEST_LOG_ABORT_DUMP 1
#endif

const size_t LOG_DUMP_MAX_BYTES = 64 * 1024;

namespace {

// the captures the workers have open right now - an abort dumps all of them, because the thread
//   that aborts is not necessarily the one that runs the model that matters
// the abort handler reads this while the other workers keep running, so it is a fixed array of
//   borrowed C strings and no mutex - the strings are owned by the loggers that publish them
constexpr size_t TEST_LOG_MAX_OPEN = 64;

std::atomic<const char *> g_open_logs[TEST_LOG_MAX_OPEN];

void test_log_open_add(const char * path) {
    for (auto & slot : g_open_logs) {
        const char * empty = nullptr;
        if (slot.compare_exchange_strong(empty, path)) {
            return;
        }
    }

    // more captures than slots - the run continues, only the abort dump misses this one
}

void test_log_open_del(const char * path) {
    for (auto & slot : g_open_logs) {
        const char * mine = path;
        slot.compare_exchange_strong(mine, nullptr);
    }
}

std::once_flag g_install_once;

} // namespace

static const char * test_log_prefix(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return "D";
        case GGML_LOG_LEVEL_INFO:  return "I";
        case GGML_LOG_LEVEL_WARN:  return "W";
        case GGML_LOG_LEVEL_ERROR: return "E";
        default:                   return "O";
    }
}

void test_logger::set_threshold(int thold_) {
    std::lock_guard<std::mutex> lock(mtx);

    thold = thold_;
}

void test_logger::open(const std::string & path_) {
    std::lock_guard<std::mutex> lock(mtx);

    capturing = true;

    if (!path.empty()) {
        test_log_open_del(path.c_str()); // before the string moves
    }
    path = path_;

    file.close();
    file.open(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        fprintf(stderr, "E failed to open the log file '%s'\n", path.c_str());
        return;
    }

    test_log_open_add(path.c_str()); // the logger owns the string, the pointer lives until close

    file << pending;
    pending.clear();
}

void test_logger::close() {
    std::lock_guard<std::mutex> lock(mtx);

    if (!path.empty()) {
        test_log_open_del(path.c_str());
    }

    file.close();
    path.clear();
}

void test_logger::flush() {
    std::lock_guard<std::mutex> lock(mtx);

    if (file.is_open()) {
        file.flush();
    }
}

void test_logger::write(enum ggml_log_level level, const char * fmt, va_list args) {
    if (common_log_get_verbosity(level) > thold) {
        return;
    }

    char buf[4096];
    vsnprintf(buf, sizeof(buf), fmt, args);

    // the messages are written with a leading newline for visual separation, in a per-line
    //   capture that only adds noise - unless the message is nothing but newlines
    char * text = buf;
    if (text[0] == '\n') {
        char * p = text;
        while (*p == '\n') { ++p; }
        if (*p != '\0') { text = p; }
    }

    std::string line;
    if (level != GGML_LOG_LEVEL_NONE) {
        line += test_log_prefix(level);
        line += ' ';
    }
    line += text;

    std::lock_guard<std::mutex> lock(mtx);

    if (file.is_open()) {
        file << line;
        if (level == GGML_LOG_LEVEL_ERROR) {
            file.flush();
        }
        return;
    }

    if (capturing) {
        pending += line;   // belongs to the model that is about to be run
        return;
    }

    fputs(line.c_str(), stderr);
    fflush(stderr);
}

void test_logger::dump_tail(const std::string & path, size_t max_bytes) {
#if !TEST_LOG_ABORT_DUMP
    (void) path;
    (void) max_bytes;
#else
    if (path.empty()) {
        return;
    }

    // note: the members named open/close/write shadow the POSIX calls
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        return;
    }

    const std::string banner = "\n=== aborted, tail of the captured log: " + path + " ===\n";
    if (::write(STDERR_FILENO, banner.c_str(), banner.size()) < 0) {
        // nothing else can be done
    }

    const off_t end   = ::lseek(fd, 0, SEEK_END);
    const off_t start = end > (off_t) max_bytes ? end - (off_t) max_bytes : 0;

    ::lseek(fd, start, SEEK_SET);

    char buf[4096];
    for (size_t left = (size_t) (end - start); left > 0; ) {
        const size_t n = left < sizeof(buf) ? left : sizeof(buf);

        const ssize_t r = ::read(fd, buf, n);
        if (r <= 0) {
            break;
        }
        if (::write(STDERR_FILENO, buf, r) < 0) {
            break;
        }
        left -= (size_t) r;
    }

    ::close(fd);
#endif // TEST_LOG_ABORT_DUMP
}

test_logger & test_log() {
    // one capture per thread: table mode runs a model per worker and the llama, ggml and backend
    //   callbacks fire on the thread that produced the message
    thread_local test_logger log;

    return log;
}

int test_log_threshold() {
    return test_log().thold;
}

void test_log_add(enum ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    test_log().write(level, fmt, args);
    va_end(args);
}

void test_log_callback(enum ggml_log_level level, const char * text, void * /* user_data */) {
    test_log_add(level, "%s", text);
}

static void test_log_abort(int sig) {
    // the other workers keep running while the dumps are written, the registry is built to be
    //   read without a lock
    for (const auto & slot : g_open_logs) {
        const char * path = slot.load();

        if (path != nullptr) {
            test_logger::dump_tail(path, LOG_DUMP_MAX_BYTES);
        }
    }

    signal(sig, SIG_DFL);
    raise(sig);
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

namespace {

// funnels an ostream into the logger, one logger line per '\n'
struct test_log_streambuf : std::streambuf {
    std::string line;

    ~test_log_streambuf() override {
        flush_line();
    }

    void flush_line() {
        if (line.empty()) {
            return;
        }

        test_log_add(GGML_LOG_LEVEL_NONE, "%s", line.c_str());
        line.clear();
    }

    int_type overflow(int_type c) override {
        if (c == traits_type::eof()) {
            flush_line();
            return c;
        }

        line.push_back(traits_type::to_char_type(c));
        if (c == '\n') {
            flush_line();
        }

        return c;
    }

    std::streamsize xsputn(const char * s, std::streamsize n) override {
        for (std::streamsize i = 0; i < n; ++i) {
            overflow(traits_type::to_int_type(s[i]));
        }

        return n;
    }

    int sync() override {
        flush_line();
        return 0;
    }
};

} // namespace

std::ostream & test_log_stream() {
    // per thread, like the capture it funnels into - the streambuf accumulates a partial line,
    //   two workers writing to one of them would interleave their verdicts
    thread_local test_log_streambuf sb;
    thread_local std::ostream       out(&sb);

    out.flush();

    return out;
}

void test_log_capture_start(int verbosity) {
    test_logger & log = test_log();

    // capture everything but the debug chatter of the graph builder - it is orders of
    //   magnitude more than everything else and buries the messages of the failing case
    // note: -lv raises this, use -lv 5 to capture the graph debug as well
    log.set_threshold(std::max(verbosity, LOG_LEVEL_INFO));
    log.capturing = true;

    // the callback and the handlers are process wide, the captures are per thread
    std::call_once(g_install_once, []() {
        llama_log_set(test_log_callback, nullptr);

        // a model that aborts would take the run, and with it the log of the very model that
        //   aborted, down with it - print what was captured so far before going away
#if TEST_LOG_ABORT_DUMP
        signal(SIGABRT, test_log_abort);
        signal(SIGSEGV, test_log_abort);
        signal(SIGFPE,  test_log_abort);
        signal(SIGILL,  test_log_abort);
#  ifdef SIGBUS
        signal(SIGBUS, test_log_abort);
#  endif
#endif // TEST_LOG_ABORT_DUMP
    });
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
