#pragma once
#define NOMINMAX

// llm_ab.hpp -- A/B testing framework for LLM prompts and models.
// Welch t-test, Cohen d, custom scorers. libcurl required.
//
// USAGE: #define LLM_AB_IMPLEMENTATION in ONE .cpp file.

#include <string>
#include <vector>
#include <functional>
#include <cstddef>

namespace llm {

struct Variant {
    std::string name;
    std::string prompt;       // supports {input} placeholder
    std::string model;
    std::string api_key;
    double      temperature = 0.7;
};

struct ABSample {
    std::string input;
    std::string expected;     // optional ground truth
};

struct VariantResult {
    std::string name;
    std::vector<std::string> outputs;
    std::vector<double>      scores;
    double mean_score = 0.0;
    double std_dev    = 0.0;
    size_t n          = 0;
};

struct ABResult {
    VariantResult control;
    VariantResult treatment;
    double t_statistic = 0.0;
    double p_value     = 1.0;
    double cohens_d    = 0.0;
    bool   significant = false;
    double alpha       = 0.05;
    std::string winner;
    std::string summary;
};

struct ABConfig {
    std::string api_key;
    double alpha = 0.05;
    std::function<double(const std::string&, const std::string&, const std::string&)> scorer;
};

VariantResult run_variant(
    const Variant& variant,
    const std::vector<ABSample>& samples,
    const ABConfig& cfg = {});

ABResult run_ab_test(
    const Variant& control,
    const Variant& treatment,
    const std::vector<ABSample>& samples,
    const ABConfig& cfg = {});

} // namespace llm

// ---------------------------------------------------------------------------
#ifdef LLM_AB_IMPLEMENTATION

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include <curl/curl.h>

namespace llm {
namespace detail_ab {

static size_t wcb(char* p, size_t s, size_t n, void* ud) {
    static_cast<std::string*>(ud)->append(p, s * n); return s * n;
}

static std::string jesc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if      (c == '"')  { o += '\\'; o += '"'; }
        else if (c == '\\') { o += '\\'; o += '\\'; }
        else if (c == '\n') { o += '\\'; o += 'n'; }
        else if (c == '\r') { o += '\\'; o += 'r'; }
        else if (c == '\t') { o += '\\'; o += 't'; }
        else o += static_cast<char>(c);
    }
    return o;
}

static std::string llm_call(const std::string& prompt, const Variant& v) {
    CURL* c = curl_easy_init();
    if (!c) return {};
    curl_slist* hdrs = nullptr;
    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
    hdrs = curl_slist_append(hdrs, ("Authorization: Bearer " + v.api_key).c_str());
    char tbuf[32]; snprintf(tbuf, sizeof(tbuf), "%.2f", v.temperature);
    std::string body = "{\"model\":\"" + jesc(v.model) + "\","
        "\"temperature\":" + tbuf + ","
        "\"messages\":[{\"role\":\"user\",\"content\":\"" + jesc(prompt) + "\"}]}";
    std::string resp;
    curl_easy_setopt(c, CURLOPT_URL, "https://api.openai.com/v1/chat/completions");
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, wcb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, 30L);
    curl_easy_perform(c);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(c);
    auto p2 = resp.find("\"content\":\"");
    if (p2 == std::string::npos) return {};
    p2 += 11;
    std::string out;
    while (p2 < resp.size() && resp[p2] != '"') {
        char ch = resp[p2];
        if (ch == '\\' && p2 + 1 < resp.size()) {
            char e = resp[++p2];
            if (e == 'n') out += '\n'; else out += e;
        } else out += ch;
        ++p2;
    }
    return out;
}

static std::string replace_placeholder(const std::string& tmpl, const std::string& input) {
    std::string out = tmpl;
    size_t pos = 0;
    while ((pos = out.find("{input}", pos)) != std::string::npos)
        out.replace(pos, 7, input);
    return out;
}

static std::string trim_lower(std::string s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\n')) s.erase(s.begin());
    while (!s.empty() && (s.back()  == ' ' || s.back()  == '\n')) s.pop_back();
    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

static double default_score(const std::string& out, const std::string& expected, const std::string&) {
    if (expected.empty()) return 1.0;
    return (trim_lower(out) == trim_lower(expected)) ? 1.0 : 0.0;
}

static void compute_stats(VariantResult& r) {
    r.n = r.scores.size();
    if (r.n == 0) return;
    r.mean_score = std::accumulate(r.scores.begin(), r.scores.end(), 0.0) / (double)r.n;
    double var = 0.0;
    for (double sc : r.scores) { double d = sc - r.mean_score; var += d * d; }
    r.std_dev = (r.n > 1) ? std::sqrt(var / (double)(r.n - 1)) : 0.0;
}

static double approx_pvalue(double t) {
    double at = std::abs(t);
    double p = 2.0 / (1.0 + std::exp(0.717 * at + 0.416 * t * t));
    return (p < 1.0) ? p : 1.0;
}

} // namespace detail_ab

VariantResult run_variant(const Variant& variant,
                           const std::vector<ABSample>& samples,
                           const ABConfig& cfg) {
    auto scorer = cfg.scorer ? cfg.scorer : detail_ab::default_score;
    VariantResult r;
    r.name = variant.name;
    for (auto& s : samples) {
        std::string prompt = detail_ab::replace_placeholder(variant.prompt, s.input);
        std::string output = detail_ab::llm_call(prompt, variant);
        double score = scorer(output, s.expected, s.input);
        r.outputs.push_back(output);
        r.scores.push_back(score);
    }
    detail_ab::compute_stats(r);
    return r;
}

ABResult run_ab_test(const Variant& control,
                      const Variant& treatment,
                      const std::vector<ABSample>& samples,
                      const ABConfig& cfg) {
    ABResult result;
    result.alpha     = cfg.alpha;
    result.control   = run_variant(control,   samples, cfg);
    result.treatment = run_variant(treatment, samples, cfg);

    auto& a = result.control;
    auto& b = result.treatment;

    if (a.n < 2 || b.n < 2) {
        result.winner  = "tie";
        result.summary = "Not enough samples for significance test.";
        return result;
    }

    double va = a.std_dev * a.std_dev;
    double vb = b.std_dev * b.std_dev;
    double se = std::sqrt(va / (double)a.n + vb / (double)b.n);
    if (se < 1e-12) {
        result.p_value = 1.0;
    } else {
        result.t_statistic = (a.mean_score - b.mean_score) / se;
        result.p_value     = detail_ab::approx_pvalue(result.t_statistic);
    }

    double pooled = std::sqrt((va + vb) / 2.0);
    result.cohens_d    = (pooled > 1e-12) ? std::abs(a.mean_score - b.mean_score) / pooled : 0.0;
    result.significant = result.p_value < cfg.alpha;

    if (!result.significant) {
        result.winner = "tie";
    } else {
        result.winner = (a.mean_score > b.mean_score) ? "control" : "treatment";
    }

    char buf[256];
    snprintf(buf, sizeof(buf),
             "%s (control=%.3f, treatment=%.3f, p=%.4f, d=%.2f)",
             result.winner == "tie" ? "No winner" :
             result.winner == "control" ? "Control wins" : "Treatment wins",
             a.mean_score, b.mean_score, result.p_value, result.cohens_d);
    result.summary = buf;
    return result;
}

} // namespace llm
#endif // LLM_AB_IMPLEMENTATION
