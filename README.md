# llm-ab

[![CI](https://github.com/Mattbusel/llm-ab/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-ab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Single header](https://img.shields.io/badge/single-header-green.svg)

A/B test two LLM prompts or models in C++ and get a real significance test back.

> Part of **[llm-cpp](https://github.com/Mattbusel/llm-cpp)**, a family of 26 single-header C++ libraries for building on LLM APIs. Each one stands alone: copy one header, include it, done.

Switching prompts because one output "looked better" is guesswork. llm-ab runs both variants over the same samples, scores every output, and applies Welch's t-test, so you can tell a real improvement from noise before you ship it.

## Features

- Two variants, each with its own prompt template (`{input}` placeholder), model, temperature, endpoint URL and API key
- Default scorer: case-insensitive exact match against the sample's `expected` answer
- Custom scorers: pass any `double(output, expected, input)` function, for example keyword checks or an LLM judge
- Welch's t-test with an exact two-tailed p-value, Cohen's d effect size and a configurable alpha
- Per-variant outputs, scores, mean and standard deviation, plus a winner of `control`, `treatment`, `tie` or `error`
- Works with any OpenAI-compatible `/v1/chat/completions` endpoint through `Variant::api_url`

## Quick start

Requirements: a C++17 compiler and libcurl (`apt install libcurl4-openssl-dev`, `brew install curl`, or `vcpkg install curl`).

1. Copy [`include/llm_ab.hpp`](include/llm_ab.hpp) into your project.
2. In exactly one `.cpp` file, `#define LLM_AB_IMPLEMENTATION` before including it. Other files just `#include "llm_ab.hpp"`.

```cpp
#define LLM_AB_IMPLEMENTATION
#include "llm_ab.hpp"
#include <cstdlib>
#include <iostream>

int main() {
    llm::ABConfig cfg;
    cfg.api_key = std::getenv("OPENAI_API_KEY");

    llm::Variant control;
    control.name   = "plain";
    control.prompt = "Answer with one word. {input}";
    control.model  = "gpt-4o-mini";

    llm::Variant treatment = control;
    treatment.name   = "expert";
    treatment.prompt = "You are a geography teacher. Answer with one word. {input}";

    std::vector<llm::ABSample> samples = {
        {"Capital of France?", "Paris"},
        {"Capital of Japan?",  "Tokyo"},
        {"Capital of Canada?", "Ottawa"},
    };

    llm::ABResult r = llm::run_ab_test(control, treatment, samples, cfg);
    std::cout << r.summary << "\n";  // "Treatment wins (control=0.667, treatment=1.000, p=..., d=...)"
}
```

Build and run:

```bash
g++ -std=c++17 -I include example.cpp -o example -lcurl
export OPENAI_API_KEY=sk-...
./example
```

## API

Everything lives in namespace `llm`.

| Function / type | What it does |
|---|---|
| `run_ab_test(control, treatment, samples, cfg)` | Run both variants over the samples and return an `ABResult` with the t-test, p-value, Cohen's d, winner and a one-line summary |
| `run_variant(variant, samples, cfg)` | Run a single variant and return its outputs, scores, mean and standard deviation |
| `Variant`, `ABSample`, `ABConfig` | Inputs: prompt/model per variant, input/expected per sample, shared API key, alpha, timeout and scorer |

## How it works

For each sample the header substitutes `{input}` into the variant's prompt, sends one chat completion through libcurl, and scores the reply. Once both variants have run it computes the mean and standard deviation of each score list, the Welch t statistic and degrees of freedom, and a two-tailed p-value from the regularised incomplete beta function. The result is significant when `p < alpha`; the variant with the higher mean wins.

## Examples

The [`examples/`](examples) folder has runnable programs:

- [`basic_ab.cpp`](examples/basic_ab.cpp)
- [`local_ab.cpp`](examples/local_ab.cpp)
- [`model_comparison.cpp`](examples/model_comparison.cpp)
- [`scored_ab.cpp`](examples/scored_ab.cpp)

Build the examples with CMake (needs libcurl):

```bash
cmake -B build
cmake --build build
```

Examples that call the API read `OPENAI_API_KEY` from the environment.

## Limitations

- Requests run sequentially, one sample at a time, with no retry on rate limits or network errors. An API error stops that variant and sets `winner` to `"error"`.
- At least two samples per variant are needed for a test. With the default 0/1 exact-match scorer you need a reasonable number of samples to reach significance.

## License

MIT. See [LICENSE](LICENSE).
