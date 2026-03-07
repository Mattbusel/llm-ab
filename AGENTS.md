# llm-ab

Single-header C++17 A/B testing framework for LLM prompts and models with Welch t-test.

## Structure
- `include/llm_ab.hpp` — single-header implementation
- `examples/` — usage examples
- `CMakeLists.txt` — cmake build (requires vcpkg curl where applicable)

## Build
```bash
cmake -B build && cmake --build build
```
