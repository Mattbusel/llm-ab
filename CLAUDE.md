# llm-ab

Single-header C++17 A/B testing framework for LLM prompts and models with Welch t-test.

- Single-header C++17, namespace `llm`, stb-style `#ifdef LLM_LLM_AB_IMPLEMENTATION` guard
- MIT License — Mattbusel, 2026
- No `/WX` in CMakeLists (avoids C4702 on MSVC)
- Examples use `std::getenv("OPENAI_API_KEY")` for credentials
