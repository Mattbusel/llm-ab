cmake_minimum_required(VERSION 3.14)
project(llm-ab CXX)
set(CMAKE_CXX_STANDARD 17)

find_library(CURL_LIB NAMES curl libcurl PATHS "C:/Users/Matthew/vcpkg/installed/x64-windows/lib" NO_DEFAULT_PATH)
set(CURL_INC "C:/Users/Matthew/vcpkg/installed/x64-windows/include")

foreach(EXAMPLE basic_ab scored_ab model_comparison local_ab)
    add_executable(${EXAMPLE} examples/${EXAMPLE}.cpp)
    target_include_directories(${EXAMPLE} PRIVATE include ${CURL_INC})
    target_link_libraries(${EXAMPLE} PRIVATE ${CURL_LIB})
    target_compile_options(${EXAMPLE} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
    )
endforeach()
