cmake_minimum_required(VERSION 3.16)
project(minitorch VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Library ---
add_library(minitorch
    src/tensor.cpp
)

target_include_directories(minitorch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# --- GoogleTest (FetchContent) ---
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Tests ---
enable_testing()
add_executable(minitorch_tests
    tests/tensor_test.cpp
)

target_link_libraries(minitorch_tests
    PRIVATE
        minitorch
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(minitorch_tests)
