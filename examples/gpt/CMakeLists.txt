add_library(gpt_lib
    src/tokenizer.cpp
    src/gpt.cpp
)

target_include_directories(gpt_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gpt_lib
    PUBLIC
        minitorch
)

add_executable(train_gpt
    src/train_gpt.cpp
)

target_link_libraries(train_gpt
    PRIVATE
        gpt_lib
        minitorch
)

# GPT tests
add_executable(gpt_tests
    tests/gpt_test.cpp
)

target_link_libraries(gpt_tests
    PRIVATE
        gpt_lib
        minitorch
        GTest::gtest_main
)

gtest_discover_tests(gpt_tests)
