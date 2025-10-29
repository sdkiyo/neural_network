cmake_minimum_required (VERSION 3.31.7)

project( test1
	VERSION 0.0.0
	LANGUAGES C )

set (CMAKE_C_STANDARD 23)

set(P ../../src/)
add_library(test_lib SHARED ${P}activation.c ${P}network.c ${P}print_network.c ${P}train.c)
target_include_directories(test_lib PUBLIC ../../include)
target_link_libraries(test_lib PUBLIC m)


add_executable(${PROJECT_NAME} main.c)
target_link_libraries(${PROJECT_NAME} PRIVATE test_lib)
