cmake_minimum_required (VERSION 3.31.7)

project( nn
	VERSION 0.0.0
	LANGUAGES C )

set (CMAKE_C_STANDARD 23)

set(P src/)
add_library(test_lib SHARED ${P}activation.c ${P}network.c ${P}train.c)

target_link_libraries(test_lib PUBLIC m)# i love c
