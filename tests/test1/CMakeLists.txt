cmake_minimum_required(VERSION 3.31.7)

project( test1
	LANGUAGES C )

set(CMAKE_C_STANDARD 23)

set(P ../../src/)
add_library(nn_base_lib SHARED
	${P}activation.c
	${P}network.c
	${P}print_network.c
	${P}activation.c
)
target_include_directories(nn_base_lib PUBLIC ../../include)
target_link_libraries(nn_base_lib PUBLIC m)
set_target_properties(nn_base_lib PROPERTIES
	PREFIX "lib_"
	OUTPUT_NAME "nn_base"
	SUFFIX ".so"
)


add_library(nn_train_lib SHARED
	${P}train.c
	${P}loss.c
	${P}derivative.c
)
target_include_directories(nn_train_lib PUBLIC ../../include)
target_link_libraries(nn_train_lib PUBLIC m)
set_target_properties(nn_train_lib PROPERTIES
	PREFIX "lib_"
	OUTPUT_NAME "nn_train"
	SUFFIX ".so"
)


add_executable(${PROJECT_NAME} main.c)
