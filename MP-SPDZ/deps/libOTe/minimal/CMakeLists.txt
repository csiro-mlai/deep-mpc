
#project(minimal)


file(GLOB_RECURSE SRCS *.cpp ../frontend/util.cpp)

include_directories(${CMAKE_SOURCE_DIR})


add_executable(minimal  ${SRCS})

target_link_libraries(minimal libOTe_Tests tests_cryptoTools)
