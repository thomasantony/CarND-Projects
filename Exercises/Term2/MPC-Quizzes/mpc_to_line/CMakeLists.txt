project(mpc)
# export PYTHONHOME=/Users/tantony/anaconda
cmake_minimum_required (VERSION 3.5)

# -g allows for gdb debugging
# turn on -O3 for best performance
add_definitions(-std=c++11)
add_definitions(-O3)
add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=/Users/tantony/anaconda/include/python3.5m/Python.h)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(sources src/MPC.cpp)

include_directories(src/Eigen-3.3)
include_directories(/usr/local/include)
include_directories(/Users/tantony/anaconda/include/python3.5m)

link_directories(/usr/local/lib)
link_directories(/Users/tantony/anaconda/lib)

add_executable(mpc ${sources})

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
target_link_libraries(mpc ipopt python3.5)
else()
target_link_libraries(mpc ipopt python3.5m)
endif()
