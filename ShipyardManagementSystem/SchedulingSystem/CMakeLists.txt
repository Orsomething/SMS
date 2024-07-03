cmake_minimum_required(VERSION 3.10)
project(SchedulingSystem)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
add_executable(SchedulingSystem src/main.cpp src/Scheduler.cpp src/DataLoader.cpp)
