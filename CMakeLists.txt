cmake_minimum_required(VERSION 3.5)
project(LicensePlateDetector CXX)

# Compile Options
set(CMAKE_CXX_STANDARD 17)

# Source Files
add_executable(${CMAKE_PROJECT_NAME} main.cpp)

# Configure OpenCV Library
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
