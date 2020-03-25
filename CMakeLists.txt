
cmake_minimum_required(VERSION 3.10)
# set the project name
project(opencv_cuda)
find_package(X11 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# add the executable
add_executable(${PROJECT_NAME} camera_calibration.cpp)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# PTHREAD
SET(THREADS_PREFER_PTHREAD_FLAG ON)
#Add lib for fs
TARGET_LINK_LIBRARIES(${PROJECT_NAME} stdc++fs)

# Set C++14 support
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

# link opencv library
include_directories(${X11_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME} Threads::Threads)
target_link_libraries(${PROJECT_NAME} X11)