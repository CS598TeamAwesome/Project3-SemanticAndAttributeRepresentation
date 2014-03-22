cmake_minimum_required (VERSION 2.6)

project(SemanticAndAttributeRepresentation)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
find_package(LibSVM REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(Train)
#add_subdirectory(Test)

