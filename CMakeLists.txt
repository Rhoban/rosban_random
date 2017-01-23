cmake_minimum_required(VERSION 2.8.3)
project(rosban_random)

find_package(catkin REQUIRED)

find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rosban_random
  DEPENDS Eigen3
)

include_directories(
  include
  ${Eigen3_INCLUDE_DIRS}
  )

set(DIRECTORIES
  src/rosban_random
  )

# Build ALL_SOURCES
set (SOURCES)
set (ALL_SOURCES)
set (PREFIXED_SOURCES)
foreach (DIRECTORY ${DIRECTORIES})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}")
    include (${DIRECTORY}/sources.cmake)

    foreach (SOURCE ${SOURCES})
        set (PREFIXED_SOURCES ${PREFIXED_SOURCES} ${DIRECTORY}/${SOURCE})
    endforeach (SOURCE)

    set (ALL_SOURCES ${ALL_SOURCES} ${PREFIXED_SOURCES})
endforeach (DIRECTORY)

add_library(rosban_random
  ${ALL_SOURCES}
)