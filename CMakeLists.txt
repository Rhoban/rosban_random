cmake_minimum_required(VERSION 2.8.3)
project(rosban_random)

find_package(catkin REQUIRED COMPONENTS
  Eigen)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rosban_random
  CATKIN_DEPENDS Eigen
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
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