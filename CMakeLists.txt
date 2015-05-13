cmake_minimum_required(VERSION 2.8)
project(image_understanding)

#Set(CMAKE_BUILD_TYPE Debug)

set(INCLUDES include/)

# Begin Sources
file(GLOB SRCS src/*.cpp include/*.hpp)
# End Sources

# Begin Libraries
# OpenCV	
find_package(OpenCV)

if(NOT OpenCV_FOUND)
	message( FATAL_ERROR "OpenCV Not Found exit ..." )
endif(NOT OpenCV_FOUND)

set(INCLUDES ${INCLUDES} ${OpenCV_INCLUDE_DIR})
set(LIBS ${LIBS} ${OpenCV_LIBS})

# End Libraries

# Begin Executable
message(STATUS "INCLUDES: ${INCLUDES}")
include_directories(${INCLUDES})

message(STATUS "SRCS: ${SRCS}")
add_executable(caltechClassifier ${SRCS})

message(STATUS "LIBS: ${LIBS}")
target_link_libraries(caltechClassifier ${LIBS})
# End Executable

