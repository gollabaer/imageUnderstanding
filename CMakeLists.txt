cmake_minimum_required(VERSION 2.8)
project(image_understanding)

#Set(CMAKE_BUILD_TYPE Debug)

set(INCLUDES include/)

# Begin Sources
file(GLOB_RECURSE SRCS src/*.cpp include/*.hpp)
# End Sources

# Begin Libraries
# OpenCV	
find_package(OpenCV)

if(NOT OpenCV_FOUND)
	message( FATAL_ERROR "OpenCV Not Found exit ..." )
endif(NOT OpenCV_FOUND)

set(INCLUDES ${INCLUDES} ${OpenCV_INCLUDE_DIR})
set(LIBS ${LIBS} ${OpenCV_LIBS})

# MPEG7Fex

file(GLOB_RECURSE MPEG_SRCS ThirdParty/MPEG7Fex/*.cpp ThirdParty/MPEG7Fex/*.c)

add_library(MPEG7Fex ${MPEG_SRCS})

set_target_properties(MPEG7Fex PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(MPEG7Fex ${OpenCV_LIBRARIES})

list(APPEND INCLUDES ThirdParty/MPEG7Fex/)
list(APPEND LIBS MPEG7Fex)

# End Libraries

# Begin Executable
message(STATUS "INCLUDES: ${INCLUDES}")
include_directories(${INCLUDES})

message(STATUS "SRCS: ${SRCS}")
add_executable(caltechClassifier ${SRCS})

message(STATUS "LIBS: ${LIBS}")
target_link_libraries(caltechClassifier ${LIBS})
# End Executable

# Copy etc-files

file(GLOB_RECURSE ETC_FILES etc/*)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/etc/)
file(COPY ${ETC_FILES} DESTINATION ${CMAKE_BINARY_DIR}/etc/)
