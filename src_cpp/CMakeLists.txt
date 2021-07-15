cmake_minimum_required (VERSION 3.8)

project ("Boat_detection")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable (${PROJECT_NAME}  "main.cpp" 
                                 "utils/region_proposal.h" "utils/region_proposal.cpp"
                                 "utils/boat_detector.h" "utils/boat_detector.cpp" 
                                 "utils/boat_detector_process.cpp")

file(COPY "../src_python/frozen_graph.pb"
          DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )