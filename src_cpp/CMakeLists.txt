cmake_minimum_required (VERSION 3.8)

project ("Boat_detection")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable (${PROJECT_NAME} "main.cpp" 
                                 "utils/region_proposal.h" "utils/region_proposal.cpp"
                                 "utils/classifier.h" "utils/classifier.cpp")
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )