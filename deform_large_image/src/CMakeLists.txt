cmake_minimum_required(VERSION 2.8)
 
project(DeformLargeImage)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(DeformLargeImage MACOSX_BUNDLE DeformLargeImage.cxx)
target_link_libraries(DeformLargeImage
 ${ITK_LIBRARIES})
