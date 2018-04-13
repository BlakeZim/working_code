cmake_minimum_required(VERSION 2.8)
 
project(DeformLargeVolume)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(DeformLargeVolume MACOSX_BUNDLE DeformLargeVolume.cxx)
target_link_libraries(DeformLargeVolume
 ${ITK_LIBRARIES})
