cmake_minimum_required(VERSION 3.0.2)
project(depth_anything_ros)

# 添加 C++14 支持
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++14)

# 添加優化選項
add_compile_options(-O3 -march=native)
add_definitions(-DNDEBUG)  # 關閉調試
add_definitions(-w)        # 關閉警告

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  jsk_data
  pcl_ros
  pcl_conversions
  visualization_msgs
  image_transport
  tf2_ros
  nodelet
  depth_image_proc
)

# Find CUDA
find_package(CUDA REQUIRED)
# Enable CUDA optimization
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math)
set(CUDA_PROPAGATE_HOST_FLAGS ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find PCL
find_package(PCL 1.10 REQUIRED)

# Set TensorRT path and optimization
if(NOT DEFINED ENV{TENSORRT_DIR})
    set(TENSORRT_DIR "/usr/lib/x86_64-linux-gnu")
    set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
else()
    set(TENSORRT_DIR "$ENV{TENSORRT_DIR}")
    set(TENSORRT_INCLUDE_DIR "$ENV{TENSORRT_DIR}/include")
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

link_directories(
  ${TENSORRT_DIR}
  ${PCL_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# TensorRT libraries
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    cv_bridge
    jsk_data
    pcl_ros
    pcl_conversions
    visualization_msgs
    image_transport
    tf2_ros
    nodelet
    depth_image_proc
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

# 只包含原始的源文件
set(LIB_SOURCES
  src/utils.cpp
  src/depth_anything.cpp
)

# Add library with optimization flags
add_library(${PROJECT_NAME} ${LIB_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-O3")

# Add executable
add_executable(depth_estimation_node src/depth_estimation_node.cpp)

# Link libraries with optimization
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBS}
  ${PCL_LIBRARIES}
)

target_link_libraries(depth_estimation_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBS}
  ${PCL_LIBRARIES}
)

# Install targets
install(TARGETS ${PROJECT_NAME} depth_estimation_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install header files
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# Install nodelet plugin description
install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# 在現有的 executable 部分後添加
add_executable(depth_visualizer_node src/depth_visualizer_node.cpp)
target_link_libraries(depth_visualizer_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)
