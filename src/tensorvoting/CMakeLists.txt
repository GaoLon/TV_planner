cmake_minimum_required(VERSION 3.0.2)
project(tensorvoting CXX CUDA)

add_compile_options(-std=c++14)
# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Eigen3 REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost COMPONENTS thread filesystem system REQUIRED)

# add pointmatcher, pointmatcher_ros, libnabo
find_package(catkin REQUIRED COMPONENTS
  roscpp
  libpointmatcher_ros
  geometry_msgs
  visualization_msgs
)

catkin_package(
  INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS libpointmatcher_ros roscpp
  DEPENDS EIGEN3 libpointmatcher PCL
)

include_directories(
    SYSTEM
    include
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
	  ${Boost_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

link_directories(${LINK_DIRECTORIES} /usr/lib)
link_directories(${Boost_LIBRARY_DIR})
link_directories(${PCL_LIBRARY_DIRS})

# look for yaml-cpp library
find_path(yaml-cpp_INCLUDE_DIRS yaml-cpp/yaml.h
	/usr/local/include
	/usr/include
)
find_library(yaml-cpp_LIBRARIES yaml-cpp PATHS
	/usr/local/lib
	/usr/lib
NO_DEFAULT_PATH)
if (yaml-cpp_INCLUDE_DIRS AND yaml-cpp_LIBRARIES)
	include_directories(${yaml-cpp_INCLUDE_DIRS})
	set(EXTRA_LIBS ${EXTRA_LIBS} ${yaml-cpp_LIBRARIES})
	add_definitions(-DHAVE_YAML_CPP)
endif (yaml-cpp_INCLUDE_DIRS AND yaml-cpp_LIBRARIES)

# CUDA
find_package(CUDA REQUIRED)

SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math;)
set(CUDA_NVCC_FLAGS
  -gencode arch=compute_75,code=sm_75;
)
set(CUDA_SEPARABLE_COMPILATION ON)

add_library(CudaVoting
  src/CudaVoting.cu
)
target_link_libraries(CudaVoting 
  cuda cudart rt cublas cudadevrt
  ${CUDA_LIBRARIES}
)

# sparse voting demo
add_executable(demo_sparse_gpu src/demo_sparse_gpu.cpp src/tensor_voting.cpp)
target_link_libraries(demo_sparse_gpu 
  CudaVoting 
  ${catkin_LIBRARIES}
  ${POINT_MATCHER_LIBRARY}
  ${NABO_LIBRARY} 
  ${EXTRA_LIBS} 
  ${Boost_LIBRARIES}
)

# sparse + dense stick voting demo (use unified GPU implementation)
add_executable(demo_dense_gpu src/demo_dense_gpu.cpp src/tensor_voting.cpp)
target_link_libraries(demo_dense_gpu 
  CudaVoting ${catkin_LIBRARIES}
  ${POINT_MATCHER_LIBRARY}
  ${NABO_LIBRARY} 
  ${EXTRA_LIBS} 
  ${Boost_LIBRARIES}
)

# test planner
add_executable(test_plan src/test_plan.cpp src/tensor_voting.cpp)
target_link_libraries(test_plan 
  CudaVoting ${catkin_LIBRARIES}
  ${POINT_MATCHER_LIBRARY}
  ${NABO_LIBRARY} 
  ${EXTRA_LIBS} 
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
)
