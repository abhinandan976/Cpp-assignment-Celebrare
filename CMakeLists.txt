cmake_minimum_required(VERSION 3.10)
project(ImageEnhancer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/downloads/opencv/build")
set(Torch_DIR"C:/downloads/libtorch/share/cmake/Torch")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(enhancer main.cpp)

target_link_libraries(enhancer PRIVATE "${OpenCV_LIBS}" "${TORCH_LIBS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")