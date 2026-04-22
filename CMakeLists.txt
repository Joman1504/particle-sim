cmake_minimum_required(VERSION 3.20)
project(ParticleSim CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(OpenGL REQUIRED)

add_executable(particle_sim
    src/main.cu
    src/particle_cpu.cpp
    src/particle_gpu.cu
    src/glad.c
)

target_include_directories(particle_sim PRIVATE
    include/
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(particle_sim PRIVATE
    OpenGL::GL
    ${CMAKE_SOURCE_DIR}/lib/glfw3.lib
)

target_link_libraries(particle_sim PRIVATE gdi32 user32 shell32)