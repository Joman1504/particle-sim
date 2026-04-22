// src/main.cu
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void testKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = idx * 2.0f;
}

int main() {
    // --- Test CUDA ---
    const int N = 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    testKernel<<<(N + 255) / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    std::cout << "CUDA kernel ran successfully.\n";
    cudaFree(d_data);

    // --- Test OpenGL/GLFW ---
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return -1; }
    GLFWwindow* window = glfwCreateWindow(800, 600, "Particle Sim", nullptr, nullptr);
    if (!window) { std::cerr << "Window creation failed\n"; return -1; }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n"; return -1;
    }
    std::cout << "OpenGL " << glGetString(GL_VERSION) << "\n";

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}