// src/main.cu
#include <iostream>
#include <chrono>
#include <string>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "particle.h"
#include "particle_cpu.h"
#include "particle_gpu.h"


const int   N  = 10000000; // number of particles
const float DT = 0.016f;

// ---- Simulation mode ---- //
// Toggled with the G key. When true, physics run on the GPU.
bool useGPU = false;

// ---- Particle arrays ---- //
Particle* h_particles = nullptr;  // CPU-side array
Particle* d_particles = nullptr;  // GPU-side array (device pointer)

// ---- CUDA-OpenGL interop resource ---- //
// Registering the VBO with CUDA lets kernels write positions directly into
// it without a device-to-host-to-GL copy every frame.
cudaGraphicsResource* cudaVBOResource = nullptr;


// ---- Shader sources ---- //
const char* vertexShaderSrc = R"(
    #version 460 core
    layout (location = 0) in vec2 aPos;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        gl_PointSize = 4.0;
    }
)";

const char* fragmentShaderSrc = R"(
    #version 460 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(1.0, 0.6, 0.1, 1.0); // orange particles
    }
)";


// ---- Compile a shader and check for errors ---- //
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Shader error: " << log << "\n";
    }
    return shader;
} // end compileShader


// ---- Key callback: G toggles CPU/GPU mode ---- //
// On switch, particle state is synced between host and device so the
// simulation continues seamlessly from the same positions/velocities.
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_G && action == GLFW_PRESS) {
        if (!useGPU) {
            // CPU -> GPU: upload current CPU state to device
            cudaMemcpy(d_particles, h_particles,
                       N * sizeof(Particle), cudaMemcpyHostToDevice);
            useGPU = true;
            std::cout << "[Mode] Switched to GPU\n";
        } else {
            // GPU -> CPU: download current device state to host
            cudaDeviceSynchronize();
            cudaMemcpy(h_particles, d_particles,
                       N * sizeof(Particle), cudaMemcpyDeviceToHost);
            useGPU = false;
            std::cout << "[Mode] Switched to CPU\n";
        }
    }
} // end keyCallback


int main() {
    // ---------------- Init GLFW ----------------
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Particle Sim", nullptr, nullptr);
    if (!window) {
        std::cerr << "Window creation failed\n";
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n";
        return -1;
    }

    glEnable(GL_PROGRAM_POINT_SIZE);

    // ---------------- Build shader program ----------------
    GLuint vertShader   = compileShader(GL_VERTEX_SHADER,   vertexShaderSrc);
    GLuint fragShader   = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    // ---------------- Create VBO + VAO ----------------
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Allocate VBO for N particles (x, y per particle)
    glBufferData(GL_ARRAY_BUFFER, N * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Register the VBO with CUDA so kernels can write into it directly.
    // WriteDiscard signals that CUDA will overwrite the entire buffer each frame.
    cudaGraphicsGLRegisterBuffer(&cudaVBOResource, VBO,
                                  cudaGraphicsMapFlagsWriteDiscard);

    // ---------------- Init particles ----------------
    // Both arrays start from the same random state.
    h_particles = new Particle[N];
    initParticlesCPU(h_particles, N);

    // Alloc device array and copy the CPU state up so both modes start identical.
    initParticlesGPU(&d_particles, N);
    cudaMemcpy(d_particles, h_particles, N * sizeof(Particle), cudaMemcpyHostToDevice);

    // CPU-side staging buffer for position upload (used only in CPU mode)
    float* h_positions = new float[N * 2];

    // Mouse state
    double mouseX = 0.0, mouseY = 0.0;
    bool attracting = false;

    std::cout << "Press G to toggle between CPU and GPU simulation.\n";
    std::cout << "Hold left mouse button to attract particles.\n";

    // ---------------- Render loop ----------------
    while (!glfwWindowShouldClose(window)) {
        auto frameStart = std::chrono::high_resolution_clock::now();

        // ---- Mouse input ---- //
        attracting = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        // Convert screen coords to NDC
        float mxNDC = (float)(mouseX / 1920.0) * 2.0f - 1.0f;
        float myNDC = 1.0f - (float)(mouseY / 1080.0) * 2.0f; // flip Y

        if (useGPU) {
            // ---- GPU mode ---- //

            // Apply attraction and physics on the device
            if (attracting) {
                applyAttractionGPU(d_particles, N, mxNDC, myNDC, DT);
            }
            updateParticlesGPU(d_particles, N, DT);

            // Map the VBO into CUDA address space and extract positions
            // directly into it — no host round-trip.
            float*  d_positions = nullptr;
            size_t  bufSize     = 0;
            cudaGraphicsMapResources(1, &cudaVBOResource, 0);
            cudaGraphicsResourceGetMappedPointer(
                (void**)&d_positions, &bufSize, cudaVBOResource);

            extractPositionsGPU(d_particles, d_positions, N);
            cudaDeviceSynchronize(); // ensure kernel finishes before GL reads the buffer

            cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);

        } else {
            // ---- CPU mode ---- //

            // Apply attraction force
            if (attracting) {
                for (int i = 0; i < N; i++) {
                    float dx     = mxNDC - h_particles[i].x;
                    float dy     = myNDC - h_particles[i].y;
                    float distSq = dx * dx + dy * dy + 0.0001f;
                    float force  = 4.0f / distSq;
                    h_particles[i].vx += force * dx * DT;
                    h_particles[i].vy += force * dy * DT;
                } // end for
            } // end if

            updateParticlesCPU(h_particles, N, DT);

            // Extract positions into flat array, then upload to VBO
            for (int i = 0; i < N; i++) {
                h_positions[i * 2]     = h_particles[i].x;
                h_positions[i * 2 + 1] = h_particles[i].y;
            } // end for

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, N * 2 * sizeof(float), h_positions);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

        } // end if/else useGPU

        // ---- Draw ---- //
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, N);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // ---- Update window title with mode + timing ---- //
        auto  frameEnd = std::chrono::high_resolution_clock::now();
        float ms       = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();
        int   fps      = (int)(1000.0f / ms);

        std::string title = "Particle Sim  |  Mode: ";
        title += (useGPU ? "GPU (press G for CPU)" : "CPU (press G for GPU)");
        title += "  |  FPS: "   + std::to_string(fps);
        title += "  |  Frame: " + std::to_string(ms).substr(0, 5) + " ms";
        glfwSetWindowTitle(window, title.c_str());

    } // end render loop

    // ---------------- Cleanup ----------------
    cudaGraphicsUnregisterResource(cudaVBOResource);
    freeParticlesGPU(d_particles);
    delete[] h_particles;
    delete[] h_positions;
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return 0;
} // end main
