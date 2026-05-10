// src/main.cu
#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "particle.h"
#include "particle_cpu.h"
#include "particle_gpu.h"

// ---- Simulation constants ---- //
const int   MAX_N  = 10000000; // maximum particle capacity (pre-allocated)
const int   MIN_N  = 100000;   // minimum allowed particle count
const int   STEP_N = 500000;   // add/remove step size
const float DT     = 0.016f;

// ---- Simulation state (globals for callback access) ---- //
bool useGPU   = false;
int  currentN = 1000000;       // currently active particle count

// ---- Host arrays ---- //
Particle* h_particles = nullptr;  // physics data (MAX_N slots)
float*    h_colors    = nullptr;  // RGB per particle (MAX_N * 3 floats, static)
float*    h_positions = nullptr;  // CPU-mode staging buffer (MAX_N * 2 floats)

// ---- Device array ---- //
Particle* d_particles = nullptr;  // GPU physics data (MAX_N slots)

// ---- CUDA-GL interop resource (position VBO only) ---- //
cudaGraphicsResource* cudaVBOResource = nullptr;


// ---- Shader sources ---- //
// Vertex shader reads position (attrib 0) and color (attrib 1).
// Point size scales down as N increases so the window doesn't get swamped.
const char* vertexShaderSrc = R"(
    #version 460 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec3 aColor;
    uniform float uPointSize;
    out vec3 vColor;
    void main() {
        gl_Position  = vec4(aPos, 0.0, 1.0);
        gl_PointSize = uPointSize;
        vColor       = aColor;
    }
)";

const char* fragmentShaderSrc = R"(
    #version 460 core
    in  vec3 vColor;
    out vec4 FragColor;
    void main() {
        FragColor = vec4(vColor, 1.0);
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


// ---- Helper: init particles in [start, start+count) on host, ---- //
// and sync that range to the device if in GPU mode.
void initParticlesRange(int start, int count) {
    for (int i = start; i < start + count; i++) {
        h_particles[i].x  = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_particles[i].y  = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_particles[i].vx = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        h_particles[i].vy = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
    }
    // Only upload the new range — no full-array copy needed
    if (useGPU && d_particles != nullptr) {
        cudaMemcpy(d_particles + start, h_particles + start,
                   count * sizeof(Particle), cudaMemcpyHostToDevice);
    }
} // end initParticlesRange


// ---- Helper: clamp and apply a new particle count ---- //
void setParticleCount(int newN) {
    newN = std::max(MIN_N, std::min(newN, MAX_N));
    if (newN > currentN) {
        // Initialize the newly active slots
        initParticlesRange(currentN, newN - currentN);
    }
    // Decreasing: just update the count — existing data stays in the arrays
    currentN = newN;
    std::cout << "[Particles] N = " << currentN << "\n";
} // end setParticleCount


// ---- Key callback ---- //
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {

    // G: toggle CPU <-> GPU
    if (key == GLFW_KEY_G && action == GLFW_PRESS) {
        if (!useGPU) {
            // CPU -> GPU: push current host state to device
            cudaMemcpy(d_particles, h_particles,
                       currentN * sizeof(Particle), cudaMemcpyHostToDevice);
            useGPU = true;
            std::cout << "[Mode] Switched to GPU\n";
        } else {
            // GPU -> CPU: pull device state back to host
            cudaDeviceSynchronize();
            cudaMemcpy(h_particles, d_particles,
                       currentN * sizeof(Particle), cudaMemcpyDeviceToHost);
            useGPU = false;
            std::cout << "[Mode] Switched to CPU\n";
        }
    }

    // = (same physical key as +): add STEP_N particles
    if (key == GLFW_KEY_EQUAL && (action == GLFW_PRESS || action == GLFW_REPEAT))
        setParticleCount(currentN + STEP_N);

    // -: remove STEP_N particles
    if (key == GLFW_KEY_MINUS && (action == GLFW_PRESS || action == GLFW_REPEAT))
        setParticleCount(currentN - STEP_N);

    // 1-9: jump to 1M ... 9M;   0: jump to MAX_N (10M)
    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9 && action == GLFW_PRESS)
        setParticleCount((key - GLFW_KEY_0) * 1000000);
    if (key == GLFW_KEY_0 && action == GLFW_PRESS)
        setParticleCount(MAX_N);

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
    GLuint vertShader    = compileShader(GL_VERTEX_SHADER,   vertexShaderSrc);
    GLuint fragShader    = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    GLint pointSizeLoc = glGetUniformLocation(shaderProgram, "uPointSize");

    // ---------------- Allocate host memory ----------------
    h_particles = new Particle[MAX_N];
    h_colors    = new float[(long long)MAX_N * 3];
    h_positions = new float[(long long)MAX_N * 2]; // CPU-mode staging buffer

    // Generate random bright colors for all MAX_N slots upfront.
    // Colors are static — assigned once, never updated again.
    // Values biased toward [0.4, 1.0] to avoid muddy/dark particles.
    for (int i = 0; i < MAX_N; i++) {
        h_colors[i * 3]     = 0.4f + ((float)rand() / RAND_MAX) * 0.6f; // R
        h_colors[i * 3 + 1] = 0.4f + ((float)rand() / RAND_MAX) * 0.6f; // G
        h_colors[i * 3 + 2] = 0.4f + ((float)rand() / RAND_MAX) * 0.6f; // B
    }

    // Initialize the starting active particles
    initParticlesRange(0, currentN);

    // ---------------- Create VAO + VBOs ----------------
    GLuint VAO, posVBO, colorVBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &posVBO);
    glGenBuffers(1, &colorVBO);

    glBindVertexArray(VAO);

    // Attrib 0: position (x, y) — pre-allocated at MAX_N, updated every frame
    glBindBuffer(GL_ARRAY_BUFFER, posVBO);
    glBufferData(GL_ARRAY_BUFFER, (long long)MAX_N * 2 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Attrib 1: color (r, g, b) — uploaded once, never changed
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, (long long)MAX_N * 3 * sizeof(float),
                 h_colors, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Register the position VBO with CUDA for zero-copy position writes.
    // The color VBO is static and never touched by CUDA.
    cudaGraphicsGLRegisterBuffer(&cudaVBOResource, posVBO,
                                  cudaGraphicsMapFlagsWriteDiscard);

    // ---------------- Init GPU particles ----------------
    // Pre-allocate MAX_N slots on device, then sync the starting particle state.
    initParticlesGPU(&d_particles, MAX_N);
    cudaMemcpy(d_particles, h_particles,
               currentN * sizeof(Particle), cudaMemcpyHostToDevice);

    // Mouse state
    double mouseX = 0.0, mouseY = 0.0;
    bool   attracting = false;

    std::cout << "Controls:\n";
    std::cout << "  G        : toggle CPU / GPU mode\n";
    std::cout << "  = / -    : add / remove 500K particles (hold to repeat)\n";
    std::cout << "  1-9 / 0  : jump to 1M-9M / 10M particles\n";
    std::cout << "  LMB      : attract particles toward cursor\n";

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
            // ---- GPU path ---- //

            if (attracting)
                applyAttractionGPU(d_particles, currentN, mxNDC, myNDC, DT);
            updateParticlesGPU(d_particles, currentN, DT);

            // Map posVBO into CUDA address space; extract positions directly
            // into it — no device-to-host copy needed.
            float*  d_positions = nullptr;
            size_t  bufSize     = 0;
            cudaGraphicsMapResources(1, &cudaVBOResource, 0);
            cudaGraphicsResourceGetMappedPointer(
                (void**)&d_positions, &bufSize, cudaVBOResource);
            extractPositionsGPU(d_particles, d_positions, currentN);
            cudaDeviceSynchronize(); // kernel must finish before GL reads the buffer
            cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);

        } else {
            // ---- CPU path ---- //

            if (attracting) {
                for (int i = 0; i < currentN; i++) {
                    float dx     = mxNDC - h_particles[i].x;
                    float dy     = myNDC - h_particles[i].y;
                    float distSq = dx * dx + dy * dy;
                    float force  = 4.0f / distSq;
                    h_particles[i].vx += force * dx * DT;
                    h_particles[i].vy += force * dy * DT;
                } // end for
            } // end if

            updateParticlesCPU(h_particles, currentN, DT);

            // Extract positions into staging buffer, then upload to posVBO
            for (int i = 0; i < currentN; i++) {
                h_positions[i * 2]     = h_particles[i].x;
                h_positions[i * 2 + 1] = h_particles[i].y;
            }
            glBindBuffer(GL_ARRAY_BUFFER, posVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0,
                            (long long)currentN * 2 * sizeof(float), h_positions);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

        } // end if/else useGPU

        // ---- Draw ---- //
        // Point size scales down as N grows to keep the visual density reasonable
        float pointSize = std::max(1.0f, 4.0f - (float)currentN / 3000000.0f);

        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glUniform1f(pointSizeLoc, pointSize);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, currentN); // only draws currentN particles

        glfwSwapBuffers(window);
        glfwPollEvents();

        // ---- Update window title with live stats ---- //
        auto  frameEnd = std::chrono::high_resolution_clock::now();
        float ms       = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();
        int   fps      = (ms > 0.0f) ? (int)(1000.0f / ms) : 9999;

        char title[256];
        snprintf(title, sizeof(title),
                 "Particle Sim  |  %s  |  N: %d  |  FPS: %d  |  %.2f ms",
                 useGPU ? "GPU (G=CPU)" : "CPU (G=GPU)",
                 currentN, fps, ms);
        glfwSetWindowTitle(window, title);

    } // end render loop

    // ---------------- Cleanup ----------------
    cudaGraphicsUnregisterResource(cudaVBOResource);
    freeParticlesGPU(d_particles);
    delete[] h_particles;
    delete[] h_colors;
    delete[] h_positions;
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &posVBO);
    glDeleteBuffers(1, &colorVBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return 0;
} // end main
