// src/main.cu

#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "particle.h"
#include "particle_cpu.h"
#include "particle_gpu.h"

// ============================================================
// Simulation constants
// ============================================================

const int   MAX_N  = 10000000;
const int   MIN_N  = 100;
const int   STEP_N = 10000;
const float DT     = 0.016f;

// Same magnitude as simulation gravity when gravity is on (must match physics files).
const float BASE_GRAVITY = -9.8f;

// dt multiplier when combined float + slow-mo mode is on.
const float SLOW_MO_FACTOR = 0.25f;

const float VIEWPORT_W = 1920.0f;
const float VIEWPORT_H = 1080.0f;

// ============================================================
// Triangle obstacle
// ============================================================

struct Triangle {
    float x1, y1;
    float x2, y2;
    float x3, y3;
};

Triangle gTriangle = {
    -0.2f, -0.1f,
     0.2f, -0.1f,
     0.0f,  0.3f
};

bool draggingTriangle = false;
float dragOffsetX = 0.0f;
float dragOffsetY = 0.0f;

// ============================================================
// Simulation state
// ============================================================

bool  useGPU     = false;
int   currentN   = 10000;
float windX      = 0.0f;
float spawnSpeed = 0.5f;

// Zero-G + slow-mo together (toggle with M).
bool floatSlowMo = false;

// ============================================================
// Host arrays
// ============================================================

Particle* h_particles = nullptr;
float*    h_colors    = nullptr;
float*    h_radii     = nullptr;
float*    h_positions = nullptr;

// ============================================================
// Device array
// ============================================================

Particle* d_particles = nullptr;

// ============================================================
// CUDA-GL interop
// ============================================================

cudaGraphicsResource* cudaVBOResource = nullptr;

// ============================================================
// Shader sources
// ============================================================

const char* vertexShaderSrc = R"(
#version 460 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aRadius;

uniform float uViewportH;

out vec3 vColor;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    gl_PointSize = aRadius * uViewportH;
    vColor = aColor;
}
)";

const char* fragmentShaderSrc = R"(
#version 460 core

in vec3 vColor;
out vec4 FragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);

    if (dot(coord, coord) > 0.25)
        discard;

    FragColor = vec4(vColor, 1.0);
}
)";

// Core-profile triangle (immediate mode is invalid in GL 3.2+ core).
const char* triVertShaderSrc = R"(
#version 460 core
layout (location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

const char* triFragShaderSrc = R"(
#version 460 core
out vec4 FragColor;
uniform vec3 uColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}
)";

// ============================================================
// Shader compile helper
// ============================================================

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
}

// ============================================================
// Triangle helpers
// ============================================================

float sign(float x1, float y1,
           float x2, float y2,
           float x3, float y3) {
    return (x1 - x3) * (y2 - y3) -
           (x2 - x3) * (y1 - y3);
}

bool pointInTriangle(float px, float py, const Triangle& t) {

    float d1 = sign(px, py, t.x1, t.y1, t.x2, t.y2);
    float d2 = sign(px, py, t.x2, t.y2, t.x3, t.y3);
    float d3 = sign(px, py, t.x3, t.y3, t.x1, t.y1);

    bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(hasNeg && hasPos);
}

// ============================================================
// Particle initialization
// ============================================================

void initParticlesRange(int start, int count) {

    for (int i = start; i < start + count; i++) {

        h_particles[i].x =
            ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        h_particles[i].y =
            ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        h_particles[i].vx = 0.0f;
        h_particles[i].vy = -spawnSpeed;

        h_particles[i].r = h_radii[i];
    }

    if (useGPU && d_particles != nullptr) {

        cudaMemcpy(d_particles + start,
                   h_particles + start,
                   count * sizeof(Particle),
                   cudaMemcpyHostToDevice);
    }
}

// ============================================================
// Particle count
// ============================================================

void setParticleCount(int newN) {

    newN = std::max(MIN_N,
           std::min(newN, MAX_N));

    if (newN > currentN)
        initParticlesRange(currentN,
                           newN - currentN);

    currentN = newN;

    std::cout << "[Particles] N = "
              << currentN << "\n";
}

// ============================================================
// Keyboard
// ============================================================

void keyCallback(GLFWwindow* window,
                 int key,
                 int scancode,
                 int action,
                 int mods) {

    if (key == GLFW_KEY_G &&
        action == GLFW_PRESS) {

        if (!useGPU) {

            cudaMemcpy(d_particles,
                       h_particles,
                       currentN * sizeof(Particle),
                       cudaMemcpyHostToDevice);

            useGPU = true;

            std::cout << "[Mode] GPU\n";
        }
        else {

            cudaMemcpy(h_particles,
                       d_particles,
                       currentN * sizeof(Particle),
                       cudaMemcpyDeviceToHost);

            useGPU = false;

            std::cout << "[Mode] CPU\n";
        }
    }

    if (key == GLFW_KEY_EQUAL &&
       (action == GLFW_PRESS ||
        action == GLFW_REPEAT))
        setParticleCount(currentN + STEP_N);

    if (key == GLFW_KEY_MINUS &&
       (action == GLFW_PRESS ||
        action == GLFW_REPEAT))
        setParticleCount(currentN - STEP_N);

    if (key == GLFW_KEY_M &&
        action == GLFW_PRESS) {

        floatSlowMo = !floatSlowMo;

        std::cout << "[Float + slow-mo] "
                  << (floatSlowMo ? "on\n" : "off\n");
    }
}

// ============================================================
// Main
// ============================================================

int main() {

    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

    GLFWwindow* window =
        glfwCreateWindow(
            (int)VIEWPORT_W,
            (int)VIEWPORT_H,
            "Particle Sim",
            nullptr,
            nullptr);

    if (!window) {
        std::cerr << "Window failed\n";
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, keyCallback);

    if (!gladLoadGLLoader(
        (GLADloadproc)glfwGetProcAddress)) {

        std::cerr << "GLAD failed\n";
        return -1;
    }

    glEnable(GL_PROGRAM_POINT_SIZE);

    GLuint vertShader =
        compileShader(GL_VERTEX_SHADER,
                       vertexShaderSrc);

    GLuint fragShader =
        compileShader(GL_FRAGMENT_SHADER,
                       fragmentShaderSrc);

    GLuint shaderProgram = glCreateProgram();

    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);

    glLinkProgram(shaderProgram);

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    glUseProgram(shaderProgram);

    glUniform1f(
        glGetUniformLocation(shaderProgram,
        "uViewportH"),
        VIEWPORT_H);

    GLuint triVertShader =
        compileShader(GL_VERTEX_SHADER,
                      triVertShaderSrc);

    GLuint triFragShader =
        compileShader(GL_FRAGMENT_SHADER,
                      triFragShaderSrc);

    GLuint triProgram = glCreateProgram();

    glAttachShader(triProgram, triVertShader);
    glAttachShader(triProgram, triFragShader);

    glLinkProgram(triProgram);

    glDeleteShader(triVertShader);
    glDeleteShader(triFragShader);

    GLuint triVAO, triVBO;

    glGenVertexArrays(1, &triVAO);
    glGenBuffers(1, &triVBO);

    glBindVertexArray(triVAO);

    glBindBuffer(GL_ARRAY_BUFFER, triVBO);

    float triInit[6] = {
        gTriangle.x1, gTriangle.y1,
        gTriangle.x2, gTriangle.y2,
        gTriangle.x3, gTriangle.y3
    };

    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(triInit),
                 triInit,
                 GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2,
                          GL_FLOAT,
                          GL_FALSE,
                          2 * sizeof(float),
                          (void*)0);

    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    GLint triColorLoc =
        glGetUniformLocation(triProgram, "uColor");

    // ========================================================
    // Allocate memory
    // ========================================================

    h_particles = new Particle[MAX_N];
    h_colors    = new float[(long long)MAX_N * 3];
    h_radii     = new float[MAX_N];
    h_positions = new float[(long long)MAX_N * 2];

    for (int i = 0; i < MAX_N; i++) {

        h_colors[i * 3] =
            0.4f + ((float)rand() / RAND_MAX) * 0.6f;

        h_colors[i * 3 + 1] =
            0.4f + ((float)rand() / RAND_MAX) * 0.6f;

        h_colors[i * 3 + 2] =
            0.4f + ((float)rand() / RAND_MAX) * 0.6f;

        h_radii[i] =
            R_MIN +
            ((float)rand() / RAND_MAX)
            * (R_MAX - R_MIN);
    }

    initParticlesRange(0, currentN);

    // ========================================================
    // VAO/VBO
    // ========================================================

    GLuint VAO, posVBO, colorVBO, radiusVBO;

    glGenVertexArrays(1, &VAO);

    glGenBuffers(1, &posVBO);
    glGenBuffers(1, &colorVBO);
    glGenBuffers(1, &radiusVBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, posVBO);

    glBufferData(GL_ARRAY_BUFFER,
                 (long long)MAX_N *
                 2 * sizeof(float),
                 nullptr,
                 GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2,
                          GL_FLOAT,
                          GL_FALSE,
                          2 * sizeof(float),
                          (void*)0);

    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);

    glBufferData(GL_ARRAY_BUFFER,
                 (long long)MAX_N *
                 3 * sizeof(float),
                 h_colors,
                 GL_STATIC_DRAW);

    glVertexAttribPointer(1, 3,
                          GL_FLOAT,
                          GL_FALSE,
                          3 * sizeof(float),
                          (void*)0);

    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, radiusVBO);

    glBufferData(GL_ARRAY_BUFFER,
                 (long long)MAX_N *
                 sizeof(float),
                 h_radii,
                 GL_STATIC_DRAW);

    glVertexAttribPointer(2, 1,
                          GL_FLOAT,
                          GL_FALSE,
                          sizeof(float),
                          (void*)0);

    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    // ========================================================
    // CUDA interop
    // ========================================================

    cudaGraphicsGLRegisterBuffer(
        &cudaVBOResource,
        posVBO,
        cudaGraphicsMapFlagsWriteDiscard);

    initParticlesGPU(&d_particles, MAX_N);

    // ========================================================
    // Render loop
    // ========================================================

    while (!glfwWindowShouldClose(window)) {

        double mouseX, mouseY;

        glfwGetCursorPos(window,
                         &mouseX,
                         &mouseY);

        float mx =
            (float)(mouseX / VIEWPORT_W)
            * 2.0f - 1.0f;

        float my =
            1.0f -
            (float)(mouseY / VIEWPORT_H)
            * 2.0f;

        // ====================================================
        // Triangle dragging
        // ====================================================

        bool mousePressed =
            glfwGetMouseButton(window,
                               GLFW_MOUSE_BUTTON_LEFT)
            == GLFW_PRESS;

        // Right button: attract particles (left is reserved for triangle drag).
        bool attractPressed =
            glfwGetMouseButton(window,
                                GLFW_MOUSE_BUTTON_RIGHT)
            == GLFW_PRESS;

        if (mousePressed && !draggingTriangle) {

            if (pointInTriangle(mx, my, gTriangle)) {

                draggingTriangle = true;

                dragOffsetX = mx - gTriangle.x1;
                dragOffsetY = my - gTriangle.y1;
            }
        }

        if (!mousePressed)
            draggingTriangle = false;

        if (draggingTriangle) {

            float dx =
                mx - dragOffsetX - gTriangle.x1;

            float dy =
                my - dragOffsetY - gTriangle.y1;

            gTriangle.x1 += dx;
            gTriangle.y1 += dy;

            gTriangle.x2 += dx;
            gTriangle.y2 += dy;

            gTriangle.x3 += dx;
            gTriangle.y3 += dy;
        }

        // ====================================================
        // Physics
        // ====================================================

        static unsigned int frameCount = 0;
        frameCount++;

        const float gravityY =
            floatSlowMo ? 0.0f : BASE_GRAVITY;

        const float simDt =
            DT * (floatSlowMo ? SLOW_MO_FACTOR : 1.0f);

        if (useGPU) {

            updateParticlesGPU(
                d_particles,
                currentN,
                simDt,
                gravityY,
                windX,
                spawnSpeed,
                frameCount);

            if (attractPressed)
                applyAttractionGPU(
                    d_particles,
                    currentN,
                    mx,
                    my,
                    simDt);

            collideParticlesWithTriangleGPU(
                d_particles,
                currentN,
                gTriangle.x1, gTriangle.y1,
                gTriangle.x2, gTriangle.y2,
                gTriangle.x3, gTriangle.y3);

            float* d_positions = nullptr;
            size_t size = 0;

            cudaGraphicsMapResources(
                1,
                &cudaVBOResource,
                0);

            cudaGraphicsResourceGetMappedPointer(
                (void**)&d_positions,
                &size,
                cudaVBOResource);

            extractPositionsGPU(
                d_particles,
                d_positions,
                currentN);

            cudaGraphicsUnmapResources(
                1,
                &cudaVBOResource,
                0);
        }
        else {

            updateParticlesCPU(
                h_particles,
                currentN,
                simDt,
                gravityY,
                windX,
                spawnSpeed,
                frameCount);

            if (attractPressed)
                applyAttractionCPU(
                    h_particles,
                    currentN,
                    mx,
                    my,
                    simDt);

            collideParticlesWithTriangleCPU(
                h_particles,
                currentN,
                gTriangle.x1, gTriangle.y1,
                gTriangle.x2, gTriangle.y2,
                gTriangle.x3, gTriangle.y3);

            for (int i = 0; i < currentN; i++) {

                h_positions[i * 2] =
                    h_particles[i].x;

                h_positions[i * 2 + 1] =
                    h_particles[i].y;
            }

            glBindBuffer(GL_ARRAY_BUFFER,
                         posVBO);

            glBufferSubData(GL_ARRAY_BUFFER,
                            0,
                            (long long)currentN *
                            2 * sizeof(float),
                            h_positions);
        }

        // ====================================================
        // Draw
        // ====================================================

        glClearColor(0.08f,
                     0.08f,
                     0.08f,
                     1.0f);

        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

        glBindVertexArray(VAO);

        glDrawArrays(GL_POINTS,
                     0,
                     currentN);

        // ====================================================
        // Draw triangle (VBO + shader; fixed-function removed in core)
        // ====================================================

        float triVerts[6] = {
            gTriangle.x1, gTriangle.y1,
            gTriangle.x2, gTriangle.y2,
            gTriangle.x3, gTriangle.y3
        };

        glBindBuffer(GL_ARRAY_BUFFER, triVBO);

        glBufferSubData(GL_ARRAY_BUFFER,
                        0,
                        sizeof(triVerts),
                        triVerts);

        glUseProgram(triProgram);

        glUniform3f(triColorLoc, 1.0f, 0.2f, 0.2f);

        glBindVertexArray(triVAO);

        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Title bar: mode, particle count, smoothed framerate (EMA of 1/dt)
        static auto prevLoopTime = std::chrono::steady_clock::now();
        static double fpsDisplay = 0.0;

        const auto loopTime = std::chrono::steady_clock::now();
        const double loopDt =
            std::chrono::duration<double>(loopTime - prevLoopTime).count();
        prevLoopTime = loopTime;

        if (loopDt > 1e-6) {
            const double instantFps = 1.0 / loopDt;
            fpsDisplay =
                (fpsDisplay < 1e-6)
                    ? instantFps
                    : (fpsDisplay * 0.9 + instantFps * 0.1);
        }

        char titleBuf[256];
        std::snprintf(
            titleBuf,
            sizeof(titleBuf),
            "Particle Sim | %s | N = %d | %.1f FPS | %s",
            useGPU ? "GPU" : "CPU",
            currentN,
            fpsDisplay,
            floatSlowMo ? "float+slo-mo" : "standard");

        glfwSetWindowTitle(window, titleBuf);
    }

    // ========================================================
    // Cleanup
    // ========================================================

    cudaGraphicsUnregisterResource(cudaVBOResource);

    glDeleteProgram(triProgram);
    glDeleteVertexArrays(1, &triVAO);
    glDeleteBuffers(1, &triVBO);

    freeParticlesGPU(d_particles);

    delete[] h_particles;
    delete[] h_colors;
    delete[] h_radii;
    delete[] h_positions;

    glfwTerminate();

    return 0;
}