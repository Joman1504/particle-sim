// src/main.cu
#include <iostream>
#include <chrono>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "particle.h"
#include "particle_cpu.h"


const int N = 1000; // The number of particles to simulate at once
const float DT = 0.016f; 

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
}

int main() {
    // ---------------- Init GLFW ----------------
    if (!glfwInit()) {
        std::cerr << "GLFW init failted\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, "Particle Sim", nullptr, nullptr);
    if (!window) {
        std::cerr << "Window creation failed\n"; 
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n";
        return -1;
    }

    glEnable(GL_PROGRAM_POINT_SIZE);

    // ---------------- Build shader program ---------------- //
    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    // ---------------- Create VBO + VAO ---------------- //
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Allocate VBO for N particles (x, y, per particle)
    glBufferData(GL_ARRAY_BUFFER, N * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // ---------------- Init particles ---------------- //
    Particle* particles = new Particle[N];
    initParticlesCPU(particles, N);

    // Temp buffer to extract positions for the VBO upload
    float* positions = new float[N*2];


    // Variables for mouse interaction
    double mouseX = 0, mouseY = 0;
    bool attracting = false;

    // ---------------- Render loop ---------------- //
    while (!glfwWindowShouldClose(window)) {
        auto frameStart = std::chrono::high_resolution_clock::now();

        // ---- Mouse interaction block ---- //
        // Get mouse input
        attracting = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        // Convert mouse position from screen space to NDC (-1 to 1)
        float mxNDC = (float)(mouseX / 800.0) * 2.0f - 1.0f;
        float myNDC = 1.0f - (float)(mouseY / 800.0) * 2.0f; // flip Y

        // Apply attraction force if mouse button is held
        if (attracting) {
            for (int i = 0; i < N; i++) {
                float dx = mxNDC - particles[i].x;
                float dy = myNDC - particles[i].y;
                float distSq = dx * dx + dy * dy + 0.0001f; // avoid div by zero
                float force = 4.0f / distSq; // stronger when closer
                particles[i].vx += force * dx * DT;
                particles[i].vy += force * dy * DT;
            } // end for
        } // end if
        // ---- End of mouse interaction block ---- //

        // Update simulation
        updateParticlesCPU(particles, N, DT);

        // Extract positions into flat float array for VBO
        for (int i = 0; i < N; i++) {
            positions[i * 2] = particles[i].x;
            positions[i * 2 + 1] = particles[i].y;
        } // end for

        // Upload to GPU for *rendering*
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, N * 2 * sizeof(float), positions);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Draw
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, N);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // FPS Counter
        auto frameEnd = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();
        std::cout << "Frame time: " << ms << " ms | FPS: " << 1000.0f / ms << "\n";
    } // end of while render loop

    // ---------------- Cleanup ----------------
    delete[] particles;
    delete[] positions;
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return 0;
} // end main

