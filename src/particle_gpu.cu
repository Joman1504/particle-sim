/** src/particle_gpu.cu
 *  Logic for the GPU particle simulation.
 *  One CUDA thread handles one particle per kernel launch.
 */

#include "particle_gpu.h"
#include "particle.h"
#include <cstdlib>
#include <cuda_runtime.h>

// ---- Physics constants (must match particle_cpu.cpp) ---- //
#define GRAVITY     -9.8f
#define BOUND_X      1.0f
#define BOUND_Y      1.0f
#define RESTITUTION  0.8f

#define BLOCK_SIZE   256


// ---- Kernel: physics update ---- //
// Each thread updates one particle: applies gravity, integrates position,
// and bounces off the four walls using the particle's own radius.
__global__ void updateKernel(Particle* particles, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float r = particles[i].r;

    // Apply gravity
    particles[i].vy += GRAVITY * dt;

    // Integrate position
    particles[i].x += particles[i].vx * dt;
    particles[i].y += particles[i].vy * dt;

    // Boundary collision - X axis (wall at ± BOUND_X, particle edge at x ± r)
    if (particles[i].x + r > BOUND_X) {
        particles[i].x = BOUND_X - r;
        particles[i].vx *= -RESTITUTION;
    }
    if (particles[i].x - r < -BOUND_X) {
        particles[i].x = -BOUND_X + r;
        particles[i].vx *= -RESTITUTION;
    }

    // Boundary collision - Y axis
    if (particles[i].y + r > BOUND_Y) {
        particles[i].y = BOUND_Y - r;
        particles[i].vy *= -RESTITUTION;
    }
    if (particles[i].y - r < -BOUND_Y) {
        particles[i].y = -BOUND_Y + r;
        particles[i].vy *= -RESTITUTION;
    }
} // end updateKernel


// ---- Kernel: mouse attraction ---- //
// Each thread applies an attraction force toward (mx, my) in NDC for one particle.
__global__ void attractionKernel(Particle* particles, int n,
                                float mx, float my, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float dx     = mx - particles[i].x;
    float dy     = my - particles[i].y;
    float distSq = dx * dx + dy * dy + 0.0001f; // avoid div by zero
    float force  = 4.0f / distSq;               // stronger when closer

    particles[i].vx += force * dx * dt;
    particles[i].vy += force * dy * dt;
} // end attractionKernel


// ---- Kernel: position extraction ---- //
// Writes (x, y) from each particle into a flat float array (d_positions),
// suitable for direct upload into an OpenGL VBO.
__global__ void extractPositionsKernel(const Particle* particles,
                                        float* positions, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    positions[i * 2]     = particles[i].x;
    positions[i * 2 + 1] = particles[i].y;
} // end extractPositionsKernel


// ---- Host wrappers ---- //

// Allocates device memory for n particles, fills with random state
// (including radius), and uploads to the device.
// Note: main immediately overwrites slots [0, currentN) with the host
// particle state via cudaMemcpy, so the random data here only persists
// for the uninitialised tail [currentN, n).
void initParticlesGPU(Particle** d_particles, int n) {
    Particle* h_temp = new Particle[n];
    for (int i = 0; i < n; i++) {
        h_temp[i].x  = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_temp[i].y  = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_temp[i].vx = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        h_temp[i].vy = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        h_temp[i].r  = R_MIN + ((float)rand() / RAND_MAX) * (R_MAX - R_MIN);
    }

    cudaMalloc(d_particles, n * sizeof(Particle));
    cudaMemcpy(*d_particles, h_temp, n * sizeof(Particle), cudaMemcpyHostToDevice);

    delete[] h_temp;
} // end initParticlesGPU


void updateParticlesGPU(Particle* d_particles, int n, float dt) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    updateKernel<<<gridSize, BLOCK_SIZE>>>(d_particles, n, dt);
} // end updateParticlesGPU


void applyAttractionGPU(Particle* d_particles, int n,
                        float mx, float my, float dt) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    attractionKernel<<<gridSize, BLOCK_SIZE>>>(d_particles, n, mx, my, dt);
} // end applyAttractionGPU


void extractPositionsGPU(const Particle* d_particles, float* d_positions, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    extractPositionsKernel<<<gridSize, BLOCK_SIZE>>>(d_particles, d_positions, n);
} // end extractPositionsGPU


void freeParticlesGPU(Particle* d_particles) {
    cudaFree(d_particles);
} // end freeParticlesGPU
