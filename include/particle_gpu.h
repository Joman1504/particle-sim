/** include/particle_gpu.h
 *  Header to expose functions in "src/particle_gpu.cu"
 */

#pragma once
#include "particle.h"

// Allocates a GPU particle array (sets *d_particles) and fills it with random state.
void initParticlesGPU(Particle** d_particles, int n);

// Runs one physics timestep on the GPU (gravity, wind, integration, wall
// bouncing, and bottom-exit respawn).
// windX     — horizontal force (NDC/s²), positive = rightward
// spawnSpeed — downward speed assigned on respawn (NDC/s)
// seed       — per-frame value used to randomise respawn X positions
void updateParticlesGPU(Particle* d_particles, int n, float dt,
                        float windX, float spawnSpeed, unsigned int seed);

// Applies mouse attraction toward (mx, my) in NDC on the GPU.
void applyAttractionGPU(Particle* d_particles, int n, float mx, float my, float dt);

// Extracts (x, y) pairs from d_particles into a flat device float array d_positions.
// d_positions must be a device pointer with room for n * 2 floats.
void extractPositionsGPU(const Particle* d_particles, float* d_positions, int n);

// Frees GPU particle memory.
void freeParticlesGPU(Particle* d_particles);

// Collides particles with a triangle defined by (x1, y1), (x2, y2), (x3, y3).
void collideParticlesWithTriangleGPU(
    Particle* d_particles,
    int n,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3);

// Same collision rules on the host (CPU path).
void collideParticlesWithTriangleCPU(
    Particle* particles,
    int n,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3);