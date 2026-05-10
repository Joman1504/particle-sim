/** include/particle_gpu.h
 *  Header to expose functions in "src/particle_gpu.cu"
 */

#pragma once
#include "particle.h"

// Allocates a GPU particle array (sets *d_particles) and fills it with random state.
void initParticlesGPU(Particle** d_particles, int n);

// Runs one physics timestep on the GPU (gravity, integration, wall bouncing).
void updateParticlesGPU(Particle* d_particles, int n, float dt);

// Applies mouse attraction toward (mx, my) in NDC on the GPU.
void applyAttractionGPU(Particle* d_particles, int n, float mx, float my, float dt);

// Extracts (x, y) pairs from d_particles into a flat device float array d_positions.
// d_positions must be a device pointer with room for n * 2 floats.
void extractPositionsGPU(const Particle* d_particles, float* d_positions, int n);

// Frees GPU particle memory.
void freeParticlesGPU(Particle* d_particles);
