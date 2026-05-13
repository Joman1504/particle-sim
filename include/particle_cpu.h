/* include/particle_cpu.h
    Header to expose functions in "src/particle_cpu.c"
*/
#pragma once
#include "particle.h"

// Initializes an <n> number of particles
void initParticlesCPU(Particle* particles, int n);

// Updates particle physics: gravity, wind, integration, wall bouncing,
// and bottom-exit respawn.
// windX      — horizontal force (NDC/s²), positive = rightward
// spawnSpeed — downward speed assigned on respawn (NDC/s)
// seed       — per-frame value used to randomise respawn X positions
void updateParticlesCPU(Particle* particles, int n, float dt,
                        float windX, float spawnSpeed, unsigned int seed);

// Pulls each particle toward (mx, my) in NDC (same model as applyAttractionGPU).
void applyAttractionCPU(Particle* particles, int n,
                        float mx, float my, float dt);