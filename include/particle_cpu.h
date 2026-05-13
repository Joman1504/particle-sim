/* include/particle_cpu.h
    Header to expose functions in "src/particle_cpu.c"
*/
#pragma once
#include "particle.h"

// Initializes an <n> number of particles
void initParticlesCPU(Particle* particles, int n);

// Updates particle physics: gravity, integration, wall bouncing,
// and bottom-exit respawn.
// gravityY   — vertical acceleration (NDC/s²); use -9.8 for Earth-like, 0 for none
// spawnSpeed — downward speed assigned on respawn (NDC/s)
// seed       — per-frame value used to randomise respawn X positions
void updateParticlesCPU(Particle* particles, int n, float dt,
                        float gravityY,
                        float spawnSpeed, unsigned int seed);

// Pulls each particle toward (mx, my) in NDC (same model as applyAttractionGPU).
void applyAttractionCPU(Particle* particles, int n,
                        float mx, float my, float dt);