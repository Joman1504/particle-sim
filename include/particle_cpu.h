/* include/particle_cpu.h
    Header to expose functions in "src/particle_cpu.c"
*/
#pragma once
#include "particle.h"

// Initializes an <n> number of particles
void initParticlesCPU(Particle* particles, int n);

// Updates the status of the particles
void updateParticlesCPU(Particle* particles, int n, float dt);