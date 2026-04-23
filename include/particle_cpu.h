/* include/particle_cpu.h
    Header to expose functions in "src/particle_cpu.c"
*/
#pragma once
#include "particle.h"

void initParticlesCPU(Particle* particles, int n);
void updateParticlesCPU(Particle* particles, int n, float dt);