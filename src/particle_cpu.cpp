/** src/particle_cpu.cpp
 *  Logic for the CPU particle simulation
 */

#include "particle.h"
#include <cstdlib>

const float GRAVITY     = -9.8f;
const float BOUND_X     = 1.0f; // boundary: -1.0 to 1.0 (OpenGL NDC)
const float BOUND_Y     = 1.0f;
const float RESTITUTION = 0.8f; // bounciness (1.0 = perfectly elastic)


// Initializes an <n> number of particles
void initParticlesCPU(Particle* particles, int n) {
    for (int i = 0; i < n; i++) {

        // Random position within bounds
        particles[i].x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        particles[i].y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        // Random initial velocity
        particles[i].vx = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        particles[i].vy = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
    }; // end for
}; // end initParticlesCPU

// Updates the status of the particles
void updateParticlesCPU(Particle* particles, int n, float dt) {
    for (int i = 0; i < n; i++) {

        // Apply gravity
        particles[i].vy += GRAVITY * dt;

        // Update position
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;

        // Boundary collision - X axis
        if (particles[i].x > BOUND_X) {
            particles[i].x = BOUND_X;
            particles[i].vx *= -RESTITUTION;
        }
        if (particles[i].x < -BOUND_X) {
            particles[i].x = -BOUND_X;
            particles[i].vx *= -RESTITUTION;
        }

        // Boundary collision - Y axis
        if (particles[i].y > BOUND_Y) {
            particles[i].y = BOUND_Y;
            particles[i].vy *= -RESTITUTION;
        }
        if (particles[i].y < -BOUND_Y) {
            particles[i].y = -BOUND_Y;
            particles[i].vy *= -RESTITUTION;
        }
    }; // end for
}; // end updateParticlesCPU