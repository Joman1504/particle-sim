/** src/particle_cpu.cpp
 *  Logic for the CPU particle simulation.
 */

#include "particle.h"
#include <cstdlib>

const float GRAVITY     = -9.8f;
const float BOUND_X     = 1.0f; // boundary: -1.0 to 1.0 (OpenGL NDC)
const float BOUND_Y     = 1.0f;
const float RESTITUTION = 0.8f; // bounciness (1.0 = perfectly elastic)


// Initializes <n> particles with random positions, velocities, and radii.
void initParticlesCPU(Particle* particles, int n) {
    for (int i = 0; i < n; i++) {

        // Random position within bounds
        particles[i].x  = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        particles[i].y  = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        // Random initial velocity
        particles[i].vx = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        particles[i].vy = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;

        // Random radius in [R_MIN, R_MAX]
        particles[i].r  = R_MIN + ((float)rand() / RAND_MAX) * (R_MAX - R_MIN);
    }; // end for
}; // end initParticlesCPU


// Updates particle physics: gravity, integration, radius-aware wall bouncing.
void updateParticlesCPU(Particle* particles, int n, float dt) {
    for (int i = 0; i < n; i++) {
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
    }; // end for
}; // end updateParticlesCPU
