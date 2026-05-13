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


// Updates particle physics: gravity, wind, integration, wall bouncing,
// and bottom-exit respawn.
void updateParticlesCPU(Particle* particles, int n, float dt,
                        float windX, float spawnSpeed, unsigned int seed) {
    for (int i = 0; i < n; i++) {
        const float r = particles[i].r;

        // Apply gravity and horizontal wind
        particles[i].vy += GRAVITY * dt;
        particles[i].vx += windX  * dt;

        // Integrate position
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;

        // Left / right wall bounce
        if (particles[i].x + r > BOUND_X) {
            particles[i].x  =  BOUND_X - r;
            particles[i].vx *= -RESTITUTION;
        }
        if (particles[i].x - r < -BOUND_X) {
            particles[i].x  = -BOUND_X + r;
            particles[i].vx *= -RESTITUTION;
        }

        // Top wall bounce
        if (particles[i].y + r > BOUND_Y) {
            particles[i].y  =  BOUND_Y - r;
            particles[i].vy *= -RESTITUTION;
        }

        // Bottom exit — respawn at top with a fresh random X position
        if (particles[i].y - r < -BOUND_Y) {
            unsigned int h = ((unsigned int)i + 1u) * 2654435761u ^ seed;
            h ^= h >> 16;
            float rx = ((float)(h & 0x00FFFFFFu) / (float)0x01000000u) * 2.0f - 1.0f;
            particles[i].x  = rx;
            particles[i].y  = BOUND_Y - r;
            particles[i].vx = 0.0f;
            particles[i].vy = -spawnSpeed;
        }
    }
} // end updateParticlesCPU
