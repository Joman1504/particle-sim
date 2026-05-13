/** src/particle_gpu.cu
 *  Logic for the GPU particle simulation.
 *  One CUDA thread handles one particle per kernel launch.
 */

#include "particle_gpu.h"
#include "particle.h"
#include <cstdlib>
#include <cuda_runtime.h>

// ---- Physics constants (must match particle_cpu.cpp) ---- //
#define BOUND_X      1.0f
#define BOUND_Y      1.0f
#define RESTITUTION  0.8f

#define BLOCK_SIZE   256


// ---- Kernel: physics update ---- //
// Each thread updates one particle: applies gravity and wind, integrates
// position, bounces off left/right/top walls, and respawns at the top
// when the particle exits through the bottom.
//
// Respawn uses a fast integer hash of (particle index, seed) to pick a
// pseudo-random X so the top edge fills evenly across frames.
__global__ void updateKernel(Particle* particles, int n, float dt,
                              float gravityY,
                              float windX, float spawnSpeed, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float r = particles[i].r;

    // Apply gravity and horizontal wind
    particles[i].vy += gravityY * dt;
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

    // Top wall bounce (prevents upward escape under strong reverse wind)
    if (particles[i].y + r > BOUND_Y) {
        particles[i].y  =  BOUND_Y - r;
        particles[i].vy *= -RESTITUTION;
    }

    // Bottom exit — respawn at top with a fresh random X position
    if (particles[i].y - r < -BOUND_Y) {
        // Integer hash of index + seed for cheap per-thread randomness
        unsigned int h = ((unsigned int)i + 1u) * 2654435761u ^ seed;
        h ^= h >> 16;
        float rx = ((float)(h & 0x00FFFFFFu) / (float)0x01000000u) * 2.0f - 1.0f;

        // Second hash for vertical jitter — prevents horizontal banding that
        // forms when every same-frame respawn lands at exactly y = BOUND_Y - r.
        unsigned int h2 = h * 2246822519u;
        h2 ^= h2 >> 13;
        float yJitter = ((float)(h2 & 0x00FFFFFFu) / (float)0x01000000u) * 0.4f;

        particles[i].x  = rx;
        particles[i].y  = BOUND_Y - r - yJitter;
        particles[i].vx = 0.0f;
        particles[i].vy = -spawnSpeed;
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


void updateParticlesGPU(Particle* d_particles, int n, float dt,
                        float gravityY,
                        float windX, float spawnSpeed, unsigned int seed) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    updateKernel<<<gridSize, BLOCK_SIZE>>>(d_particles, n, dt,
                                           gravityY,
                                           windX, spawnSpeed, seed);
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

__device__ __host__ float signDev(float x1, float y1,
                                  float x2, float y2,
                                  float x3, float y3) {
    return (x1 - x3) * (y2 - y3) -
           (x2 - x3) * (y1 - y3);
}

__device__ __host__ bool pointInTriangleDev(
    float px, float py,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3) {

    float d1 = signDev(px, py, x1, y1, x2, y2);
    float d2 = signDev(px, py, x2, y2, x3, y3);
    float d3 = signDev(px, py, x3, y3, x1, y1);

    bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(hasNeg && hasPos);
}

// Squared distance from P to closest point on segment AB; writes that closest point.
__device__ __host__ static float closestPointOnSegmentSq(
    float px, float py,
    float ax, float ay, float bx, float by,
    float* outSx, float* outSy) {

    float abx = bx - ax;
    float aby = by - ay;
    float apx = px - ax;
    float apy = py - ay;
    float ab2 = abx * abx + aby * aby;
    float t = 0.0f;
    if (ab2 > 1e-20f)
        t = fminf(1.0f, fmaxf(0.0f, (apx * abx + apy * aby) / ab2));

    float sx = ax + t * abx;
    float sy = ay + t * aby;
    *outSx = sx;
    *outSy = sy;

    float dx = px - sx;
    float dy = py - sy;
    return dx * dx + dy * dy;
}

// Closest point on the three edges of the triangle (boundary), and its squared distance.
__device__ __host__ static float closestPointOnTriangleBoundarySq(
    float px, float py,
    float x1, float y1, float x2, float y2, float x3, float y3,
    float* outCx, float* outCy) {

    float sx, sy;
    float bestSq = closestPointOnSegmentSq(px, py, x1, y1, x2, y2, &sx, &sy);
    *outCx = sx;
    *outCy = sy;

    float sx2, sy2;
    float d2 = closestPointOnSegmentSq(px, py, x2, y2, x3, y3, &sx2, &sy2);
    if (d2 < bestSq) {
        bestSq = d2;
        *outCx = sx2;
        *outCy = sy2;
    }

    float sx3, sy3;
    float d3 = closestPointOnSegmentSq(px, py, x3, y3, x1, y1, &sx3, &sy3);
    if (d3 < bestSq) {
        bestSq = d3;
        *outCx = sx3;
        *outCy = sy3;
    }

    return bestSq;
}

// Circle vs filled triangle: resolve penetration using closest boundary point,
// outward normal, and velocity reflection (matches wall restitution).
__device__ __host__ static void resolveParticleTriangleCollision(
    Particle* p,
    float x1, float y1, float x2, float y2, float x3, float y3) {

    const float eps = 3e-4f;
    const float r = p->r;
    float px = p->x;
    float py = p->y;

    float cx, cy;
    float distSq = closestPointOnTriangleBoundarySq(
        px, py, x1, y1, x2, y2, x3, y3, &cx, &cy);
    float dist = sqrtf(distSq);

    bool inside = pointInTriangleDev(px, py, x1, y1, x2, y2, x3, y3);

    if (!inside && dist >= r - 1e-9f)
        return;

    float nx, ny;
    if (dist < 1e-7f) {
        float gx = (x1 + x2 + x3) * (1.0f / 3.0f);
        float gy = (y1 + y2 + y3) * (1.0f / 3.0f);
        nx = px - gx;
        ny = py - gy;
        float nl = sqrtf(nx * nx + ny * ny);
        if (nl < 1e-8f)
            return;
        nx /= nl;
        ny /= nl;
        if (!inside) {
            nx = -nx;
            ny = -ny;
        }
    } else if (inside) {
        nx = (cx - px) / dist;
        ny = (cy - py) / dist;
    } else {
        nx = (px - cx) / dist;
        ny = (py - cy) / dist;
    }

    p->x = cx + nx * (r + eps);
    p->y = cy + ny * (r + eps);

    float vn = p->vx * nx + p->vy * ny;
    if (vn < 0.0f) {
        float j = -(1.0f + RESTITUTION) * vn;
        p->vx += j * nx;
        p->vy += j * ny;
    }
}

__global__ void triangleCollisionKernel(
    Particle* particles,
    int n,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    resolveParticleTriangleCollision(
        &particles[i], x1, y1, x2, y2, x3, y3);
}

void collideParticlesWithTriangleCPU(
    Particle* particles,
    int n,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3) {

    for (int i = 0; i < n; ++i)
        resolveParticleTriangleCollision(
            &particles[i], x1, y1, x2, y2, x3, y3);
}

void collideParticlesWithTriangleGPU(
    Particle* d_particles,
    int n,
    float x1, float y1,
    float x2, float y2,
    float x3, float y3) {

    int gridSize =
        (n + BLOCK_SIZE - 1) /
        BLOCK_SIZE;

    triangleCollisionKernel<<<gridSize,
                              BLOCK_SIZE>>>(
        d_particles,
        n,
        x1, y1,
        x2, y2,
        x3, y3);
}

void freeParticlesGPU(Particle* d_particles) {
    cudaFree(d_particles);
} // end freeParticlesGPU
