/* include/particle.h
    Header for the Particle struct and shared simulation constants.
*/
#pragma once

// ---- Particle radius range (NDC units) ---- //
// Used by init, boundary checks, and (upcoming) collision detection.
// Cell size for the uniform collision grid = 2 * R_MAX.
constexpr float R_MIN = 0.005f;
constexpr float R_MAX = 0.015f;

struct Particle {
    float x, y;   // position  (NDC)
    float vx, vy; // velocity  (NDC / second)
    float r;      // radius    (NDC) — assigned at init, never changes
};
