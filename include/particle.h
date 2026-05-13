/* include/particle.h
    Header for the Particle struct and shared simulation constants.
*/
#pragma once

// ---- Particle radius range (NDC units) ---- //
// Set R_MIN = R_MAX for uniform size, or use a range for variety.
constexpr float R_MIN = 0.0025f;
constexpr float R_MAX = 0.0025f;

struct Particle {
    float x, y;   // position  (NDC)
    float vx, vy; // velocity  (NDC / second)
    float r;      // radius    (NDC) — assigned at init, never changes
};
