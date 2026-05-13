# GPU-Accelerated Particle Simulation

**CS4220 — GPU Computing**

Real-time 2D particle simulation in OpenGL normalized device coordinates (NDC), with a **CPU reference path** and a **CUDA GPU path** you can toggle at runtime. Physics includes gravity, wall bounces, bottom respawn, **mouse attraction**, **circle–triangle collisions** with a draggable obstacle, and optional **zero-gravity** / **slow-motion** modes. Positions are pushed to the GPU through **CUDA–OpenGL buffer interop** (registered VBO) to avoid per-frame host copies when using the GPU.

---

## Hardware used (reference setup)

| Component | Model |
|-----------|--------|
| **CPU** | Intel **Core i7-13700K** |
| **GPU** | NVIDIA **GeForce RTX 4080** (Ada Lovelace, **sm_89**) |

Other machines: install matching CUDA drivers and, if you use a different GPU, set `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` to your device’s compute capability.

---

## Features

- **Dual backends:** sequential CPU update (`particle_cpu.cpp`) vs one-CUDA-thread-per-particle (`particle_gpu.cu`), switched with **G** (host state is synced when toggling).
- **Rendering:** OpenGL **4.6 core** — point sprites for particles (vertex + fragment shaders), separate **VBO + shader** for the triangle (core profile has no fixed-function `glBegin`/`glEnd`).
- **Interop:** `cudaGraphicsGLRegisterBuffer` on the dynamic position VBO; each frame the mapped pointer is filled on device (`extractPositionsGPU`).
- **Triangle collision:** closest point on triangle edges, separation along the contact normal, velocity reflection with the same restitution as walls (shared math on host and device).
- **HUD:** window title shows **CPU/GPU mode**, **particle count N**, **smoothed FPS**, gravity label, and simulation time scale.

---

## Prerequisites

| Tool | Version used | Notes |
|------|----------------|------|
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 13.2 (example) | Provides `nvcc` — check with `nvcc --version` |
| [Visual Studio 2022](https://visualstudio.microsoft.com/) | Community or higher | **Desktop development with C++** workload (MSVC required by CUDA on Windows) |
| [CMake](https://cmake.org/download/) | 3.20+ | `cmake --version` |
| [Ninja](https://ninja-build.org/) | Any recent | Optional but recommended; VS can generate Ninja |

**Bundled in repo:** GLFW (`lib/glfw3.lib`), GLAD (`src/glad.c`, `include/glad/`), headers under `include/`. No separate GLFW install needed on Windows x64.

---

## Building

Use a **Developer PowerShell for VS 2022** (or Developer Command Prompt) so **MSVC** and optionally **Ninja** are on your `PATH`.

```powershell
cd <path-to-particle-sim>
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Output: `build/particle_sim.exe`.

You can also use a **Visual Studio** generator instead of Ninja, for example:

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## Running

```powershell
.\build\Release\particle_sim.exe
```

If you built with Ninja into `build/` without a multi-config subfolder:

```powershell
.\build\particle_sim.exe
```

A **1920×1080** window opens with **10,000** colored point-sprite particles, a red **triangle obstacle**, and a live-updating **title bar** (mode, **N**, FPS, gravity / time-scale flags).

---

## Controls

| Input | Action |
|--------|--------|
| **G** | Toggle **CPU** ↔ **GPU** simulation (copies particle state across the toggle). |
| **+** / **=** | Increase particle count by **10,000** (clamped to `MAX_N`). |
| **-** | Decrease particle count by **10,000** (clamped to `MIN_N`). |
| **1** | Set particle count to **10,000** |
| **2** | Set particle count to **100,000** |
| **3** | Set particle count to **1,000,000** |
| **4** | Set particle count to **10,000,000** |
| **Left mouse** (hold) | **Drag the triangle** when the cursor starts inside it. |
| **Right mouse** (hold) | **Attract** particles toward the cursor (inverse-square style force). |
| **Z** | Toggle **zero gravity** (`gravityY = 0`). While on, **respawn / spawn speed** is set to **0** and restored when you turn zero-G off. |
| **M** | Toggle **slow motion** (simulation `dt` scaled by `SLOW_MO_FACTOR`). |
| **Close window** / **Alt+F4** | Exit. |

---

## Project structure

```
particle-sim/
├── CMakeLists.txt
├── poster.html              # Optional one-page print layout (course poster)
├── src/
│   ├── main.cu              # GLFW, GL 4.6, render loop, input, CUDA–GL interop
│   ├── particle_cpu.cpp     # CPU physics + attraction
│   ├── particle_gpu.cu      # CUDA kernels: update, attraction, triangle collision, position extract
│   └── glad.c               # OpenGL loader
├── include/
│   ├── particle.h           # Particle { x, y, vx, vy, r }; R_MIN / R_MAX
│   ├── particle_cpu.h
│   ├── particle_gpu.h
│   ├── glad/
│   ├── GLFW/
│   └── KHR/
├── lib/
│   └── glfw3.lib            # Prebuilt GLFW (Windows x64)
└── build/                   # CMake output (generated; not committed)
```

---

## Simulation model

- **Coordinates:** NDC-style box \([-1,1]\) horizontally and vertically (`BOUND_X` / `BOUND_Y` = 1 in `particle_cpu.cpp` / `particle_gpu.cu`).
- **Particle:** `Particle` in `particle.h` — position, velocity, fixed radius `r` in NDC (currently `R_MIN` = `R_MAX` = **0.0025f** for uniform point size).
- **Integrator:** semi-implicit Euler each frame: apply acceleration, then integrate position with the (possibly slow-mo scaled) timestep.
- **Walls:** Left, right, top, and bottom; bottom crossing **respawns** at the top with hashed random **x** and small **y** jitter (CPU and GPU use the same hash idea).
- **Gravity:** `BASE_GRAVITY` = **-9.8** (NDC/s² style scale) from `main.cu`, passed into updates as `gravityY` (**0** when zero-G is on).
- **Restitution:** **0.8** on wall bounces (CPU/GPU aligned).

---

## Configuration (`main.cu`)

| Symbol | Default | Role |
|--------|---------|------|
| `MAX_N` | `10_000_000` | Upper cap for particle count |
| `MIN_N` | `1000` | Lower cap |
| `STEP_N` | `10000` | Count change per +/- key |
| `DT` | `0.016f` | Nominal seconds per frame |
| `BASE_GRAVITY` | `-9.8f` | Downward acceleration when gravity is on |
| `SLOW_MO_FACTOR` | `0.25f` | `dt` multiplier when slow-mo is on |
| `VIEWPORT_W` / `H` | `1920` / `1080` | Window size used for mouse NDC mapping and point size |
| `spawnSpeed` | `0.5f` | Downward speed assigned on respawn (overridden while zero-G is active) |

CUDA block size for kernels is **256** (`BLOCK_SIZE` in `particle_gpu.cu`).

---

## CUDA architecture

`CMakeLists.txt` sets:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

for **sm_89** (RTX 4080 class). For other GPUs, change this to your architecture (e.g. `86` for many RTX 30-series, `75` for Turing, etc.).

---

## VS Code

Suggested extensions: **CMake Tools**, **C/C++**, **CUDA C++** (NVIDIA). Select kit **Visual Studio 2022 — amd64**, configure `build/`, then Build / Run `particle_sim.exe`.

---

## License / course use

This repository is maintained for **CS4220** coursework and demonstration.
