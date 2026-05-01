# GPU-Accelerated Particle Simulation
CS4220 — GPU Computing

Real-time particle simulation comparing CPU and GPU performance, built with CUDA, OpenGL 4.6, and GLFW.

---

## Prerequisites

Make sure the following are installed before building:

| Tool | Version Used | Notes |
|------|-------------|-------|
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | 13.2 | Installs `nvcc`; verify with `nvcc --version` |
| [Visual Studio 2022](https://visualstudio.microsoft.com/) | Community or higher | Required by CUDA on Windows — install the **Desktop development with C++** workload |
| [CMake](https://cmake.org/download/) | 3.20+ | Verify with `cmake --version` |
| [Ninja](https://ninja-build.org/) | Any recent | Usually bundled with VS 2022; verify with `ninja --version` |
| GPU | NVIDIA RTX 4080 | Ada Lovelace, sm_89 — configured in `CMakeLists.txt` |

GLFW and GLAD are already bundled in the repo (`lib/glfw3.lib`, `include/`, `src/glad.c`). No separate installation needed.

---

## Building

All commands below should be run from a **Developer PowerShell for VS 2022** (or Developer Command Prompt), so that MSVC and Ninja are on your PATH. You can open one from the Start menu, or inside VS Code by selecting the MSVC kit in the CMake Tools extension.

### 1. Clone / open the project

```powershell
cd <project folder path>
```

### 2. Configure with CMake

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

This generates the Ninja build files inside the `build/` directory.

### 3. Compile

```powershell
cmake --build build
```

A successful build produces `build/particle_sim.exe`.

---

## Running

```powershell
.\build\particle_sim.exe
```

An 800×800 window will open showing **10,000 orange particles** falling under gravity and bouncing off the walls.

Frame time and FPS are printed to the terminal each frame:
```
Frame time: 2.34 ms | FPS: 427.35
```

---

## Controls

| Input | Action |
|-------|--------|
| **Left mouse button** (hold) | Attract all particles toward the cursor |
| **Close window / Alt+F4** | Exit the simulation |

---

## Project Structure

```
particle-sim/
├── src/
│   ├── main.cu            # Entry point: GLFW/OpenGL setup and render loop
│   ├── particle_cpu.cpp   # CPU simulation (sequential gravity + collision)
│   ├── particle_gpu.cu    # GPU simulation (CUDA kernel — in progress)
│   └── glad.c             # OpenGL function loader
├── include/
│   ├── particle.h         # Particle struct (x, y, vx, vy)
│   ├── particle_cpu.h     # CPU function declarations
│   ├── glad/              # GLAD headers
│   ├── GLFW/              # GLFW headers
│   └── KHR/               # Khronos platform header
├── lib/
│   └── glfw3.lib          # Precompiled GLFW static library (Windows x64)
├── build/                 # CMake/Ninja build output (generated)
└── CMakeLists.txt         # Build configuration
```

---

## Configuration

Key simulation parameters are defined at the top of `src/main.cu`:

| Constant | Default | Description |
|----------|---------|-------------|
| `N` | `10000` | Number of particles |
| `DT` | `0.016f` | Timestep per frame (~60 FPS target) |

Physics constants are in `src/particle_cpu.cpp`:

| Constant | Value | Description |
|----------|-------|-------------|
| `GRAVITY` | `-9.8f` | Downward acceleration |
| `BOUND_X / BOUND_Y` | `1.0f` | NDC boundary (window edge) |
| `RESTITUTION` | `0.8f` | Bounce damping factor |

---

## VS Code Setup

The project is configured for use with VS Code. Recommended extensions:

- **CMake Tools** — configure, build, and run directly from the status bar
- **C/C++** (Microsoft) — IntelliSense and debugging
- **CUDA C++** (NVIDIA Nsight) — syntax highlighting for `.cu` files

When CMake Tools prompts you to select a kit, choose **Visual Studio Community 2022 — amd64**.

To build and run without a terminal, use the CMake Tools **Build** button (bottom status bar) and then **Run** the generated `particle_sim.exe` from the `build/` folder.

---

## CUDA Architecture Note

`CMakeLists.txt` sets `CMAKE_CUDA_ARCHITECTURES 89` targeting the RTX 4080 (Ada Lovelace / sm_89). If you build on a different GPU, update this value to match your device's compute capability (e.g., `86` for RTX 30-series, `75` for RTX 20-series).
