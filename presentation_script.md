# CS4220 — GPU-Accelerated Particle Simulation
## Presentation Script

---

> **Formatting guide:**
> `[SCREEN]` — what to have visible on screen at that moment.
> `[ACTION]` — something to do live during the recording.
> Estimated total runtime: **~6–7 minutes**.

---

## SECTION 1 — Project Idea (~1 minute)

`[SCREEN: Running simulation at 1M particles, GPU mode]`

Hey, I'm [Name], and this is my CS4220 project — a real-time particle simulation that runs on both the CPU and GPU, built with CUDA and OpenGL.

So the idea came from a pretty simple problem. Particle simulations show up all over the place — games, physics engines, visual effects. But the moment you start scaling up the number of particles, a CPU just can't keep up. It has to update every single particle one by one in a loop, and that gets slow fast once you're in the hundreds of thousands or millions.

The thing is, each particle doesn't actually care about what any other particle is doing — they all update independently. So this is a perfect problem for the GPU, where you can throw thousands of threads at it and update all the particles at the same time. The whole point of this project was to build that out and show exactly how big the difference is side by side.

---

## SECTION 2 — Methods & Pipeline (~2 minutes)

`[SCREEN: Open particle_gpu.cu, scroll to updateKernel]`

Let me walk through what actually happens each frame.

Every particle stores its position, velocity, and radius. On the GPU, all of that lives in one big device buffer. Each frame, the first thing we do is run the **update kernel** — this is where the physics happens.

`[SCREEN: Highlight gravity and position integration lines in updateKernel]`

Gravity is straightforward: we just add a downward acceleration to each particle's vertical velocity every frame, then move it by multiplying that velocity by the timestep. That's the basic physics. What makes it fast is that every particle is doing this at the same time — one CUDA thread per particle.

`[SCREEN: Highlight the wall bounce checks]`

After moving, each thread checks if its particle went out of bounds. Left and right walls flip the horizontal velocity and snap the position back inside. Same for the top. The bottom is a little different — instead of bouncing, the particle respawns at the top with a randomized X position, so the screen stays full of particles continuously falling through.

`[SCREEN: Scroll to attractionKernel]`

If the user's holding the right mouse button, we run the **attraction kernel** on top of that. It just pulls each particle toward the cursor — the closer a particle is, the stronger the pull. Again, all particles handled in parallel.

`[SCREEN: Scroll to triangleCollisionKernel]`

Then there's the **triangle collision kernel**. For each particle, we check whether it's overlapping the triangle obstacle on screen. If it is, we figure out the closest point on the triangle's edge, push the particle back out, and bounce its velocity off the surface — same bounciness as the walls. We'll go a bit deeper on this in the code section.

`[SCREEN: Switch to main.cu, highlight the cudaGraphicsMapResources block]`

The last step is getting the positions into OpenGL so we can actually draw them. On the GPU path, we do something called **CUDA–OpenGL buffer interop** — we registered the position buffer that OpenGL draws from directly with CUDA at startup, so each frame we just map it, write all the positions into it straight from the GPU, and unmap it. OpenGL draws right from there, no copying back to the CPU involved. On the CPU path we don't have that option, so we manually copy positions into a staging buffer and upload them to OpenGL every frame — which adds overhead on top of an already slower update loop.

---

## SECTION 3 — Implementation & Code (~2 minutes)

`[SCREEN: Open particle_gpu.cu, top of updateKernel]`

Let me highlight a few things in the code that are worth understanding.

The first thing every kernel does is figure out which particle it's responsible for:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n) return;
```

Each thread gets a unique index `i` from its block and thread position. The guard at the end is just making sure we don't go out of bounds if N isn't a perfect multiple of our block size — which is 256. From there, the thread only ever touches `particles[i]`, so there's zero communication between threads. That's why it scales so cleanly.

`[SCREEN: Highlight the respawn hash block]`

The respawn logic is a fun one. CUDA kernels can't call `rand()` — that's a CPU thing. So instead, we generate randomness by hashing the particle's index together with a per-frame seed that gets passed in from the host each frame:

```cpp
unsigned int h = ((unsigned int)i + 1u) * 2654435761u ^ seed;
h ^= h >> 16;
float rx = ((float)(h & 0x00FFFFFFu) / (float)0x01000000u) * 2.0f - 1.0f;
```

It's a standard integer hash — cheap, no memory access, and each thread gets a different value because each has a different index. We actually run a second hash on top of that to get a vertical jitter too, so particles don't all respawn at exactly the same Y and create visible horizontal bands.

`[SCREEN: Scroll to resolveParticleTriangleCollision]`

The triangle collision is the most involved part. When a particle overlaps the triangle, we find the closest point on each of its three edges, take whichever is nearest, build a normal pointing away from the surface, push the particle out to just outside the edge, then reflect its velocity:

```cpp
float vn = p->vx * nx + p->vy * ny;
if (vn < 0.0f) {
    float j = -(1.0f + RESTITUTION) * vn;
    p->vx += j * nx;
    p->vy += j * ny;
}
```

The `vn < 0` check makes sure we only apply the bounce if the particle is actually moving into the surface, not away from it. Notice the function has the `__device__ __host__` qualifier — that tells NVCC to compile it for both GPU and CPU, so the exact same collision math runs in both paths without duplicating any code.

`[SCREEN: Switch to main.cu, highlight interop block]`

And the interop extraction, just to show it clearly:

```cpp
cudaGraphicsMapResources(1, &cudaVBOResource, 0);
cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &size, cudaVBOResource);
extractPositionsKernel<<<gridSize, BLOCK_SIZE>>>(d_particles, d_positions, currentN);
cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);
```

Map the buffer, get a device pointer to it, write positions in parallel, unmap. When we unmap, CUDA guarantees the kernel is done before OpenGL touches it. That's the whole thing — no CPU copy, no round trip across the bus.

---

## SECTION 4 — Results & Timings (~1 minute)

`[SCREEN: Show poster results table]`

Here are the framerates we measured on our machine — i7-13700K and RTX 4080. We read these straight from the window title bar after setting N with keys 1 through 4.

| N Particles | CPU FPS | GPU FPS |
|-------------|---------|---------|
| 10,000      | 144     | 144     |
| 100,000     | 130     | 144     |
| 1,000,000   | 13.3    | 144     |
| 10,000,000  | 1.4     | 144     |

At 10k both are capped at 144 — the work is too small to tell apart. At 100k the CPU starts dropping. At 1 million it's down to 13 FPS, which is pretty much unusable, while the GPU is still sitting at the display cap. At 10 million the CPU is basically a slideshow at 1.4 FPS, and the GPU hasn't moved.

That's over a 100x speedup at 1 million particles, and it only gets wider from there. The CPU scales linearly with N like you'd expect from a sequential loop. The GPU stays flat because the RTX 4080 has enough parallel cores to chew through even 10 million particles well within a single frame budget.

---

## SECTION 5 — Demo (~1.5 minutes)

`[SCREEN: Full simulation window, GPU mode, 1M particles]`

Alright, let's see it running. We're on GPU mode with a million particles right now — 144 FPS in the title bar.

`[ACTION: Press G to switch to CPU]`

Hitting G switches to CPU. You can see the frame rate drop right away — we're at about 13 FPS now with the same particle count. It's noticeably sluggish. Switching back to GPU recovers instantly.

`[ACTION: Press 4 to jump to 10M, stay on GPU]`

Key 4 jumps to 10 million particles. Still smooth on the GPU. Now let's flip to CPU at this count.

`[ACTION: Press G]`

Yeah, 1.4 FPS — basically frozen. That's the gap we're talking about. Switching back to GPU.

`[ACTION: Press 3 to return to 1M, GPU mode]`

Back to 1 million. Let's show the interactive features. Holding the right mouse button attracts particles toward the cursor.

`[ACTION: Hold right mouse and sweep cursor around]`

You can drag the triangle around with the left mouse button — particles collide with it and bounce off the edges.

`[ACTION: Drag the triangle]`

Z toggles zero gravity, so particles just float freely. M is slow motion — it scales the timestep down to 25%, which is useful for watching collisions up close.

`[ACTION: Press Z, then M to demonstrate, then toggle both off]`

And pressing them again brings everything back to normal.

---

## SECTION 6 — Closing (~20 seconds)

`[SCREEN: Simulation running, GPU mode, 1M particles]`

So to wrap up — particle simulation is one of those problems that maps really naturally onto GPU hardware, since every particle is independent. Pairing CUDA with OpenGL buffer interop means the GPU handles both the physics and the rendering side without the CPU getting in the way. The result is a simulation that stays interactive at scales where the CPU just can't compete. Thanks for watching.

---

*End of script.*
