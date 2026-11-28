# Sliver

A from-scratch 3D engine in pure Rust: an `f32`-stable triangle geometry library, a multithreaded CPU software rasterizer, a sphere-based first-person physics controller, a Doom `.WAD` level loader, Cardboard-style stereo rendering and an Android frontend.

The math and the engine logic have no dependencies. The desktop window and input use one optional crate (`minifb`); the Android frontend uses the platform bindings it needs. No GPU is involved.

## Highlights

- **Branchless sliver detector** (`ChordSliver`): flags triangles whose smallest angle is below a threshold with 0% error in `f32`, where the Law of Cosines fails near 0°.
- **Software rasterizer**: perspective-correct textured triangles, `1/z` depth buffer, near-plane clipping, fog, Doom-style panoramic sky, parallel bands on a persistent thread pool.
- **Doom `.WAD` loader**: maps, textures and flats converted to triangles (zero dependencies).
- **First-person physics at 120 Hz**: sphere vs. triangles with substeps, surface response, steps, jump windup, sprint.
- **Camera layer**: view-bob, head micro-rotation, step smoothing and interpolation between physics ticks.
- **Stereo (side-by-side) with Cardboard lens warp**, adjustable IPD.
- **Headless engine core**, so the same code runs on desktop and Android.
- **Built-in profiler** with an on-screen HUD.

## Workspace layout

| Crate | Role |
|---|---|
| `crates/sliver_math` | Zero-dependency geometry: `Vec3`, `ChordSliver` (and `CosineSliver` as a benchmark reference), `ray_triangle`, `closest_point_on_triangle`, `sphere_triangle`. |
| `crates/sliver_engine` | The engine library plus the desktop binary (feature `desktop`, enabled by default). |
| `crates/sliver_android` | `cdylib` frontend based on `NativeActivity`. It compiles only for Android targets. |

Modules of `sliver_engine`:

| Module | Contents |
|---|---|
| `engine` | Headless core: `Engine`, `World`, `InputState`, `FrameReport`. The host feeds input and presents `framebuffer()`. |
| `raster` | Camera, framebuffer, clipping, texturing, sky, fog, band-parallel rasterization. |
| `collision`, `motion` | Generic engine pieces: swept sphere movement with per-plane contact telemetry, and parametric surface response. They know nothing about players. |
| `controller`, `view_bob` | Game-side policy: `Walker` (gravity, jump, steps, sprint) and the camera presentation layer. |
| `raycast`, `scene` | Ray queries, demo room, test arena, `.obj` loader, sliver tagging. |
| `wad` | Doom map and texture loader. |
| `stereo` | Side-by-side renderer with a cached inverse barrel-distortion map. |
| `perf` | Zero-allocation rolling frame profiler and HUD. |

## Quick start

Requires Rust 1.85 or newer (edition 2024).

```powershell
# Run the tests
cargo test --workspace

# Built-in demo room
cargo run --release -p sliver_engine

# External .obj mesh
cargo run --release -p sliver_engine -- path\to\model.obj

# Doom map (the map name is optional and defaults to the first map)
cargo run --release -p sliver_engine -- path\to\DOOM.WAD E1M1

# Options: --2x (double-size window), --sbs (start in stereo)
cargo run --release -p sliver_engine -- --2x --sbs

# sliver_math microbenchmark with native SIMD instructions
$env:RUSTFLAGS="-C target-cpu=native"; cargo run --release -p sliver_math --example bench; Remove-Item Env:\RUSTFLAGS
```

## Controls

Desktop:

| Input | Action |
|---|---|
| `W` `A` `S` `D` | Move |
| `Shift` | Sprint |
| `Space` | Jump |
| Mouse drag (left or right button), arrows, `Q` / `E` | Look |
| `F1` | Sliver debug colors |
| `F2` | Cycle sliver threshold (1°, 5°, 12°, 25°) |
| `F3` | Fog on/off |
| `F4` | Raycast from the screen center (prints the hit) |
| `F5` | Switch between the demo room and the 80 × 80 m test arena |
| `F6` | Slow motion (0.2x physics) |
| `F7` | Textures on/off |
| `F8` | Side-by-side stereo on/off |
| `F10` | Cycle lens warp: flat / Cardboard |
| `[` `]` | Decrease / increase IPD |
| `F9` | Performance HUD |
| `F11` | Toggle 2x window |
| `Esc` | Quit |

Android: left half of the screen is a virtual stick, right half drags to look, the bottom-right corner jumps, and the top strip toggles the performance HUD, lens warp and stereo.

## Rendering

- CPU only, `0x00RRGGBB` framebuffer. Native `960x540` on desktop; Android renders at up to 480 lines and scales to the window.
- Incremental edge stepping with a `1/z` depth buffer, near-plane clipping and backface culling.
- Perspective-correct texture mapping for WAD textures and flats.
- Hemispherical plus directional lighting, distance fog, panoramic sky, and Doom's 1.2x vertical stretch applied to WAD map heights.
- The screen is split into horizontal bands rendered on a persistent worker pool (`BandPool`), using up to 8 threads.
- Sliver debug mode highlights every triangle under the selected minimum angle, using `ChordSliver`.
- Stereo renders each eye at half width and applies an optional Brown–Conrady barrel warp through a precomputed lookup table.

## Physics and camera

- **Sphere controller (`Walker`)**: radius `0.35 m`, eye height `1.65 m`, fixed `120 Hz` timestep. Walk `5.5 m/s`, sprint `9.5 m/s`, separate acceleration (`18`) and deceleration (`40`), low air control, gravity `10.5`, jump speed `3.5 m/s` after a `50 ms` knee-compression windup.
- **Collision**: movement is split into substeps no longer than half the radius, with iterative push-out from the deepest contact. Each move returns per-plane telemetry (normal, impact speed, tangent velocity, fresh or sustained).
- **Surface response**: Coulomb friction with a pressure threshold and a capped bounce above an impact threshold. Walkable floor is frictionless and never bounces, and step risers never bounce.
- **Steps**: step-up probing perpendicular to blocking risers (diagonal approaches work), step-down snapping, ledge grace time and a static grip so the body does not slide off an edge when standing still. Maximum step height is `0.35 m`.
- **Camera (`ViewBob`)**: stride-driven bob with a longer stride when sprinting, non-harmonic variation, head micro-rotation, step smoothing, step-up stumble, jump windup, landing absorption and acceleration pitch. The renderer interpolates the body and the camera effects between the last two physics ticks.

## `sliver_math` benchmark (`f32`, `-C target-cpu=native`)

Tested across 2,000,000 triangles (best of 7 runs) for throughput and 200,000 triangles per threshold near the boundary for precision:

| Detector | SoA Throughput | Error @ 1.00 deg (±0.01%) | Error @ 0.05 deg (±1.00%) | Error @ 0.01 deg (±5.00%) |
|---|---|---|---|---|
| Law of Cosines (trig-free) | 0.62 ns/tri | 36.806% | 47.538% | 49.954% |
| `ChordSliver` (`sort3`) | 0.61 ns/tri | 0.000% | 0.000% | 0.000% |

Why it works: with sides sorted as `min <= mid <= max` and `d = max - mid`, the Law of Cosines gives `min² - d² = 4·mid·max·sin²(θ/2)` for the smallest angle `θ`. The detector compares `(min - d)(min + d)` against `s0²·mid·max`, where `s0 = 2·sin(θ0/2)` is computed once from the threshold. That avoids the cancellation of `cos θ` near 0°, and needs no `sqrt` or trigonometry per triangle.

## Measurements

Headless engine core (`Engine::step`), built-in demo room, release build with LTO, on a single-core Linux VM (so the band pool is not exercised in parallel):

| Mode | Median render | p99 |
|---|---|---|
| Mono `960x540` | 1.50 ms | 2.66 ms |
| Mono `480x270` | 0.38 ms | 0.78 ms |
| Stereo `960x540` | 2.15 ms | 3.77 ms |
| Stereo `640x360` | 0.85 ms | 1.24 ms |

The release desktop binary is about 1.2 MB on Linux x86-64. For real hardware, use the `F9` HUD, which breaks each frame into physics, clear, rasterization and presentation time.

## Android

`sliver_android` wraps the engine in a `NativeActivity`: it blits the framebuffer to the `ANativeWindow`, maps touch input to `InputState`, and supports the same stereo and HUD toggles. It looks for `DOOM.WAD` in the app assets (`crates/sliver_android/assets/`, git-ignored) and loads `E1M1`; without it, it falls back to the demo room.

The package metadata targets `aarch64-linux-android` (min SDK 24, target SDK 34). This frontend is experimental and needs the Android SDK and NDK. The development signing configuration expects a local `dev.keystore`, which is git-ignored; do not use its development password for a real release.

## Quality checks

```powershell
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --check
```

`sliver_math` and `sliver_engine` are `#![forbid(unsafe_code)]`. The only `unsafe` code lives in the Android frontend, in two small FFI blocks with `SAFETY` comments.

## Design principles

- Write the math and the logic in-house; depend on external crates only at the hardware and OS boundary (window, input, platform APIs).
- Keep the engine and the game separate: `collision`, `motion` and `raster` are generic, while `controller` and `view_bob` are game-side policy built on top of them.
- Run physics at a fixed timestep and interpolate for rendering.

## Assets

No game assets are included. To load Doom maps, use a `.WAD` you are allowed to use (for example a freely distributable one such as Freedoom, or one you own). `*.wad` files are git-ignored.
