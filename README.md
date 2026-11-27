# Sliver

A lightweight 3D software rasterizer and `f32`-stable triangle geometry library written in pure Rust (`#![forbid(unsafe_code)]`).

## Workspace Layout

- `crates/sliver_math`: Zero-dependency geometry library providing a branchless chord/half-angle sliver triangle detector (`ChordSliver`) and basic 3D vector math (`Vec3`).
- `crates/sliver_engine`: CPU-only 3D software rasterizer (`960x540`, `1/z` depth buffer, near-plane clipping, backface culling, incremental edge stepping, hemispherical lighting, distance fog, raycasting, and a minimal `.obj` loader) paired with a fixed-timestep (`120 Hz`) sphere-based first-person controller and a decoupled camera presentation layer (`ViewBob`).

## Physics & Camera

- **Sphere controller (`Walker`)**: `120 Hz` fixed-timestep movement with horizontal inertia, sprint, coyote/ledge grace, static edge grip, and diagonal step-up probing perpendicular to blocking risers.
- **Jump & landing feel**: Brief pre-jump knee-compression windup (`50 ms`), natural ballistic arc, and critically damped vertical/pitch absorption scaled by real landing impact speed.
- **Decoupled camera (`ViewBob`)**: Stride-driven view-bob, non-harmonic organic variation, subtle head micro-rotation, step smoothing, and interpolation between physics ticks and render frames.

## `sliver_math` Benchmark (`f32`, `-C target-cpu=native`)

Tested across 2,000,000 triangles (best of 7 runs) for throughput and 200,000 triangles per threshold near the boundary for precision:

| Detector | SoA Throughput | Error @ 1.00 deg (±0.01%) | Error @ 0.05 deg (±1.00%) | Error @ 0.01 deg (±5.00%) |
|---|---|---|---|---|
| Law of Cosines (trig-free) | 0.62 ns/tri | 36.806% | 47.538% | 49.954% |
| `ChordSliver` (`sort3`) | 0.61 ns/tri | 0.000% | 0.000% | 0.000% |

## `sliver_engine` Baseline (`v0.1.0`, Software Backend)

- Resolution: `960x540` native (`Scale::X1`)
- Frame time: ~5.1–5.5 ms/frame on a single CPU thread (60 FPS cap)
- Memory footprint: ~5.1 MB RAM, 0% GPU
- Binary size: ~238 KB (release)

## Usage

```powershell
# Run workspace tests
cargo test --workspace

# Run sliver_math microbenchmark with native SIMD instructions
$env:RUSTFLAGS="-C target-cpu=native"; cargo run --release -p sliver_math --example bench; Remove-Item Env:\RUSTFLAGS

# Run the built-in test room
cargo run --release -p sliver_engine

# Load an external .obj mesh
cargo run --release -p sliver_engine -- path\to\model.obj