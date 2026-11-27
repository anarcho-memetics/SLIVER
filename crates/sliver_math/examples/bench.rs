#![forbid(unsafe_code)]

use sliver_math::{ChordSliver, CosineSliver};
use std::hint::black_box;
use std::time::Instant;

const SPEED_N: usize = 2_000_000;
const RUNS: usize = 7;
const PREC_N: usize = 200_000;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Returns a uniform f64 in [0.0, 1.0).
    #[inline(always)]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Generates a valid f32 triangle whose minimum interior angle is `min_angle_deg`,
/// with randomized scale, slight asymmetry, and random side permutation.
fn make_triangle(rng: &mut Rng, min_angle_deg: f64) -> [f32; 3] {
    let theta = min_angle_deg.to_radians();
    let sin_half = (theta * 0.5).sin();

    // Scale across several orders of magnitude
    let u = 0.5 + rng.next_f64() * 50.0;
    let c_iso = 2.0 * u * sin_half;

    // Keep v < 2*u*cos(theta) so `c` stays strictly the shortest side,
    // and cap (v - u) relative to `c_iso` so f32 quantization does not collapse the triangle.
    let delta_max = (0.35 * c_iso)
        .min(0.8 * u * (2.0 * theta.cos() - 1.0))
        .max(0.0);
    let v = u + rng.next_f64() * delta_max;

    // Stable f64 chord formula for the side opposite `theta`
    let dv = v - u;
    let c = (dv * dv + 4.0 * u * v * sin_half * sin_half).sqrt();

    let s = [u as f32, v as f32, c as f32];
    match rng.next_u64() % 6 {
        0 => [s[0], s[1], s[2]],
        1 => [s[0], s[2], s[1]],
        2 => [s[1], s[0], s[2]],
        3 => [s[1], s[2], s[0]],
        4 => [s[2], s[0], s[1]],
        _ => [s[2], s[1], s[0]],
    }
}

/// Ground-truth sliver check in f64 on the exact quantized f32 side lengths.
fn is_sliver_exact_f64(tri: [f32; 3], limit_deg: f64) -> bool {
    let mut s = [tri[0] as f64, tri[1] as f64, tri[2] as f64];
    s.sort_by(|x, y| x.total_cmp(y));
    let (min_s, mid_s, max_s) = (s[0], s[1], s[2]);
    let d = max_s - mid_s;
    let s0 = 2.0 * (limit_deg.to_radians() * 0.5).sin();
    (min_s - d) * (min_s + d) < (s0 * s0) * (mid_s * max_s)
}

#[inline(never)]
fn bench_aos<F: Fn(f32, f32, f32) -> bool>(tris: &[[f32; 3]], f: F) -> (f64, u32) {
    let mut best_ns = f64::INFINITY;
    let mut last_count = 0u32;

    for _ in 0..RUNS {
        let t0 = Instant::now();
        let mut count = 0u32;
        for &[a, b, c] in black_box(tris) {
            count += f(a, b, c) as u32;
        }
        let dt = t0.elapsed().as_nanos() as f64 / tris.len() as f64;
        last_count = black_box(count);
        if dt < best_ns {
            best_ns = dt;
        }
    }
    (best_ns, last_count)
}

#[inline(never)]
fn bench_soa<F: Fn(f32, f32, f32) -> bool>(sa: &[f32], sb: &[f32], sc: &[f32], f: F) -> (f64, u32) {
    let n = sa.len().min(sb.len()).min(sc.len());
    let sa = &sa[..n];
    let sb = &sb[..n];
    let sc = &sc[..n];

    let mut best_ns = f64::INFINITY;
    let mut last_count = 0u32;

    for _ in 0..RUNS {
        let t0 = Instant::now();
        let mut count = 0u32;
        let a_ref = black_box(sa);
        let b_ref = black_box(sb);
        let c_ref = black_box(sc);
        for i in 0..n {
            count += f(a_ref[i], b_ref[i], c_ref[i]) as u32;
        }
        let dt = t0.elapsed().as_nanos() as f64 / n as f64;
        last_count = black_box(count);
        if dt < best_ns {
            best_ns = dt;
        }
    }
    (best_ns, last_count)
}

fn run_speed_bench() {
    let limit_deg = 12.0_f32;
    let cos_det = CosineSliver::from_degrees(limit_deg);
    let chord_det = ChordSliver::from_degrees(limit_deg);

    let mut rng = Rng::new(0x00C0_FFEE_1234_5678);
    let mut aos = Vec::with_capacity(SPEED_N);
    let mut sa = Vec::with_capacity(SPEED_N);
    let mut sb = Vec::with_capacity(SPEED_N);
    let mut sc = Vec::with_capacity(SPEED_N);

    for i in 0..SPEED_N {
        // Half below threshold (slivers), half above threshold (non-slivers)
        let angle = if i & 1 == 0 {
            0.25 + rng.next_f64() * 11.5
        } else {
            12.25 + rng.next_f64() * 45.0
        };
        let t = make_triangle(&mut rng, angle);
        aos.push(t);
        sa.push(t[0]);
        sb.push(t[1]);
        sc.push(t[2]);
    }

    println!("=== THROUGHPUT ({SPEED_N} triangles, best of {RUNS} runs, f32) ===");
    println!("-- AoS Layout (&[[f32; 3]]) --");

    let (ns, c) = bench_aos(&aos, |a, b, c| cos_det.is_sliver_min_vertex(a, b, c));
    println!("Cosine min_vertex (if):       {ns:>4.2} ns/tri  (slivers: {c})");

    let (ns, c) = bench_aos(&aos, |a, b, c| chord_det.is_sliver_min_vertex(a, b, c));
    println!("Chord  min_vertex (if):       {ns:>4.2} ns/tri  (slivers: {c})");

    let (ns, c) = bench_aos(&aos, |a, b, c| cos_det.is_sliver_branchless(a, b, c));
    println!("Cosine 3-vert (branchless):   {ns:>4.2} ns/tri  (slivers: {c})");

    let (ns, c) = bench_aos(&aos, |a, b, c| chord_det.is_sliver_branchless(a, b, c));
    println!("Chord  3-vert (branchless):   {ns:>4.2} ns/tri  (slivers: {c})");

    let (ns, c) = bench_aos(&aos, |a, b, c| cos_det.is_sliver_sort3(a, b, c));
    println!("Cosine sort3  (branchless):   {ns:>4.2} ns/tri  (slivers: {c})");

    let (ns, c) = bench_aos(&aos, |a, b, c| chord_det.is_sliver_sort3(a, b, c));
    println!("Chord  sort3  (branchless):   {ns:>4.2} ns/tri  (slivers: {c})");

    println!();
    println!("-- SoA Layout (&[f32] x 3, optimal for SIMD auto-vectorization) --");

    let (ns, c) = bench_soa(&sa, &sb, &sc, |a, b, c| {
        cos_det.is_sliver_branchless(a, b, c)
    });
    println!("Cosine 3-vert SoA:            {ns:>4.2} ns/tri  (slivers: {c})");

    let (ns, c) = bench_soa(&sa, &sb, &sc, |a, b, c| {
        chord_det.is_sliver_branchless(a, b, c)
    });
    println!("Chord  3-vert SoA:            {ns:>4.2} ns/tri  (slivers: {c})");

    let (ns, c) = bench_soa(&sa, &sb, &sc, |a, b, c| chord_det.is_sliver_sort3(a, b, c));
    println!("Chord  sort3  SoA:            {ns:>4.2} ns/tri  (slivers: {c})");
}

fn run_precision_bench() {
    let cases: [(f64, f64); 4] = [
        (12.0, 0.0001), // ±0.01%
        (1.0, 0.0001),  // ±0.01%
        (0.05, 0.01),   // ±1.00%
        (0.01, 0.05),   // ±5.00%
    ];

    println!();
    println!("=== f32 PRECISION NEAR THRESHOLD ({PREC_N} triangles each) ===");
    println!("Limit    | Band     | Cosine Err      | Chord 3v Err    | Chord sort3 Err");
    println!("-------------------------------------------------------------------------");

    let mut rng = Rng::new(0xBAD_5EED_9999);

    for &(limit_deg, band) in &cases {
        let cos_det = CosineSliver::from_degrees(limit_deg as f32);
        let chord_det = ChordSliver::from_degrees(limit_deg as f32);

        let mut err_cos = 0usize;
        let mut err_c3 = 0usize;
        let mut err_cs = 0usize;

        for i in 0..PREC_N {
            let angle = if i & 1 == 0 {
                limit_deg * (1.0 - band)
            } else {
                limit_deg * (1.0 + band)
            };
            let t = make_triangle(&mut rng, angle);
            let truth = is_sliver_exact_f64(t, limit_deg);

            if cos_det.is_sliver_sort3(t[0], t[1], t[2]) != truth {
                err_cos += 1;
            }
            if chord_det.is_sliver_branchless(t[0], t[1], t[2]) != truth {
                err_c3 += 1;
            }
            if chord_det.is_sliver_sort3(t[0], t[1], t[2]) != truth {
                err_cs += 1;
            }
        }

        let pct = |e: usize| 100.0 * (e as f64) / (PREC_N as f64);
        println!(
            "{limit_deg:>6.2} ° | ±{:>5.2}% | {:>14.3}% | {:>14.3}% | {:>14.3}%",
            band * 100.0,
            pct(err_cos),
            pct(err_c3),
            pct(err_cs),
        );
    }
}

fn main() {
    run_speed_bench();
    run_precision_bench();
}
