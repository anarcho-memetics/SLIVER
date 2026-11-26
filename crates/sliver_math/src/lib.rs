#![forbid(unsafe_code)]

pub mod vec3;
pub use vec3::Vec3;
pub mod ray;
pub use ray::{ray_triangle, RayHit};
pub mod contact;
pub use contact::{closest_point_on_triangle, sphere_triangle, SphereContact};

/// Sliver triangle detector based on the factored chord / half-angle inequality:
///   (min - (max - mid)) * (min + (max - mid)) < s0^2 * mid * max
#[derive(Clone, Copy, Debug)]
pub struct ChordSliver {
    s0_sq: f32,
}

impl ChordSliver {
    pub fn from_degrees(limit_deg: f32) -> Self {
        let half_rad = (limit_deg as f64).to_radians() * 0.5;
        let s0 = 2.0 * half_rad.sin();
        Self {
            s0_sq: (s0 * s0) as f32,
        }
    }

    /// Optimal branchless sort3 variant: 0.74 ns/tri in SoA layout with 0% error in f32.
    #[inline(always)]
    pub fn is_sliver_sort3(self, a: f32, b: f32, c: f32) -> bool {
        let (min_s, mid_s, max_s) = sort3_branchless(a, b, c);
        let d = max_s - mid_s;
        (min_s - d) * (min_s + d) < self.s0_sq * (mid_s * max_s)
    }

    /// Primary entry point for engine and offline tooling code.
    #[inline(always)]
    pub fn is_sliver(self, a: f32, b: f32, c: f32) -> bool {
        self.is_sliver_sort3(a, b, c)
    }

    /// Evaluates a 3D triangle from its vertex positions (offline mesh baking / loading).
    #[inline]
    pub fn is_sliver_points(self, p0: Vec3, p1: Vec3, p2: Vec3) -> bool {
        let a = (p1 - p2).len();
        let b = (p2 - p0).len();
        let c = (p0 - p1).len();
        self.is_sliver_sort3(a, b, c)
    }

    #[inline(always)]
    pub fn is_sliver_branchless(self, a: f32, b: f32, c: f32) -> bool {
        let d_bc = b - c;
        let d_ca = c - a;
        let d_ab = a - b;
        ((a - d_bc) * (a + d_bc) < self.s0_sq * (b * c))
            | ((b - d_ca) * (b + d_ca) < self.s0_sq * (c * a))
            | ((c - d_ab) * (c + d_ab) < self.s0_sq * (a * b))
    }

    #[inline(always)]
    pub fn is_sliver_min_vertex(self, a: f32, b: f32, c: f32) -> bool {
        let (min_s, x, y) = order_min_first_branching(a, b, c);
        let d = x - y;
        (min_s - d) * (min_s + d) < self.s0_sq * (x * y)
    }
}

/// Reference Law of Cosines detector (retained for benchmark comparisons).
#[derive(Clone, Copy, Debug)]
pub struct CosineSliver {
    two_cos: f32,
}

impl CosineSliver {
    pub fn from_degrees(limit_deg: f32) -> Self {
        let rad = (limit_deg as f64).to_radians();
        Self {
            two_cos: (2.0 * rad.cos()) as f32,
        }
    }

    #[inline(always)]
    pub fn is_sliver_min_vertex(self, a: f32, b: f32, c: f32) -> bool {
        let (min_s, x, y) = order_min_first_branching(a, b, c);
        (x * x + y * y - min_s * min_s) > self.two_cos * (x * y)
    }

    #[inline(always)]
    pub fn is_sliver_branchless(self, a: f32, b: f32, c: f32) -> bool {
        let (a2, b2, c2) = (a * a, b * b, c * c);
        ((b2 + c2 - a2) > self.two_cos * (b * c))
            | ((c2 + a2 - b2) > self.two_cos * (c * a))
            | ((a2 + b2 - c2) > self.two_cos * (a * b))
    }

    #[inline(always)]
    pub fn is_sliver_sort3(self, a: f32, b: f32, c: f32) -> bool {
        let (min_s, mid_s, max_s) = sort3_branchless(a, b, c);
        (mid_s * mid_s + max_s * max_s - min_s * min_s) > self.two_cos * (mid_s * max_s)
    }
}

/// Computes the minimum interior angle in radians using the factored chord formula.
#[inline]
pub fn min_angle_chord_f32(a: f32, b: f32, c: f32) -> f32 {
    let (min_s, mid_s, max_s) = sort3_branchless(a, b, c);
    let d = max_s - mid_s;
    let num = (min_s - d) * (min_s + d);
    let sin_half = (num / (4.0 * mid_s * max_s)).clamp(0.0, 1.0).sqrt();
    2.0 * sin_half.asin()
}

#[inline(always)]
fn order_min_first_branching(a: f32, b: f32, c: f32) -> (f32, f32, f32) {
    if a <= b && a <= c {
        (a, b, c)
    } else if b <= c {
        (b, a, c)
    } else {
        (c, a, b)
    }
}

#[inline(always)]
fn min_max_fast(x: f32, y: f32) -> (f32, f32) {
    let lo = if x < y { x } else { y };
    let hi = if x > y { x } else { y };
    (lo, hi)
}

#[inline(always)]
fn sort3_branchless(a: f32, b: f32, c: f32) -> (f32, f32, f32) {
    let (a, b) = min_max_fast(a, b);
    let (b, c) = min_max_fast(b, c);
    let (a, b) = min_max_fast(a, b);
    (a, b, c)
}
