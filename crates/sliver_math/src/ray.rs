use crate::vec3::Vec3;

/// Result of a ray / triangle intersection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Distance along the ray (in units of `dir` length).
    pub t: f32,
    /// Barycentric coordinates of the hit: point = p0 + u*(p1-p0) + v*(p2-p0).
    pub u: f32,
    pub v: f32,
    /// True when the ray hits the CCW front face (opposite to the normal).
    pub front_face: bool,
}

/// Moller-Trumbore ray / triangle test.
/// Winding is CCW, normal = (p1 - p0) x (p2 - p0), same as the engine's `Tri::normal`.
/// Returns `None` if the ray is parallel, misses, or hits outside (0, t_max).
#[inline]
pub fn ray_triangle(
    origin: Vec3,
    dir: Vec3,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    t_max: f32,
) -> Option<RayHit> {
    const EPS: f32 = 1e-8;

    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let h = dir.cross(e2);
    let det = e1.dot(h);
    if det.abs() < EPS {
        return None; // ray parallel to the triangle plane (or degenerate triangle)
    }

    let inv_det = 1.0 / det;
    let s = origin - p0;
    let u = inv_det * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(e1);
    let v = inv_det * dir.dot(q);
    if !(v >= 0.0 && u + v <= 1.0) {
        return None;
    }

    let t = inv_det * e2.dot(q);
    if t > EPS && t < t_max {
        Some(RayHit {
            t,
            u,
            v,
            front_face: det > 0.0,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Floor triangle with +Y normal, same winding as the engine's floor quads.
    fn floor() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        )
    }

    #[test]
    fn hits_floor_from_above() {
        let (a, b, c) = floor();
        let hit = ray_triangle(Vec3::new(0.3, 2.0, 0.6), Vec3::new(0.0, -1.0, 0.0), a, b, c, 100.0)
            .expect("should hit");
        assert!((hit.t - 2.0).abs() < 1e-5);
        assert!(hit.front_face);
    }

    #[test]
    fn hits_floor_from_below_as_back_face() {
        let (a, b, c) = floor();
        let hit = ray_triangle(Vec3::new(0.3, -1.0, 0.6), Vec3::new(0.0, 1.0, 0.0), a, b, c, 100.0)
            .expect("should hit");
        assert!((hit.t - 1.0).abs() < 1e-5);
        assert!(!hit.front_face);
    }

    #[test]
    fn misses_outside_triangle() {
        let (a, b, c) = floor();
        assert!(ray_triangle(Vec3::new(0.8, 2.0, 0.2), Vec3::new(0.0, -1.0, 0.0), a, b, c, 100.0).is_none());
    }

    #[test]
    fn respects_t_max_and_direction() {
        let (a, b, c) = floor();
        // Too far for t_max.
        assert!(ray_triangle(Vec3::new(0.3, 2.0, 0.6), Vec3::new(0.0, -1.0, 0.0), a, b, c, 1.0).is_none());
        // Triangle is behind the origin.
        assert!(ray_triangle(Vec3::new(0.3, -2.0, 0.6), Vec3::new(0.0, -1.0, 0.0), a, b, c, 100.0).is_none());
    }

    #[test]
    fn parallel_ray_misses() {
        let (a, b, c) = floor();
        assert!(ray_triangle(Vec3::new(0.3, 1.0, 0.6), Vec3::new(1.0, 0.0, 0.0), a, b, c, 100.0).is_none());
    }

    #[test]
    fn hits_thin_sliver_in_f32() {
        // ~0.51 deg minimum angle, same proportions as the west-wall baseboard in the demo room.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(15.6, 0.0, 0.0);
        let c = Vec3::new(15.6, 0.14, 0.0);
        let hit = ray_triangle(Vec3::new(10.0, 0.05, 1.0), Vec3::new(0.0, 0.0, -1.0), a, b, c, 100.0)
            .expect("should hit the sliver");
        assert!((hit.t - 1.0).abs() < 1e-4);
    }
}
