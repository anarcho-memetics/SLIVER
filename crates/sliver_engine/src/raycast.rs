use crate::scene::Tri;
use sliver_math::{RayHit, Vec3, ray_triangle};

/// Nearest hit of a ray against a triangle list (linear scan).
/// `dir` should be normalized so `hit.t` is a real distance.
/// Returns the index of the triangle in `tris` together with the hit.
pub fn raycast(tris: &[Tri], origin: Vec3, dir: Vec3, t_max: f32) -> Option<(usize, RayHit)> {
    let mut best: Option<(usize, RayHit)> = None;
    let mut limit = t_max;
    for (i, tri) in tris.iter().enumerate() {
        if let Some(hit) = ray_triangle(origin, dir, tri.p0, tri.p1, tri.p2, limit) {
            // Shrinking the limit makes later triangles only report closer hits.
            limit = hit.t;
            best = Some((i, hit));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::build_demo_room;

    #[test]
    fn floor_below_camera_start() {
        let tris = build_demo_room();
        let (_, hit) = raycast(
            &tris,
            Vec3::new(0.3, 1.8, 4.2),
            Vec3::new(0.0, -1.0, 0.0),
            100.0,
        )
        .expect("should hit the floor");
        assert!((hit.t - 1.8).abs() < 1e-4);
        assert!(hit.front_face);
    }

    #[test]
    fn north_wall_ahead() {
        let tris = build_demo_room();
        // From the camera start looking toward -Z, something must be hit before the north wall at z = -10.
        let (_, hit) = raycast(
            &tris,
            Vec3::new(2.0, 1.0, 4.2),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
        )
        .expect("should hit the north wall");
        assert!((hit.t - 14.2).abs() < 1e-3);
    }

    #[test]
    fn t_max_limits_the_search() {
        let tris = build_demo_room();
        assert!(
            raycast(
                &tris,
                Vec3::new(0.3, 1.8, 4.2),
                Vec3::new(0.0, -1.0, 0.0),
                1.0,
            )
            .is_none()
        );
    }
}
