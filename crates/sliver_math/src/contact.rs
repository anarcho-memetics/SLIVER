use crate::vec3::Vec3;

/// Closest point on triangle (a, b, c) to point `p`.
/// Voronoi-region method (Ericson, "Real-Time Collision Detection", 5.1.5).
/// Only dot products and additions, no division except on the final interpolation.
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a; // vertex region A
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // vertex region B
    }

    let vc = d1 * d4 - d3 * d2;
    // The `> 1e-30` guards skip a branch whose edge has collapsed (0/0 would give NaN);
    // another region then resolves the point correctly.
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 && (d1 - d3) > 1e-30 {
        let v = d1 / (d1 - d3);
        return a + ab * v; // edge AB
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // vertex region C
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 && (d2 - d6) > 1e-30 {
        let w = d2 / (d2 - d6);
        return a + ac * w; // edge AC
    }

    let va = d3 * d6 - d5 * d4;
    let den_bc = (d4 - d3) + (d5 - d6);
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 && den_bc > 1e-30 {
        let w = (d4 - d3) / den_bc;
        return b + (c - b) * w; // edge BC
    }

    // Inside the face: barycentric projection.
    let sum = va + vb + vc;
    if sum.abs() < 1e-30 {
        return a; // degenerate triangle
    }
    let denom = 1.0 / sum;
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// Contact between a sphere and one triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereContact {
    /// Unit normal pointing from the triangle toward the sphere center.
    pub normal: Vec3,
    /// Closest point on the triangle.
    pub point: Vec3,
    /// Penetration depth (radius - distance), always > 0.
    pub depth: f32,
}

/// Sphere vs triangle overlap test. Two-sided: the normal always points toward the sphere center.
pub fn sphere_triangle(center: Vec3, radius: f32, a: Vec3, b: Vec3, c: Vec3) -> Option<SphereContact> {
    let q = closest_point_on_triangle(center, a, b, c);
    let d = center - q;
    let dist_sq = d.len_sq();
    // Explicit finiteness check so a NaN distance counts as "no contact" without tripping Clippy.
    if !dist_sq.is_finite() || dist_sq >= radius * radius {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 {
        d * (1.0 / dist)
    } else {
        // Center lies on the triangle: fall back to the face normal.
        let n = (b - a).cross(c - a);
        if n.len_sq() > 1e-20 {
            n.normalize()
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        }
    };
    Some(SphereContact {
        normal,
        point: q,
        depth: radius - dist,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).len() < 1e-5
    }

    fn tri() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        )
    }

    #[test]
    fn closest_inside_face() {
        let (a, b, c) = tri();
        let q = closest_point_on_triangle(Vec3::new(0.5, 1.0, 0.5), a, b, c);
        assert!(close(q, Vec3::new(0.5, 0.0, 0.5)));
    }

    #[test]
    fn closest_vertex_region() {
        let (a, b, c) = tri();
        let q = closest_point_on_triangle(Vec3::new(-1.0, 0.0, -1.0), a, b, c);
        assert!(close(q, a));
    }

    #[test]
    fn closest_edge_ab() {
        let (a, b, c) = tri();
        let q = closest_point_on_triangle(Vec3::new(1.0, 0.5, -1.0), a, b, c);
        assert!(close(q, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn closest_edge_bc() {
        let (a, b, c) = tri();
        let q = closest_point_on_triangle(Vec3::new(2.0, 0.0, 2.0), a, b, c);
        assert!(close(q, Vec3::new(1.0, 0.0, 1.0)));
    }

    #[test]
    fn degenerate_two_equal_vertices_is_finite_and_correct() {
        // a == b: before the guards the AB branch computed 0/0 = NaN for this point.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let c = Vec3::new(2.0, 0.0, 0.0);
        let q = closest_point_on_triangle(Vec3::new(1.0, 1.0, 0.0), a, a, c);
        assert!(q.x.is_finite() && q.y.is_finite() && q.z.is_finite());
        assert!(close(q, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn fully_collapsed_triangle_is_finite() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let q = closest_point_on_triangle(Vec3::new(5.0, 5.0, 5.0), a, a, a);
        assert!(q.x.is_finite() && q.y.is_finite() && q.z.is_finite());
        assert!(close(q, a));
    }

    #[test]
    fn sphere_vs_degenerate_triangle_has_no_nan() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let c = Vec3::new(2.0, 0.0, 0.0);
        let hit = sphere_triangle(Vec3::new(1.0, 0.2, 0.0), 0.35, a, a, c).expect("should overlap");
        assert!(hit.depth.is_finite() && (hit.depth - 0.15).abs() < 1e-5);
        assert!(close(hit.normal, Vec3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn sphere_overlaps_floor_triangle() {
        // Same floor triangle (+Y normal) used in the ray tests.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 1.0);
        let c = Vec3::new(1.0, 0.0, 1.0);
        let hit = sphere_triangle(Vec3::new(0.3, 0.2, 0.6), 0.35, a, b, c).expect("should overlap");
        assert!((hit.depth - 0.15).abs() < 1e-5);
        assert!(close(hit.normal, Vec3::new(0.0, 1.0, 0.0)));
        assert!(sphere_triangle(Vec3::new(0.3, 0.5, 0.6), 0.35, a, b, c).is_none());
    }
}
