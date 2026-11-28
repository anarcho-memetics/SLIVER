use crate::scene::Tri;
use sliver_math::{SphereContact, Vec3, sphere_triangle};

/// Distance kept between the sphere and a surface after push-out.
const SKIN: f32 = 1e-4;
/// Max push-out iterations per substep (resolves corners and multiple planes).
const MAX_PUSH_ITERS: usize = 4;
/// Upper bound on substeps per move.
const MAX_SUBSTEPS: usize = 16;
/// Two contacts whose normals have dot > this are treated as the same plane (~10 deg).
const SAME_PLANE_COS: f32 = 0.985;

/// Per-plane contact telemetry for one move. Pure geometry: knows nothing about damage, camera or game rules.
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    /// Triangle that produced the deepest penetration for this plane.
    #[allow(dead_code)]
    pub tri: usize,
    /// Unit normal pointing from the surface toward the sphere.
    pub normal: Vec3,
    /// Closest point on the surface.
    pub point: Vec3,
    /// Deepest penetration seen on this plane during the move.
    pub depth: f32,
    /// s_n = -min(0, dot(v, n)): speed pressing into the surface.
    pub impact_speed: f32,
    /// v_t = v - dot(v, n) * n: velocity parallel to the surface.
    #[allow(dead_code)]
    pub tangent_velocity: Vec3,
    /// True if no contact with a similar normal existed in the previous move.
    pub fresh: bool,
}

/// Deepest overlapping triangle (linear scan).
fn deepest_contact(tris: &[Tri], center: Vec3, radius: f32) -> Option<(usize, SphereContact)> {
    let mut best: Option<(usize, SphereContact)> = None;
    for (i, t) in tris.iter().enumerate() {
        if let Some(c) = sphere_triangle(center, radius, t.p0, t.p1, t.p2)
            && best.is_none_or(|(_, b)| c.depth > b.depth)
        {
            best = Some((i, c));
        }
    }
    best
}

fn record(out: &mut Vec<Contact>, tri: usize, c: &SphereContact, vel: Vec3) {
    let vn = vel.dot(c.normal);
    let contact = Contact {
        tri,
        normal: c.normal,
        point: c.point,
        depth: c.depth,
        impact_speed: (-vn).max(0.0),
        tangent_velocity: vel - c.normal * vn,
        fresh: false,
    };
    if let Some(e) = out
        .iter_mut()
        .find(|e| e.normal.dot(c.normal) > SAME_PLANE_COS)
    {
        if contact.depth > e.depth {
            *e = contact;
        }
    } else {
        out.push(contact);
    }
}

/// Moves a sphere by `vel * dt` against the triangle list.
///
/// - The move is split into substeps no longer than half the radius, so thin walls are not skipped.
/// - After each substep the sphere is pushed out of the deepest contact along its normal (repeated up to
///   MAX_PUSH_ITERS times). Push-out preserves tangential motion; it does NOT modify velocity.
/// - `out` receives one merged contact per plane touched, with telemetry computed from the input `vel`.
/// - `prev` is the previous move's contacts, used only to flag `fresh` contacts.
///
/// Velocity response (friction, bounce, inertia) lives in `motion.rs`, not here.
pub fn move_sphere(
    tris: &[Tri],
    pos: Vec3,
    radius: f32,
    vel: Vec3,
    dt: f32,
    prev: &[Contact],
    out: &mut Vec<Contact>,
) -> Vec3 {
    out.clear();
    let delta = vel * dt;
    let dist = delta.len();
    let steps = ((dist / (radius * 0.5)).ceil() as usize).clamp(1, MAX_SUBSTEPS);
    let step = delta * (1.0 / steps as f32);

    let mut p = pos;
    for _ in 0..steps {
        p = p + step;
        for _ in 0..MAX_PUSH_ITERS {
            let Some((tri, c)) = deepest_contact(tris, p, radius) else {
                break;
            };
            p = p + c.normal * (c.depth + SKIN);
            record(out, tri, &c, vel);
        }
    }

    for c in out.iter_mut() {
        c.fresh = !prev.iter().any(|q| q.normal.dot(c.normal) > SAME_PLANE_COS);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::build_demo_room;

    const R: f32 = 0.35;

    #[test]
    fn pushed_out_of_floor() {
        let tris = build_demo_room();
        let mut out = Vec::new();
        let p = move_sphere(
            &tris,
            Vec3::new(0.3, 0.4, 0.6),
            R,
            Vec3::new(0.0, -1.0, 0.0),
            0.1,
            &[],
            &mut out,
        );
        assert!((p.y - R).abs() < 1e-3);
        assert_eq!(out.len(), 1);
        assert!(out[0].normal.y > 0.99);
        assert!((out[0].impact_speed - 1.0).abs() < 1e-5);
        assert!(out[0].fresh);
    }

    #[test]
    fn stops_at_north_wall() {
        let tris = build_demo_room();
        let mut out = Vec::new();
        let p = move_sphere(
            &tris,
            Vec3::new(2.0, 1.5, -9.6),
            R,
            Vec3::new(0.0, 0.0, -2.0),
            0.1,
            &[],
            &mut out,
        );
        assert!((p.z - (-10.0 + R)).abs() < 1e-3);
        assert!(out[0].normal.z > 0.99);
        assert!((out[0].impact_speed - 2.0).abs() < 1e-5);
    }

    #[test]
    fn keeps_tangential_motion_along_wall() {
        let tris = build_demo_room();
        let mut out = Vec::new();
        let p = move_sphere(
            &tris,
            Vec3::new(2.0, 1.5, -9.6),
            R,
            Vec3::new(1.0, 0.0, -2.0),
            0.1,
            &[],
            &mut out,
        );
        assert!((p.x - 2.1).abs() < 1e-3);
        assert!((out[0].tangent_velocity.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fresh_flag_depends_on_previous_contacts() {
        let tris = build_demo_room();
        let start = Vec3::new(2.0, 1.5, -9.6);
        let vel = Vec3::new(0.0, 0.0, -2.0);

        let mut first = Vec::new();
        move_sphere(&tris, start, R, vel, 0.1, &[], &mut first);
        assert!(first[0].fresh);

        let mut second = Vec::new();
        move_sphere(&tris, start, R, vel, 0.1, &first, &mut second);
        assert!(!second[0].fresh);
    }
}
