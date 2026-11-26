use crate::collision::Contact;
use sliver_math::Vec3;

/// Tunable surface response. Generic: usable by the player, enemies or projectiles.
/// All values are starting points to be tuned by feel in the test room.
#[derive(Clone, Copy, Debug)]
pub struct ContactResponse {
    /// Coulomb friction coefficient (mu): tangential speed lost per unit of normal pressure.
    pub friction: f32,
    /// Restitution (e): fraction of the normal impact speed returned as bounce.
    pub restitution: f32,
    /// Minimum normal impact speed (s_n) for any bounce. Below it the surface does not bounce (no jitter).
    pub bounce_threshold: f32,
    pub max_bounce: f32,
}

/// Simple inertia: moves `vel` toward `target` by at most `accel * dt`.
pub fn accelerate_toward(vel: Vec3, target: Vec3, accel: f32, dt: f32) -> Vec3 {
    let dv = target - vel;
    let dv_len = dv.len();
    let max_dv = accel * dt;
    if dv_len <= max_dv {
        target
    } else {
        vel + dv * (max_dv / dv_len)
    }
}

/// Friction for SUSTAINED contacts, applied BEFORE `move_sphere` using the previous tick's contacts.
///
/// Only the tangential part is reduced (|v_t'| = max(0, |v_t| - mu * s_n)); the normal component is left
/// intact so the sphere keeps pressing into the surface, the contact stays alive every tick and `fresh`
/// does not flicker. Because this runs before the move, a push that stays inside the static friction cone
/// produces no sideways displacement at all.
pub fn pre_move_friction(v: Vec3, prev: &[Contact], p: &ContactResponse) -> Vec3 {
    pre_move_friction_by(v, prev, |_| p)
}

pub fn pre_move_friction_by<'a, F>(mut v: Vec3, prev: &[Contact], mut pick: F) -> Vec3
where
    F: FnMut(&Contact) -> &'a ContactResponse,
{
    for c in prev {
        let vn = v.dot(c.normal);
        if vn >= 0.0 {
            continue;
        }
        let p = pick(c);
        if p.friction <= 0.0 {
            continue;
        }
        let s_n = -vn;
        let v_t = v - c.normal * vn;
        let t_len = v_t.len();
        if t_len <= 1e-6 {
            continue;
        }
        let new_len = (t_len - p.friction * s_n).max(0.0);
        v = c.normal * vn + v_t * (new_len / t_len);
    }
    v
}

pub fn respond(v: Vec3, contacts: &[Contact], p: &ContactResponse) -> Vec3 {
    respond_by(v, contacts, |_| p)
}

pub fn respond_by<'a, F>(mut v: Vec3, contacts: &[Contact], mut pick: F) -> Vec3
where
    F: FnMut(&Contact) -> &'a ContactResponse,
{
    for c in contacts {
        let vn = v.dot(c.normal);
        if vn >= 0.0 {
            continue;
        }
        let p = pick(c);
        let s_n = -vn;
        let mut out = v - c.normal * vn;

        if c.fresh && p.friction > 0.0 {
            let t_len = out.len();
            out = if t_len > 1e-6 {
                out * ((t_len - p.friction * s_n).max(0.0) / t_len)
            } else {
                Vec3::ZERO
            };
        }

        if s_n > p.bounce_threshold {
            let bounce = (p.restitution * (s_n - p.bounce_threshold)).min(p.max_bounce);
            out = out + c.normal * bounce;
        }
        v = out;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::move_sphere;
    use crate::scene::build_demo_room;

    fn contact(normal: Vec3, fresh: bool) -> Contact {
        Contact {
            tri: 0,
            normal,
            point: Vec3::ZERO,
            depth: 0.0,
            impact_speed: 0.0,
            tangent_velocity: Vec3::ZERO,
            fresh,
        }
    }

    fn params(threshold: f32) -> ContactResponse {
        ContactResponse {
            friction: 0.6,
            restitution: 0.25,
            bounce_threshold: threshold,
            max_bounce: 10.0,
        }
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).len() < 1e-5
    }

    const WALL_N: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    #[test]
    fn head_on_impact_stops_sideways_motion() {
        // s_n = 4, v_t = 0.5: friction 0.6 * 4 = 2.4 > 0.5, so no sideways push remains.
        let v = respond(Vec3::new(0.5, 0.0, -4.0), &[contact(WALL_N, true)], &params(100.0));
        assert!(close(v, Vec3::ZERO));
    }

    #[test]
    fn grazing_contact_barely_slows() {
        // s_n = 0.1, v_t = 3: loses 0.6 * 0.1 = 0.06.
        let v = respond(Vec3::new(3.0, 0.0, -0.1), &[contact(WALL_N, true)], &params(100.0));
        assert!(close(v, Vec3::new(2.94, 0.0, 0.0)));
    }

    #[test]
    fn sustained_contact_gets_no_friction_in_respond() {
        // Friction for sustained contacts is pre_move_friction's job, so respond only cuts the normal part.
        let v = respond(Vec3::new(3.0, 0.0, -0.1), &[contact(WALL_N, false)], &params(100.0));
        assert!(close(v, Vec3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn bounce_only_above_threshold() {
        let hard = respond(Vec3::new(0.0, 0.0, -4.0), &[contact(WALL_N, true)], &params(2.0));
        assert!(close(hard, Vec3::new(0.0, 0.0, 0.5))); // e * s_n = 0.25 * 4
        let soft = respond(Vec3::new(0.0, 0.0, -1.0), &[contact(WALL_N, true)], &params(2.0));
        assert!(close(soft, Vec3::ZERO));
    }

    #[test]
    fn moving_away_is_untouched() {
        let v = Vec3::new(1.0, 0.0, 2.0);
        assert!(close(respond(v, &[contact(WALL_N, true)], &params(2.0)), v));
        assert!(close(pre_move_friction(v, &[contact(WALL_N, false)], &params(2.0)), v));
    }

    #[test]
    fn pre_move_friction_keeps_normal_velocity() {
        // Inside the cone: tangential part is removed, normal part (pressing into the wall) is kept.
        let v = pre_move_friction(Vec3::new(0.1, 0.0, -0.5), &[contact(WALL_N, false)], &params(2.0));
        assert!(close(v, Vec3::new(0.0, 0.0, -0.5)));
        // Outside the cone: only mu * s_n = 0.3 is lost.
        let v = pre_move_friction(Vec3::new(3.0, 0.0, -0.5), &[contact(WALL_N, false)], &params(2.0));
        assert!(close(v, Vec3::new(2.7, 0.0, -0.5)));
    }

    #[test]
    fn accelerate_toward_limits_the_change() {
        let target = Vec3::new(10.0, 0.0, 0.0);
        let v = accelerate_toward(Vec3::ZERO, target, 30.0, 1.0 / 120.0);
        assert!(close(v, Vec3::new(0.25, 0.0, 0.0)));
        let snapped = accelerate_toward(Vec3::new(9.9, 0.0, 0.0), target, 30.0, 1.0 / 120.0);
        assert!(close(snapped, target));
    }

    #[test]
    fn diagonal_push_inside_friction_cone_does_not_drift() {
        const STEP: f32 = 1.0 / 120.0;
        const R: f32 = 0.35;
        let tris = build_demo_room();
        let p = params(2.5);

        // Direction (1, 0, -2): tangent/normal ratio 0.5 < mu = 0.6, i.e. inside the static friction cone.
        let wish = Vec3::new(1.0, 0.0, -2.0);
        let target = wish * (5.5 / wish.len());

        let mut pos = Vec3::new(2.0, 1.5, -9.6); // 0.05 from touching the north wall
        let mut vel = Vec3::ZERO;
        let mut contacts: Vec<Contact> = Vec::new();
        let mut prev: Vec<Contact> = Vec::new();
        let mut x_ref = 0.0;

        for tick in 0..180 {
            vel = accelerate_toward(vel, target, 30.0, STEP);
            std::mem::swap(&mut prev, &mut contacts);
            vel = pre_move_friction(vel, &prev, &p);
            pos = move_sphere(&tris, pos, R, vel, STEP, &prev, &mut contacts);
            vel = respond(vel, &contacts, &p);

            if tick == 30 {
                x_ref = pos.x; // approach and first impact are over by now
            }
            if tick >= 30 {
                assert_eq!(contacts.len(), 1, "contact must persist at tick {tick}");
                assert!(!contacts[0].fresh, "contact must stay sustained at tick {tick}");
            }
        }
        assert!((pos.x - x_ref).abs() < 1e-4, "tangential drift: {}", pos.x - x_ref);
        assert!((pos.z - (-10.0 + R)).abs() < 1e-3);
    }
}
