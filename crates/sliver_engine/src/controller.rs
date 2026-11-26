//! Walking controller: gravity, ground detection, jump and per-surface response.
//!
//! This is game-side policy built on the generic engine pieces (`collision.rs`, `motion.rs`).
//! Nothing in those two files knows about players, floors or jumping; it can move to the game repo as is.

use crate::collision::{move_sphere, Contact};
use crate::motion::{accelerate_toward, pre_move_friction_by, respond_by, ContactResponse};
use crate::scene::Tri;
use sliver_math::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct WalkConfig {
    /// Body sphere radius. The capsule can replace it later without touching the engine pieces.
    pub radius: f32,
    pub gravity: f32,
    pub jump_speed: f32,
    pub max_speed: f32,
    /// Horizontal acceleration on the ground.
    pub accel: f32,
    /// Horizontal acceleration in the air (very low: no steering while floating).
    pub air_accel: f32,
    /// Contacts with normal.y above this are walkable floor (0.7 is about 45 degrees).
    pub walkable_normal_y: f32,
    /// Response for walls, ceilings and steep slopes: Coulomb friction + bounce above a threshold.
    pub wall: ContactResponse,
    /// Response for walkable floor: no tangential friction (walking traction is the horizontal
    /// acceleration), no bounce when landing.
    pub floor: ContactResponse,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            radius: 0.35,
            gravity: 15.0,
            jump_speed: 4.2,
            max_speed: 5.5,
            accel: 30.0,
            air_accel: 3.0,
            walkable_normal_y: 0.7,
            wall: ContactResponse {
                friction: 0.6,
                restitution: 0.25,
                bounce_threshold: 2.5,
                max_bounce: 1.2,
            },
            floor: ContactResponse {
                friction: 0.0,
                restitution: 0.0,
                bounce_threshold: f32::INFINITY,
                max_bounce: 0.0,
            },
        }
    }
}

fn pick<'a>(cfg: &'a WalkConfig, c: &Contact) -> &'a ContactResponse {
    if c.normal.y > cfg.walkable_normal_y {
        &cfg.floor
    } else {
        &cfg.wall
    }
}

pub struct Walker {
    pub cfg: WalkConfig,
    /// Center of the body sphere (not the eye).
    pub pos: Vec3,
    pub vel: Vec3,
    /// True when this tick produced a walkable contact.
    pub grounded: bool,
    /// Contacts of the last tick (per plane, with telemetry). Read them right after `tick`.
    pub contacts: Vec<Contact>,
    prev_contacts: Vec<Contact>,
}

impl Walker {
    pub fn new(cfg: WalkConfig, pos: Vec3) -> Self {
        Self {
            cfg,
            pos,
            vel: Vec3::ZERO,
            grounded: false,
            contacts: Vec::new(),
            prev_contacts: Vec::new(),
        }
    }

    /// One fixed physics tick.
    /// `wish` is the desired horizontal direction (length 0..=1, y ignored).
    /// `jump` must be true only on the tick where the jump is requested (press edge, not held).
    pub fn tick(&mut self, tris: &[Tri], dt: f32, wish: Vec3, jump: bool) {
        let cfg = self.cfg;

        // Jump only from the ground (grounded comes from the previous tick's contacts).
        if jump && self.grounded {
            self.vel.y = cfg.jump_speed;
            self.grounded = false;
        }

        // Horizontal inertia only: accelerate_toward never touches vel.y, so it cannot slow a fall.
        let accel = if self.grounded { cfg.accel } else { cfg.air_accel };
        let horizontal = Vec3::new(self.vel.x, 0.0, self.vel.z);
        let target = Vec3::new(wish.x, 0.0, wish.z) * cfg.max_speed;
        let h = accelerate_toward(horizontal, target, accel, dt);
        self.vel = Vec3::new(h.x, self.vel.y, h.z);

        // Friction of sustained contacts BEFORE moving (floor has friction 0, so it is skipped).
        std::mem::swap(&mut self.prev_contacts, &mut self.contacts);
        self.vel = pre_move_friction_by(self.vel, &self.prev_contacts, |c| pick(&cfg, c));

        // Gravity is applied AFTER the friction step on purpose. Wall friction acts on the whole
        // tangential velocity, and for a wall that includes vertical speed: with gravity applied first,
        // pressing into a wall while standing on the floor cancelled the gravity pressing you into the
        // floor, so the floor contact (and `grounded`) flickered.
        self.vel.y -= cfg.gravity * dt;

        self.pos = move_sphere(
            tris,
            self.pos,
            cfg.radius,
            self.vel,
            dt,
            &self.prev_contacts,
            &mut self.contacts,
        );

        self.vel = respond_by(self.vel, &self.contacts, |c| pick(&cfg, c));

        self.grounded = self
            .contacts
            .iter()
            .any(|c| c.normal.y > cfg.walkable_normal_y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::build_demo_room;

    const DT: f32 = 1.0 / 120.0;

    #[test]
    fn walking_on_floor_reaches_max_speed_without_ground_friction() {
        let tris = build_demo_room();
        let cfg = WalkConfig::default();
        let mut w = Walker::new(cfg, Vec3::new(-5.0, cfg.radius, 3.0));
        let wish = Vec3::new(1.0, 0.0, 0.0);

        for tick in 0..150 {
            w.tick(&tris, DT, wish, false);
            if tick >= 5 {
                assert_eq!(w.contacts.len(), 1, "one floor contact at tick {tick}");
                assert!(!w.contacts[0].fresh, "floor contact must stay sustained at tick {tick}");
                assert!(w.grounded, "grounded at tick {tick}");
                assert!((w.pos.y - cfg.radius).abs() < 1e-3, "height at tick {tick}");
            }
            if tick >= 40 {
                assert!(
                    (w.vel.x - cfg.max_speed).abs() < 1e-3,
                    "speed at tick {tick}: {}",
                    w.vel.x
                );
            }
        }
    }

    #[test]
    fn diagonal_push_into_wall_on_floor_does_not_drift() {
        let tris = build_demo_room();
        let cfg = WalkConfig::default();
        // East end of the north wall (the platform only reaches x = 4).
        let mut w = Walker::new(cfg, Vec3::new(5.0, cfg.radius, -9.6));
        // Tangent/normal ratio 0.5 < mu = 0.6: inside the wall's static friction cone.
        let wish = Vec3::new(1.0, 0.0, -2.0);
        let wish = wish * (1.0 / wish.len());

        let mut x_ref = 0.0;
        for tick in 0..200 {
            w.tick(&tris, DT, wish, false);
            if tick == 40 {
                x_ref = w.pos.x; // approach and first impact are over by now
            }
            if tick >= 40 {
                assert_eq!(w.contacts.len(), 2, "floor + wall at tick {tick}");
                assert!(
                    w.contacts.iter().all(|c| !c.fresh),
                    "contacts must stay sustained at tick {tick}"
                );
                assert!(w.grounded, "grounded at tick {tick}");
            }
        }
        assert!((w.pos.x - x_ref).abs() < 1e-4, "lateral drift: {}", w.pos.x - x_ref);
        assert!((w.pos.z - (-10.0 + cfg.radius)).abs() < 1e-3);
    }

    #[test]
    fn landing_stops_vertical_speed_without_bounce_and_reports_impact() {
        let tris = build_demo_room();
        let cfg = WalkConfig::default();
        let mut w = Walker::new(cfg, Vec3::new(0.3, 1.0, 3.0));

        let mut landed = false;
        for _ in 0..100 {
            let vy_before = w.vel.y;
            w.tick(&tris, DT, Vec3::ZERO, false);
            if !w.contacts.is_empty() {
                // Speed that reached the move: previous vertical speed plus this tick's gravity.
                let expected = -(vy_before - cfg.gravity * DT);
                assert_eq!(w.contacts.len(), 1);
                assert!(w.contacts[0].fresh, "first touch must be fresh");
                assert!((w.contacts[0].impact_speed - expected).abs() < 1e-3);
                // Free fall over 0.65 units: sqrt(2 g h) = 3.57, discretization stays close.
                let theoretical = (2.0 * cfg.gravity * 0.65).sqrt();
                assert!((w.contacts[0].impact_speed - theoretical).abs() < 0.15);
                assert!(w.vel.y.abs() < 1e-3, "vertical speed after landing: {}", w.vel.y);
                landed = true;
                break;
            }
        }
        assert!(landed, "the body never reached the floor");

        // It stays on the floor: no bounce, sustained contact.
        for tick in 0..30 {
            w.tick(&tris, DT, Vec3::ZERO, false);
            assert!(w.vel.y <= 1e-3, "bounce at tick {tick}: vy = {}", w.vel.y);
            assert!(w.grounded);
            assert!(!w.contacts[0].fresh);
        }
        assert!((w.pos.y - cfg.radius).abs() < 1e-3);
    }

    #[test]
    fn jump_only_from_the_ground() {
        let tris = build_demo_room();
        let cfg = WalkConfig::default();
        let mut w = Walker::new(cfg, Vec3::new(0.3, cfg.radius, 3.0));
        for _ in 0..10 {
            w.tick(&tris, DT, Vec3::ZERO, false);
        }
        assert!(w.grounded);

        w.tick(&tris, DT, Vec3::ZERO, true);
        let expected = cfg.jump_speed - cfg.gravity * DT;
        assert!(!w.grounded);
        assert!((w.vel.y - expected).abs() < 1e-3);

        // Asking again in the air does nothing: only gravity acts.
        w.tick(&tris, DT, Vec3::ZERO, true);
        assert!((w.vel.y - (expected - cfg.gravity * DT)).abs() < 1e-3);
    }
}
