//! Walking controller: gravity, ground detection, jump, steps and per-surface response.
//!
//! This is game-side policy built on the generic engine pieces (`collision.rs`, `motion.rs`).
//! Nothing in those two files knows about players, floors, jumping or steps; it can move to the game repo.

use crate::collision::{move_sphere, Contact};
use crate::motion::{accelerate_toward, pre_move_friction_by, respond_by, ContactResponse};
use crate::scene::Tri;
use sliver_math::Vec3;

/// After a step-up the body glides over the edge; step snapping and airborne rules are off for this long.
const STEP_HOLD_TIME: f32 = 0.30;
/// After support is lost from a grounded state (ledge), `grounded` stays true this long (coyote time)
/// while we look for a ground to snap down to.
const LEDGE_GRACE_TIME: f32 = 0.125;
const MIN_STEP_GAIN: f32 = 0.02;
/// Minimum push into a non-walkable surface to consider the body blocked by it.
const BLOCK_MIN_SPEED: f32 = 0.1;
/// Static grip when standing still on a step edge: below this horizontal speed and above this normal.y,
/// gravity is not allowed to roll the sphere off the ledge.
const EDGE_GRIP_MAX_SPEED: f32 = 0.35;
const EDGE_GRIP_MIN_NORMAL_Y: f32 = 0.25;
/// Extra horizontal speed fraction lost when climbing a step at full speed (stumble feel).
const FAST_STEP_STUMBLE_LOSS: f32 = 0.08;

#[derive(Clone, Copy, Debug)]
pub struct WalkConfig {
    /// Body sphere radius. The capsule can replace it later without touching the engine pieces.
    pub radius: f32,
    pub gravity: f32,
    pub jump_speed: f32,
    pub max_speed: f32,
    pub sprint_speed: f32,
    /// Horizontal acceleration on the ground.
    pub accel: f32,
    /// Horizontal deceleration on the ground when no direction is pressed.
    pub decel: f32,
    /// Horizontal acceleration in the air, applied only while a direction is pressed.
    pub air_accel: f32,
    /// Contacts with normal.y above this are walkable floor (0.7 is about 45 degrees).
    pub walkable_normal_y: f32,
    /// Highest step the body climbs (and the longest drop it snaps down).
    pub max_step_height: f32,
    /// Fraction of horizontal speed lost per meter of step climbed.
    pub step_speed_loss_per_meter: f32,
    /// Response for walls, ceilings and steep slopes: Coulomb friction + capped bounce above a threshold.
    pub wall: ContactResponse,
    /// Response for walkable floor: no tangential friction (walking traction is the horizontal
    /// acceleration), no bounce when landing.
    pub floor: ContactResponse,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            radius: 0.35,
            gravity: 16.0,
            jump_speed: 4.6,
            max_speed: 5.5,
            sprint_speed: 21.5,
            accel: 18.0,
            decel: 40.0,
            air_accel: 3.0,
            walkable_normal_y: 0.7,
            max_step_height: 0.35,
            step_speed_loss_per_meter: 0.45,
            wall: ContactResponse {
                friction: 0.6,
                restitution: 0.12,
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

fn pick<'a>(
    cfg: &'a WalkConfig,
    step_wall: &'a ContactResponse,
    foot_y: f32,
    c: &Contact,
) -> &'a ContactResponse {
    if c.normal.y > cfg.walkable_normal_y {
        &cfg.floor
    } else if c.point.y - foot_y <= cfg.max_step_height.min(cfg.radius - 0.01) {
        step_wall
    } else {
        &cfg.wall
    }
}

fn horizontal_len(a: Vec3, b: Vec3) -> f32 {
    let d = b - a;
    (d.x * d.x + d.z * d.z).sqrt()
}

/// Speculative step-up built on `move_sphere` (up, forward, down) on scratch state.
fn try_step_up(
    cfg: &WalkConfig,
    tris: &[Tri],
    start: Vec3,
    vel_h: Vec3,
    dt: f32,
    blocked_progress: f32,
) -> Option<(Vec3, f32)> {
    let speed = vel_h.len();
    if speed < 1e-4 {
        return None;
    }
    let dir = vel_h * (1.0 / speed);
    let wanted = speed * dt;
    let probe = cfg.radius * 1.1;
    let mut scratch: Vec<Contact> = Vec::new();

    // 1) up (a low ceiling stops it by itself)
    let up = move_sphere(tris, start, cfg.radius, Vec3::new(0.0, cfg.max_step_height, 0.0), 1.0, &[], &mut scratch);
    let raised = up.y - start.y;
    if raised < MIN_STEP_GAIN {
        return None;
    }

    // 2) forward over the obstacle: must go clearly further than the blocked normal move did
    let fwd = move_sphere(tris, up, cfg.radius, dir * probe, 1.0, &[], &mut scratch);
    let probe_ratio = horizontal_len(up, fwd) / probe;
    let blocked_ratio = (blocked_progress / wanted).min(1.0);
    if probe_ratio < 0.75 || probe_ratio <= blocked_ratio {
        return None;
    }

    // 3) down (a little extra so a flat surface registers as a contact)
    let down = move_sphere(tris, fwd, cfg.radius, Vec3::new(0.0, -(raised + 0.02), 0.0), 1.0, &[], &mut scratch);
    let gain = down.y - start.y;
    if gain < MIN_STEP_GAIN || gain > cfg.max_step_height + 1e-3 {
        return None;
    }
    if !scratch.iter().any(|c| c.normal.y > cfg.walkable_normal_y) {
        return None;
    }

    // Apply: this tick's normal horizontal advance, at the new height; it must be free of penetration.
    let cand = Vec3::new(start.x + vel_h.x * dt, down.y, start.z + vel_h.z * dt);
    let fixed = move_sphere(tris, cand, cfg.radius, Vec3::ZERO, 1.0, &[], &mut scratch);
    if (fixed - cand).len() > 0.02 {
        return None;
    }
    Some((fixed, gain))
}

/// Pure vertical snap down onto a walkable surface.
fn vertical_snap(cfg: &WalkConfig, tris: &[Tri], pos: Vec3) -> Option<(Vec3, f32, Vec<Contact>)> {
    let mut c: Vec<Contact> = Vec::new();
    let p = move_sphere(tris, pos, cfg.radius, Vec3::new(0.0, -cfg.max_step_height, 0.0), 1.0, &[], &mut c);
    let drop = pos.y - p.y;
    if horizontal_len(pos, p) < 0.01 && drop > MIN_STEP_GAIN && c.iter().any(|k| k.normal.y > cfg.walkable_normal_y) {
        for k in c.iter_mut() {
            k.fresh = false;
            k.impact_speed = 0.0;
        }
        Some((p, drop, c))
    } else {
        None
    }
}

pub struct Walker {
    pub cfg: WalkConfig,
    /// Center of the body sphere (not the eye).
    pub sprinting:bool,
    pub pos: Vec3,
    pub vel: Vec3,
    /// Ground state used for control (walk acceleration, jump).
    pub grounded: bool,
    /// Contacts of the last tick (per plane, with telemetry). Read them right after `tick`.
    pub contacts: Vec<Contact>,
    /// Vertical displacement TELEPORTED by a step this tick: positive when stepping up, negative when
    /// snapping down, 0.0 otherwise. Smooth physical motion is not included.
    pub last_step_y: f32,
    prev_contacts: Vec<Contact>,
    /// True when this tick had a real walkable contact (or static edge grip).
    supported: bool,
    step_hold: u32,
    ledge_grace: u32,
}

impl Walker {
    pub fn new(cfg: WalkConfig, pos: Vec3) -> Self {
        Self {
            cfg,
            pos,
            vel: Vec3::ZERO,
            grounded: false,
            sprinting: false,
            contacts: Vec::new(),
            last_step_y: 0.0,
            prev_contacts: Vec::new(),
            supported: false,
            step_hold: 0,
            ledge_grace: 0,
        }
    }

    /// One fixed physics tick.
    /// `wish` is the desired horizontal direction (length 0..=1, y ignored).
    /// `jump` must be true only on the tick where the jump is requested (press edge, not held).
    pub fn tick(&mut self, tris: &[Tri], dt: f32, wish: Vec3, jump: bool) {
        let cfg = self.cfg;
        let step_wall = ContactResponse {
            restitution: 0.0,
            max_bounce: 0.0,
            bounce_threshold: f32::INFINITY,
            ..cfg.wall
        };
        let was_supported = self.supported;
        let holding = self.step_hold > 0;
        let ground_like = self.grounded || holding;
        let start = self.pos;
        let foot_y = start.y - cfg.radius;
        self.last_step_y = 0.0;

        // Jump only from the ground (grounded comes from the previous tick).
        let jumped = jump && self.grounded;
        if jumped {
            self.vel.y = cfg.jump_speed;
            self.grounded = false;
            self.supported = false;
            self.step_hold = 0;
            self.ledge_grace = 0;
        }

        // Horizontal inertia only: it never touches vel.y. In the air it applies only while a direction
        // is pressed, so releasing the stick does not brake the body like an invisible wall.
        let wants_move = wish.x * wish.x + wish.z * wish.z > 1e-6;
        let horizontal = Vec3::new(self.vel.x, 0.0, self.vel.z);
        let horiz_speed_before = horizontal.len();
        let top_speed = if self.sprinting {
            cfg.sprint_speed
        } else {
            cfg.max_speed
        };
        let target = Vec3::new(wish.x, 0.0, wish.z) * top_speed;
        let h = if ground_like && !jumped {
            let rate = if wants_move { cfg.accel } else { cfg.decel };
            accelerate_toward(horizontal, target, rate, dt)
        } else if wants_move {
            accelerate_toward(horizontal, target, cfg.air_accel, dt)
        } else {
            horizontal
        };
        self.vel = Vec3::new(h.x, self.vel.y, h.z);

        // Friction of sustained contacts BEFORE moving (floor has friction 0, so it is skipped).
        std::mem::swap(&mut self.prev_contacts, &mut self.contacts);
        self.vel = pre_move_friction_by(self.vel, &self.prev_contacts, |c| {
            pick(&cfg, &step_wall, foot_y, c)
        });

        // Gravity AFTER the friction step on purpose.
        self.vel.y -= cfg.gravity * dt;

        let vel_move = self.vel;
        let moved = move_sphere(
            tris,
            start,
            cfg.radius,
            vel_move,
            dt,
            &self.prev_contacts,
            &mut self.contacts,
        );
        let supported_now = self.contacts.iter().any(|c| c.normal.y > cfg.walkable_normal_y);

        // Static edge grip: standing still on a step edge must not roll the sphere off the ledge.
        if !wants_move
            && !jumped
            && horiz_speed_before < EDGE_GRIP_MAX_SPEED
            && self.contacts.iter().any(|c| c.normal.y > EDGE_GRIP_MIN_NORMAL_Y && c.normal.y < 0.99)
        {
            self.pos = start;
            self.vel = Vec3::ZERO;
            self.supported = true;
            self.grounded = true;
            self.ledge_grace = 0;
            self.step_hold = 0;
            return;
        }

        // Step up: blocked by something non-walkable while supported and moving.
        if was_supported && !jumped && !holding {
            let blocked = self
                .contacts
                .iter()
                .any(|c| c.normal.y <= cfg.walkable_normal_y && c.impact_speed > BLOCK_MIN_SPEED);
            if blocked {
                let vel_h = Vec3::new(vel_move.x, 0.0, vel_move.z);
                if let Some((pos, gain)) =
                    try_step_up(&cfg, tris, start, vel_h, dt, horizontal_len(start, moved))
                {
                    let speed_ratio = (vel_h.len() / cfg.max_speed.max(1e-4)).clamp(0.0, 1.0);
                    let stumble = FAST_STEP_STUMBLE_LOSS * speed_ratio * speed_ratio;
                    let keep = (1.0 - cfg.step_speed_loss_per_meter * gain - stumble).clamp(0.4, 1.0);
                    self.pos = pos;
                    self.vel = Vec3::new(vel_h.x * keep, 0.0, vel_h.z * keep);
                    self.contacts.clear();
                    self.grounded = false;
                    self.supported = false;
                    self.ledge_grace = 0;
                    self.step_hold = (STEP_HOLD_TIME / dt).ceil() as u32;
                    self.last_step_y = gain;
                    return;
                }
            }
        }

        // Normal response.
        self.pos = moved;
        let vy_before = self.vel.y;
        self.vel = respond_by(self.vel, &self.contacts, |c| {
            pick(&cfg, &step_wall, foot_y, c)
        });
        // Edge guard: an edge or seam must not launch the body upward when no jump was requested.
        if ground_like && !jumped && vy_before <= 0.0 && self.vel.y > 0.0 {
            self.vel.y = 0.0;
        }

        if supported_now {
            self.supported = true;
            self.grounded = true;
            self.ledge_grace = 0;
            self.step_hold = 0;
            return;
        }
        self.supported = false;

        if self.step_hold > 0 {
            self.step_hold -= 1;
            self.grounded = false;
            return;
        }

        // Lost support while grounded (ledge): keep `grounded` for a short grace period and snap down as
        // soon as a pure vertical drop onto a walkable surface is possible.
        if was_supported && !jumped && self.ledge_grace == 0 {
            self.ledge_grace = (LEDGE_GRACE_TIME / dt).ceil() as u32;
        }
        if self.ledge_grace > 0 {
            if let Some((pos, drop, contacts)) = vertical_snap(&cfg, tris, self.pos) {
                self.pos = pos;
                self.vel.y = 0.0;
                self.contacts = contacts;
                self.supported = true;
                self.grounded = true;
                self.ledge_grace = 0;
                self.last_step_y = -drop;
            } else {
                self.ledge_grace -= 1;
                self.grounded = self.ledge_grace > 0;
            }
        } else {
            self.grounded = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::build_demo_room;

    const DT: f32 = 1.0 / 120.0;

    fn quad(out: &mut Vec<Tri>, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        out.push(Tri::new(a, b, c, [0, 0, 0]));
        out.push(Tri::new(a, c, d, [0, 0, 0]));
    }

    fn push_block(out: &mut Vec<Tri>, min: Vec3, max: Vec3) {
        let (x0, y0, z0) = (min.x, min.y, min.z);
        let (x1, y1, z1) = (max.x, max.y, max.z);
        quad(out, Vec3::new(x0, y0, z1), Vec3::new(x1, y0, z1), Vec3::new(x1, y1, z1), Vec3::new(x0, y1, z1));
        quad(out, Vec3::new(x1, y0, z0), Vec3::new(x0, y0, z0), Vec3::new(x0, y1, z0), Vec3::new(x1, y1, z0));
        quad(out, Vec3::new(x1, y0, z1), Vec3::new(x1, y0, z0), Vec3::new(x1, y1, z0), Vec3::new(x1, y1, z1));
        quad(out, Vec3::new(x0, y0, z0), Vec3::new(x0, y0, z1), Vec3::new(x0, y1, z1), Vec3::new(x0, y1, z0));
        quad(out, Vec3::new(x0, y1, z0), Vec3::new(x0, y1, z1), Vec3::new(x1, y1, z1), Vec3::new(x1, y1, z0));
        quad(out, Vec3::new(x0, y0, z0), Vec3::new(x1, y0, z0), Vec3::new(x1, y0, z1), Vec3::new(x0, y0, z1));
    }

    fn flat_floor() -> Vec<Tri> {
        let mut t = Vec::new();
        quad(
            &mut t,
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(-10.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, -10.0),
        );
        t
    }

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
        let mut w = Walker::new(cfg, Vec3::new(5.0, cfg.radius, -9.6));
        let wish = Vec3::new(1.0, 0.0, -2.0);
        let wish = wish * (1.0 / wish.len());

        let mut x_ref = 0.0;
        for tick in 0..200 {
            w.tick(&tris, DT, wish, false);
            if tick == 40 {
                x_ref = w.pos.x;
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
                let expected = -(vy_before - cfg.gravity * DT);
                assert_eq!(w.contacts.len(), 1);
                assert!(w.contacts[0].fresh, "first touch must be fresh");
                assert!((w.contacts[0].impact_speed - expected).abs() < 1e-3);
                let free_fall = (2.0 * cfg.gravity * 0.65).sqrt();
                assert!((w.contacts[0].impact_speed - free_fall).abs() < 0.15);
                assert!(w.vel.y.abs() < 1e-3, "vertical speed after landing: {}", w.vel.y);
                landed = true;
                break;
            }
        }
        assert!(landed, "the body never reached the floor");

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

        w.tick(&tris, DT, Vec3::ZERO, true);
        assert!((w.vel.y - (expected - cfg.gravity * DT)).abs() < 1e-3);
    }

    #[test]
    fn releasing_the_stick_in_the_air_keeps_horizontal_momentum() {
        let tris = flat_floor();
        let cfg = WalkConfig::default();
        let mut w = Walker::new(cfg, Vec3::new(0.0, 3.0, 0.0));
        w.vel = Vec3::new(4.0, 0.0, 0.0);
        for _ in 0..20 {
            w.tick(&tris, DT, Vec3::ZERO, false);
        }
        assert!((w.vel.x - 4.0).abs() < 1e-4, "air braking: vx = {}", w.vel.x);
    }

    #[test]
    fn climbs_a_small_step_without_getting_stuck() {
        let cfg = WalkConfig::default();
        let mut tris = flat_floor();
        push_block(&mut tris, Vec3::new(2.0, 0.0, -2.0), Vec3::new(6.0, 0.20, 2.0));
        let mut w = Walker::new(cfg, Vec3::new(0.0, cfg.radius, 0.0));
        let wish = Vec3::new(1.0, 0.0, 0.0);

        let mut lifts = 0;
        for _ in 0..300 {
            let vx_before = w.vel.x;
            w.tick(&tris, DT, wish, false);
            if w.last_step_y > 0.0 {
                lifts += 1;
                assert!((w.last_step_y - 0.20).abs() < 0.02, "gain {}", w.last_step_y);
                assert!(w.vel.x < vx_before, "no speed cost");
                assert!(w.vel.x > 0.8 * vx_before, "too much speed cost");
            }
            if lifts > 0 && w.pos.x > 3.0 {
                break;
            }
        }
        assert_eq!(lifts, 1, "exactly one step-up expected");
        assert!(w.grounded);
        assert!((w.pos.y - (cfg.radius + 0.20)).abs() < 0.01, "height {}", w.pos.y);
    }

    #[test]
    fn does_not_climb_a_wall_higher_than_max_step() {
        let cfg = WalkConfig::default();
        let mut tris = flat_floor();
        push_block(&mut tris, Vec3::new(2.0, 0.0, -2.0), Vec3::new(6.0, 0.60, 2.0));
        let mut w = Walker::new(cfg, Vec3::new(0.0, cfg.radius, 0.0));
        let wish = Vec3::new(1.0, 0.0, 0.0);

        for tick in 0..250 {
            w.tick(&tris, DT, wish, false);
            assert_eq!(w.last_step_y, 0.0, "unexpected step at tick {tick}");
        }
        assert!(w.pos.x < 2.0 - cfg.radius + 0.01, "x = {}", w.pos.x);
        assert!(w.pos.x > 2.0 - cfg.radius - 0.1, "bounced away: x = {}", w.pos.x);
        assert!(w.pos.y < cfg.radius + 0.01, "climbed: y = {}", w.pos.y);
    }

    #[test]
    fn walking_off_a_small_step_keeps_grounded_and_reports_the_drop() {
        let cfg = WalkConfig::default();
        let mut tris = flat_floor();
        push_block(&mut tris, Vec3::new(-6.0, 0.0, -2.0), Vec3::new(2.0, 0.20, 2.0));
        let mut w = Walker::new(cfg, Vec3::new(-1.0, cfg.radius + 0.20, 0.0));
        let wish = Vec3::new(1.0, 0.0, 0.0);

        let mut drops = 0;
        for tick in 0..220 {
            w.tick(&tris, DT, wish, false);
            if tick >= 5 {
                assert!(w.grounded, "grounded at tick {tick} (x = {})", w.pos.x);
            }
            if w.last_step_y < 0.0 {
                drops += 1;
                assert!(
                    w.last_step_y >= -cfg.max_step_height && w.last_step_y < -0.05,
                    "drop {}",
                    w.last_step_y
                );
            }
        }
        assert_eq!(drops, 1, "exactly one step-down expected");
        assert!((w.pos.y - cfg.radius).abs() < 0.01, "height {}", w.pos.y);
    }

    #[test]
    fn low_ceiling_over_the_step_blocks_the_climb() {
        let cfg = WalkConfig::default();
        let mut tris = flat_floor();
        push_block(&mut tris, Vec3::new(2.0, 0.0, -2.0), Vec3::new(6.0, 0.20, 2.0));
        push_block(&mut tris, Vec3::new(2.0, 0.80, -2.0), Vec3::new(6.0, 1.20, 2.0));
        let mut w = Walker::new(cfg, Vec3::new(0.0, cfg.radius, 0.0));
        let wish = Vec3::new(1.0, 0.0, 0.0);

        for tick in 0..250 {
            w.tick(&tris, DT, wish, false);
            assert_eq!(w.last_step_y, 0.0, "unexpected step at tick {tick}");
        }
        assert!(w.pos.x < 2.0, "x = {}", w.pos.x);
        assert!(w.pos.y < cfg.radius + 0.02, "climbed: y = {}", w.pos.y);
    }

    #[test]
    fn standing_near_step_edge_does_not_slip() {
        let cfg = WalkConfig::default();
        let mut tris = flat_floor();
        push_block(&mut tris, Vec3::new(-4.0, 0.0, -2.0), Vec3::new(2.0, 0.50, 2.0));
        // Sphere center slightly past the edge (x = 2.15), resting on the corner with tilted normal.
        let dy = (cfg.radius * cfg.radius - 0.15_f32 * 0.15_f32).sqrt();
        let start = Vec3::new(2.15, 0.50 + dy, 0.0);
        let mut w = Walker::new(cfg, start);

        for tick in 0..120 {
            w.tick(&tris, DT, Vec3::ZERO, false);
            assert!(
                (w.pos - start).len() < 1e-3,
                "slipped at tick {tick}: pos = {:?}",
                w.pos
            );
            assert!(w.grounded, "lost grounded at tick {tick}");
        }
    }

    #[test]
    fn low_step_does_not_bounce_at_high_speed() {
        let cfg = WalkConfig::default();
        let mut tris = flat_floor();
        // Step with a low ceiling so step-up is blocked and the riser takes the full impact.
        push_block(&mut tris, Vec3::new(2.0, 0.0, -2.0), Vec3::new(6.0, 0.20, 2.0));
        push_block(&mut tris, Vec3::new(2.0, 0.80, -2.0), Vec3::new(6.0, 1.20, 2.0));
        let mut w = Walker::new(cfg, Vec3::new(1.5, cfg.radius, 0.0));
        w.vel = Vec3::new(8.0, 0.0, 0.0);

        for tick in 0..30 {
            w.tick(&tris, DT, Vec3::new(1.0, 0.0, 0.0), false);
            assert!(w.vel.x > -0.1, "bounced off step at tick {tick}: vx = {}", w.vel.x);
        }
    }

    #[test]
    fn tall_wall_still_bounces_at_high_speed() {
        let cfg = WalkConfig::default();
        let mut tris = flat_floor();
        push_block(&mut tris, Vec3::new(2.0, 0.0, -2.0), Vec3::new(6.0, 2.0, 2.0));
        let mut w = Walker::new(cfg, Vec3::new(1.5, cfg.radius, 0.0));
        w.vel = Vec3::new(8.0, 0.0, 0.0);

        let mut bounced = false;
        for _ in 0..20 {
            w.tick(&tris, DT, Vec3::ZERO, false);
            if w.vel.x < -0.1 {
                bounced = true;
                break;
            }
        }
        assert!(bounced, "expected bounce off tall wall");
    }
    #[test]
    fn sprinting_reaches_sprint_speed() {
        let tris = flat_floor();
        let cfg = WalkConfig::default();
        let mut w = Walker::new(cfg, Vec3::new(0.0, cfg.radius, 0.0));
        w.sprinting = true;
        let wish = Vec3::new(1.0, 0.0, 0.0);

        for _ in 0..80 {
            w.tick(&tris, DT, wish, false);
        }
        assert!(
            (w.vel.x - cfg.sprint_speed).abs() < 1e-3,
            "sprint speed: {}",
            w.vel.x
        );
    }
}