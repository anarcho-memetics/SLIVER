//! Camera view-bob, head micro-rotation, and step smoothing (game side).
//! Pure presentation: it never feeds back into physics.

use sliver_math::Vec3;
use std::f32::consts::TAU;

/// Ground distance covered by one full left-right stride (~3.2 Hz vertical at 5.5 m/s).
const STRIDE_LENGTH: f32 = 3.4;
const AMP_Y: f32 = 0.015;
const AMP_X: f32 = 0.005;
/// Non-harmonic secondary wave to break the mechanical metronome feel on long runs.
const ORGANIC_FREQ: f32 = 1.3;
const ORGANIC_MIX: f32 = 0.12;
/// Subtle idle breathing when standing still on the ground.
const BREATH_HZ: f32 = 0.4;
const BREATH_AMP: f32 = 0.0025;
/// Subtle head micro-rotation while walking (radians: ~0.35° yaw, ~0.20° pitch).
const HEAD_YAW_AMP: f32 = 0.006;
const HEAD_PITCH_AMP: f32 = 0.0035;
/// Subtle pitch inertia when accelerating or braking hard (radians per m/s²).
const ACCEL_PITCH_GAIN: f32 = 0.00045;
const ACCEL_TAU: f32 = 0.10;
/// Time constant of the bob amount (fade in when moving, fade out when stopping or airborne).
const AMOUNT_TAU: f32 = 0.08;
/// Time constant of the step smoothing: ~95% settled in about 3 * tau = 135 ms.
const STEP_TAU: f32 = 0.045;
/// Critically damped spring frequency for the fast step-up stumble dip.
const STUMBLE_OMEGA: f32 = 22.0;
/// Downward velocity impulse per meter of step climbed at full speed (~0.8 cm measured peak dip on 0.20 m).
const STUMBLE_PER_METER: f32 = 3.6;
/// Pre-jump knee-dip depth (meters) during the brief windup.
const JUMP_CROUCH_DEPTH: f32 = 0.035;
/// Smoothing time constant for the pre-jump head pitch nod (relaxes smoothly into ascent).
const CROUCH_PITCH_TAU: f32 = 0.035;
/// Critically damped spring frequency for takeoff and landing absorption.
const LAND_SPRING_OMEGA: f32 = 12.5;
/// Downward spring kick per m/s of landing impact.
const LAND_IMPULSE_GAIN: f32 = 0.17;

#[derive(Clone, Copy, Debug, Default)]
pub struct JumpFeedback {
    pub crouch: f32,
    pub just_jumped: bool,
    pub landing_speed: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ViewBob {
    phase: f32,
    breath_phase: f32,
    amount: f32,
    sprint_blend: f32,
    breath_amount: f32,
    prev_speed: f32,
    accel_pitch: f32,
    step_offset_y: f32,
    stumble_y: f32,
    stumble_vel: f32,
    crouch_y: f32,
    crouch_pitch: f32,
    jump_y: f32,
    jump_vel: f32,
}

impl ViewBob {
    pub fn new() -> Self {
        Self::default()
    }

    /// Call once per fixed physics tick.
    /// `last_step_y` is the vertical displacement teleported by a step this tick (see `Walker`).
    pub fn tick(
        &mut self,
        dt: f32,
        speed: f32,
        max_speed: f32,
        grounded: bool,
        last_step_y: f32,
        jump: JumpFeedback,
    ) {
        // The body jumped by `last_step_y`; the camera stays where it was and catches up smoothly.
        self.step_offset_y -= last_step_y;
        self.step_offset_y *= (-dt / STEP_TAU).exp();

        let speed_ratio = (speed / max_speed.max(1e-4)).clamp(0.0, 1.0);
        let sprint_ratio = ((speed - max_speed) / max_speed.max(1e-4)).clamp(0.0, 1.0);

        let fade = 1.0 - (-dt / AMOUNT_TAU).exp();
        let target_amount = if grounded { speed_ratio } else { 0.0 };
        let target_sprint = if grounded { sprint_ratio } else { 0.0 };
        let target_breath = if grounded { 1.0 - speed_ratio } else { 0.0 };

        self.amount += (target_amount - self.amount) * fade;
        self.sprint_blend += (target_sprint - self.sprint_blend) * fade;
        self.breath_amount += (target_breath - self.breath_amount) * fade;

        if grounded {
            // Sprinting lengthens the stride: it slows the frequency growth and avoids a vibration feel.
            let stride = STRIDE_LENGTH * (1.0 + 1.15 * self.sprint_blend);
            self.phase += speed * dt / stride * TAU;
        }
        self.breath_phase = (self.breath_phase + BREATH_HZ * dt * TAU) % TAU;

        // Acceleration/braking pitch inertia.
        let raw_accel = if grounded && dt > 1e-5 {
            ((speed - self.prev_speed) / dt).clamp(-45.0, 25.0)
        } else {
            0.0
        };
        self.prev_speed = speed;
        let target_accel_pitch = raw_accel * ACCEL_PITCH_GAIN;
        self.accel_pitch +=
            (target_accel_pitch - self.accel_pitch) * (1.0 - (-dt / ACCEL_TAU).exp());

        // Fast step-up stumble: starts at 0 (no tick-0 pop), dips slightly, then recovers smoothly.
        if last_step_y > 0.0 {
            self.stumble_vel -= STUMBLE_PER_METER * last_step_y * speed_ratio;
        }
        let acc = -STUMBLE_OMEGA * STUMBLE_OMEGA * self.stumble_y
            - 2.0 * STUMBLE_OMEGA * self.stumble_vel;
        self.stumble_vel += acc * dt;
        self.stumble_y += self.stumble_vel * dt;

        // Pre-jump knee compression + landing impact absorption.
        // Transfer remaining crouch into the spring when windup ends so takeoff never snaps.
        if jump.crouch > 0.0 {
            let u = jump.crouch;
            // Smooth dip (deepest at u = 2/3) that already extends upward into takeoff.
            let dip = 6.75 * u * u * (1.0 - u);
            self.crouch_y = -dip * JUMP_CROUCH_DEPTH;
        } else if self.crouch_y != 0.0 {
            self.jump_y += self.crouch_y;
            self.crouch_y = 0.0;
        }
        let target_crouch_pitch = self.crouch_y * 0.55;
        self.crouch_pitch +=
            (target_crouch_pitch - self.crouch_pitch) * (1.0 - (-dt / CROUCH_PITCH_TAU).exp());
        if jump.just_jumped {
            // Cancel any leftover downward landing velocity if jumping right after touching down.
            self.jump_vel = self.jump_vel.max(0.0);
        }
        if jump.landing_speed > 0.0 {
            self.jump_vel -= (jump.landing_speed * LAND_IMPULSE_GAIN).min(1.7);
        }
        let j_acc = -LAND_SPRING_OMEGA * LAND_SPRING_OMEGA * self.jump_y
            - 2.0 * LAND_SPRING_OMEGA * self.jump_vel;
        self.jump_vel += j_acc * dt;
        self.jump_y += self.jump_vel * dt;
    }

    /// Camera positional offset relative to the eye position; `right` is the camera's right vector.
    pub fn offset(&self, right: Vec3) -> Vec3 {
        let depth = 1.0 + 0.8 * self.sprint_blend;

        // Smooth arc on top, sharp impact at the bottom (centered on 0) + non-harmonic micro-variation.
        let base_y = self.phase.sin().abs() * 2.0 - 1.0;
        let organic_y = (self.phase * ORGANIC_FREQ).sin() * ORGANIC_MIX;
        let walk_y = (base_y + organic_y) * AMP_Y * self.amount * depth;

        let breath_y = self.breath_phase.sin() * BREATH_AMP * self.breath_amount;

        let y =
            walk_y + breath_y + self.step_offset_y + self.stumble_y + self.crouch_y + self.jump_y;
        let x = self.phase.sin() * AMP_X * self.amount;
        Vec3::new(0.0, y, 0.0) + right * x
    }

    /// Subtle angular head micro-variations `(yaw_offset, pitch_offset)` in radians.
    /// Apply to a temporary render camera; never accumulate into the player's base look angles.
    pub fn angles(&self) -> (f32, f32) {
        let depth = 1.0 + 0.5 * self.sprint_blend;

        // Very subtle left/right head turn, out of phase with the step + slight organic drift.
        let yaw_step = (self.phase + 0.6).cos();
        let yaw_drift = (self.phase * 0.53).sin() * 0.35;
        let yaw = (yaw_step + yaw_drift) * HEAD_YAW_AMP * self.amount * depth;

        // Minimal pitch on each footfall + tilt when stumbling + inertia when accelerating/braking.
        let pitch_step = (2.0 * self.phase + 0.4).sin();
        let pitch = pitch_step * HEAD_PITCH_AMP * self.amount * depth
            + self.stumble_y * 0.4
            + self.jump_y * 0.35
            + self.crouch_pitch
            + self.accel_pitch;

        (yaw, pitch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 120.0;

    fn right() -> Vec3 {
        Vec3::new(1.0, 0.0, 0.0)
    }

    fn none() -> JumpFeedback {
        JumpFeedback::default()
    }

    /// Phase advance (full strides per second) after letting the bob amount settle.
    fn strides_per_second(speed: f32) -> f32 {
        let mut b = ViewBob::new();
        for _ in 0..240 {
            b.tick(DT, speed, 5.5, true, 0.0, none());
        }
        let start = b.phase;
        for _ in 0..120 {
            b.tick(DT, speed, 5.5, true, 0.0, none());
        }
        (b.phase - start) / TAU
    }

    #[test]
    fn step_up_starts_below_and_settles_to_zero() {
        let mut b = ViewBob::new();
        b.tick(DT, 0.0, 5.5, true, 0.20, JumpFeedback::default());
        assert!(
            b.offset(Vec3::new(1.0, 0.0, 0.0)).y < -0.15,
            "camera must lag below the new height"
        );
        for _ in 0..60 {
            b.tick(DT, 0.0, 5.5, true, 0.0, JumpFeedback::default());
        }
        assert!(b.step_offset_y.abs() < 1e-3);
        assert!(b.offset(Vec3::new(1.0, 0.0, 0.0)).y.abs() <= BREATH_AMP + 1e-3);
    }

    #[test]
    fn no_bob_in_the_air_or_standing_still() {
        let mut b = ViewBob::new();
        for _ in 0..120 {
            b.tick(DT, 5.5, 5.5, true, 0.0, JumpFeedback::default());
        }
        assert!(b.amount > 0.9);
        for _ in 0..120 {
            b.tick(DT, 5.5, 5.5, false, 0.0, JumpFeedback::default());
        }
        assert!(b.offset(Vec3::new(1.0, 0.0, 0.0)).len() < 1e-3);
    }

    #[test]
    fn step_up_stumble_is_continuous_dips_and_recovers() {
        let mut b = ViewBob::new();
        b.tick(DT, 5.5, 5.5, false, 0.20, none());
        // The dip starts from a velocity impulse, not from a position jump.
        assert!(b.stumble_y.abs() < 0.006, "tick-0 pop: {}", b.stumble_y);

        let mut lowest = b.stumble_y;
        for _ in 0..120 {
            b.tick(DT, 5.5, 5.5, false, 0.0, none());
            lowest = lowest.min(b.stumble_y);
        }
        // About 0.8 cm at full speed on a 0.20 m step.
        assert!(lowest < -0.005 && lowest > -0.012, "peak dip: {lowest}");
        assert!(b.stumble_y.abs() < 1e-4, "must recover: {}", b.stumble_y);
    }

    #[test]
    fn step_up_while_standing_has_no_stumble() {
        let mut b = ViewBob::new();
        for i in 0..120 {
            let step = if i == 0 { 0.20 } else { 0.0 };
            b.tick(DT, 0.0, 5.5, true, step, none());
        }
        assert_eq!(b.stumble_y, 0.0);
    }

    #[test]
    fn crouch_windup_is_continuous_and_returns_to_zero_after_takeoff() {
        let mut b = ViewBob::new();
        let (mut prev, mut lowest, mut max_step) = (0.0_f32, 0.0_f32, 0.0_f32);
        for i in 1..=60 {
            let u = i as f32 / 60.0;
            b.tick(
                DT,
                0.0,
                5.5,
                true,
                0.0,
                JumpFeedback {
                    crouch: u,
                    ..none()
                },
            );
            let y = b.offset(right()).y;
            max_step = max_step.max((y - prev).abs());
            lowest = lowest.min(y);
            prev = y;
        }
        // Takeoff: the windup ends and the jump spring takes over.
        b.tick(
            DT,
            0.0,
            5.5,
            true,
            0.0,
            JumpFeedback {
                just_jumped: true,
                ..none()
            },
        );
        max_step = max_step.max((b.offset(right()).y - prev).abs());

        assert!(lowest < -0.025 && lowest > -0.036, "crouch depth: {lowest}");
        assert!(max_step < 0.006, "visible snap: {max_step}");
        for _ in 0..120 {
            b.tick(DT, 0.0, 5.5, false, 0.0, none());
        }
        assert!(b.offset(right()).y.abs() < 1e-3);
    }

    #[test]
    fn cancelled_windup_relaxes_without_a_snap() {
        let mut b = ViewBob::new();
        let mut prev = 0.0_f32;
        for i in 1..=30 {
            let u = i as f32 / 60.0;
            b.tick(
                DT,
                0.0,
                5.5,
                true,
                0.0,
                JumpFeedback {
                    crouch: u,
                    ..none()
                },
            );
            prev = b.offset(right()).y;
        }
        // Windup interrupted at u = 0.5: the remaining dip moves into the spring.
        let mut max_step = 0.0_f32;
        for _ in 0..120 {
            b.tick(DT, 0.0, 5.5, true, 0.0, none());
            let y = b.offset(right()).y;
            max_step = max_step.max((y - prev).abs());
            prev = y;
        }
        assert!(max_step < 0.006, "visible snap: {max_step}");
        assert!(b.offset(right()).y.abs() <= BREATH_AMP + 1e-3);
    }

    #[test]
    fn landing_impact_dips_and_settles() {
        let mut b = ViewBob::new();
        let land = JumpFeedback {
            landing_speed: 6.0,
            ..none()
        };
        b.tick(DT, 0.0, 5.5, false, 0.0, land);
        let mut lowest = b.offset(right()).y;
        for _ in 0..120 {
            b.tick(DT, 0.0, 5.5, false, 0.0, none());
            lowest = lowest.min(b.offset(right()).y);
        }
        assert!(lowest < -0.015 && lowest > -0.03, "landing dip: {lowest}");
        assert!(b.offset(right()).y.abs() < 1e-3, "must settle");
    }

    #[test]
    fn landing_impulse_is_capped() {
        let mut b = ViewBob::new();
        let crash = JumpFeedback {
            landing_speed: 100.0,
            ..none()
        };
        b.tick(DT, 0.0, 5.5, false, 0.0, crash);
        let mut lowest = b.offset(right()).y;
        for _ in 0..120 {
            b.tick(DT, 0.0, 5.5, false, 0.0, none());
            lowest = lowest.min(b.offset(right()).y);
        }
        assert!(lowest > -0.06, "uncapped landing dip: {lowest}");
    }

    #[test]
    fn takeoff_cancels_leftover_landing_velocity() {
        let mut b = ViewBob::new();
        let land = JumpFeedback {
            landing_speed: 6.0,
            ..none()
        };
        b.tick(DT, 0.0, 5.5, true, 0.0, land);
        assert!(b.jump_vel < 0.0);
        let takeoff = JumpFeedback {
            just_jumped: true,
            ..none()
        };
        b.tick(DT, 0.0, 5.5, true, 0.0, takeoff);
        assert!(b.jump_vel >= 0.0, "still sinking: {}", b.jump_vel);
    }

    #[test]
    fn head_angles_are_zero_at_rest_and_subtle_while_sprinting() {
        assert_eq!(ViewBob::new().angles(), (0.0, 0.0));

        let mut b = ViewBob::new();
        for _ in 0..120 {
            b.tick(DT, 0.0, 5.5, true, 0.0, none());
        }
        let (yaw, pitch) = b.angles();
        assert!(yaw.abs() < 1e-5 && pitch.abs() < 1e-5);

        let (mut max_yaw, mut max_pitch) = (0.0_f32, 0.0_f32);
        for _ in 0..600 {
            b.tick(DT, 9.5, 5.5, true, 0.0, none());
            let (y, p) = b.angles();
            max_yaw = max_yaw.max(y.abs());
            max_pitch = max_pitch.max(p.abs());
        }
        let (yaw_deg, pitch_deg) = (max_yaw.to_degrees(), max_pitch.to_degrees());
        assert!(yaw_deg > 0.2 && yaw_deg < 1.0, "yaw: {yaw_deg} deg");
        assert!(
            pitch_deg > 0.05 && pitch_deg < 1.0,
            "pitch: {pitch_deg} deg"
        );
    }

    #[test]
    fn walking_cadence_matches_the_stride_length() {
        let strides = strides_per_second(5.5);
        assert!(
            (strides - 5.5 / STRIDE_LENGTH).abs() < 0.01,
            "strides/s: {strides}"
        );
    }

    #[test]
    fn sprint_lengthens_the_stride_instead_of_vibrating() {
        let walk = strides_per_second(5.5);
        let sprint = strides_per_second(9.5);
        assert!(
            sprint < walk * 1.1,
            "sprint {sprint} vs walk {walk} strides/s"
        );
        assert!(sprint > 1.3 && sprint < 1.8, "sprint strides/s: {sprint}");
        // Without the lengthening, 9.5 m/s would give ~2.8 strides/s.
        assert!(
            sprint < 9.5 / STRIDE_LENGTH * 0.7,
            "stride did not lengthen"
        );
    }
}
