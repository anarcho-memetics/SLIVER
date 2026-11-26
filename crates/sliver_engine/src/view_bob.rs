//! Camera view-bob and step smoothing (game side). Pure presentation: it never feeds back into physics.

use sliver_math::Vec3;
use std::f32::consts::TAU;

/// Ground distance covered by one full left-right stride (~3.2 Hz vertical at 5.5 m/s).
const STRIDE_LENGTH: f32 = 3.4;
const AMP_Y: f32 = 0.015;
const AMP_X: f32 = 0.008;
/// Time constant of the bob amount (fade in when moving, fade out when stopping or airborne).
const AMOUNT_TAU: f32 = 0.08;
/// Time constant of the step smoothing: ~95% settled in about 3 * tau = 135 ms.
const STEP_TAU: f32 = 0.045;
/// Critically damped spring frequency for the fast step-up stumble dip.
const STUMBLE_OMEGA: f32 = 22.0;
/// Downward velocity impulse per meter of step climbed at full speed (~1.2 cm peak dip on 0.20 m).
const STUMBLE_PER_METER: f32 = 3.6;

#[derive(Clone, Copy, Debug, Default)]
pub struct ViewBob {
    phase: f32,
    amount: f32,
    step_offset_y: f32,
    stumble_y: f32,
    stumble_vel: f32,
}

impl ViewBob {
    pub fn new() -> Self {
        Self::default()
    }

    /// Call once per fixed physics tick.
    /// `last_step_y` is the vertical displacement teleported by a step this tick (see `Walker`).
    pub fn tick(&mut self, dt: f32, speed: f32, max_speed: f32, grounded: bool, last_step_y: f32) {
        // The body jumped by `last_step_y`; the camera stays where it was and catches up smoothly.
        self.step_offset_y -= last_step_y;
        self.step_offset_y *= (-dt / STEP_TAU).exp();

        let speed_ratio = (speed / max_speed.max(1e-4)).clamp(0.0, 1.0);
        let target = if grounded { speed_ratio } else { 0.0 };
        self.amount += (target - self.amount) * (1.0 - (-dt / AMOUNT_TAU).exp());

        if grounded {
            self.phase += speed * dt / STRIDE_LENGTH * TAU;
        }

        // Fast step-up stumble: starts at 0 (no tick-0 pop), dips slightly, then recovers smoothly.
        if last_step_y > 0.0 {
            self.stumble_vel -= STUMBLE_PER_METER * last_step_y * speed_ratio;
        }
        let acc = -STUMBLE_OMEGA * STUMBLE_OMEGA * self.stumble_y
            - 2.0 * STUMBLE_OMEGA * self.stumble_vel;
        self.stumble_vel += acc * dt;
        self.stumble_y += self.stumble_vel * dt;
    }

    /// Camera offset relative to the eye position; `right` is the camera's right vector.
    pub fn offset(&self, right: Vec3) -> Vec3 {
        let y = (2.0 * self.phase).sin() * AMP_Y * self.amount
            + self.step_offset_y
            + self.stumble_y;
        let x = self.phase.sin() * AMP_X * self.amount;
        Vec3::new(0.0, y, 0.0) + right * x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 120.0;

    #[test]
    fn step_up_starts_below_and_settles_to_zero() {
        let mut b = ViewBob::new();
        b.tick(DT, 0.0, 5.5, true, 0.20);
        assert!(b.offset(Vec3::new(1.0, 0.0, 0.0)).y < -0.15, "camera must lag below the new height");
        for _ in 0..60 {
            b.tick(DT, 0.0, 5.5, true, 0.0);
        }
        assert!(b.offset(Vec3::new(1.0, 0.0, 0.0)).y.abs() < 1e-3);
    }

    #[test]
    fn no_bob_in_the_air_or_standing_still() {
        let mut b = ViewBob::new();
        for _ in 0..120 {
            b.tick(DT, 5.5, 5.5, true, 0.0);
        }
        assert!(b.amount > 0.9);
        for _ in 0..120 {
            b.tick(DT, 5.5, 5.5, false, 0.0);
        }
        assert!(b.offset(Vec3::new(1.0, 0.0, 0.0)).len() < 1e-3);
    }
}