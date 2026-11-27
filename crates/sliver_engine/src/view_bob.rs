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
/// Downward velocity impulse per meter of step climbed at full speed (~1.2 cm peak dip on 0.20 m).
const STUMBLE_PER_METER: f32 = 3.6;

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
        let sprint_ratio = ((speed - max_speed) / max_speed.max(1e-4)).clamp(0.0, 1.0);

        let fade = 1.0 - (-dt / AMOUNT_TAU).exp();
        let target_amount = if grounded { speed_ratio } else { 0.0 };
        let target_sprint = if grounded { sprint_ratio } else { 0.0 };
        let target_breath = if grounded { 1.0 - speed_ratio } else { 0.0 };

        self.amount += (target_amount - self.amount) * fade;
        self.sprint_blend += (target_sprint - self.sprint_blend) * fade;
        self.breath_amount += (target_breath - self.breath_amount) * fade;

        if grounded {
            // Al correr se alarga la zancada: frena la subida de frecuencia y evita la vibración.
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
    }

    /// Camera positional offset relative to the eye position; `right` is the camera's right vector.
    pub fn offset(&self, right: Vec3) -> Vec3 {
        let depth = 1.0 + 0.8 * self.sprint_blend;

        // Arco suave arriba, impacto marcado abajo (centrado en 0) + micro-variación no armónica.
        let base_y = self.phase.sin().abs() * 2.0 - 1.0;
        let organic_y = (self.phase * ORGANIC_FREQ).sin() * ORGANIC_MIX;
        let walk_y = (base_y + organic_y) * AMP_Y * self.amount * depth;

        let breath_y = self.breath_phase.sin() * BREATH_AMP * self.breath_amount;

        let y = walk_y + breath_y + self.step_offset_y + self.stumble_y;
        let x = self.phase.sin() * AMP_X * self.amount;
        Vec3::new(0.0, y, 0.0) + right * x
    }

    /// Subtle angular head micro-variations `(yaw_offset, pitch_offset)` in radians.
    /// Apply to a temporary render camera; never accumulate into the player's base look angles.
    pub fn angles(&self) -> (f32, f32) {
        let depth = 1.0 + 0.5 * self.sprint_blend;

        // Giro muy sutil de cabeza izquierda/derecha desfasado respecto al paso + leve deriva orgánica.
        let yaw_step = (self.phase + 0.6).cos();
        let yaw_drift = (self.phase * 0.53).sin() * 0.35;
        let yaw = (yaw_step + yaw_drift) * HEAD_YAW_AMP * self.amount * depth;

        // Cabeceo mínimo en cada pisada + inclinación al tropezar + inercia al acelerar/frenar.
        let pitch_step = (2.0 * self.phase + 0.4).sin();
        let pitch = pitch_step * HEAD_PITCH_AMP * self.amount * depth
            + self.stumble_y * 0.4
            + self.accel_pitch;

        (yaw, pitch)
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
        assert!(
            b.offset(Vec3::new(1.0, 0.0, 0.0)).y < -0.15,
            "camera must lag below the new height"
        );
        for _ in 0..60 {
            b.tick(DT, 0.0, 5.5, true, 0.0);
        }
        assert!(b.step_offset_y.abs() < 1e-3);
        assert!(b.offset(Vec3::new(1.0, 0.0, 0.0)).y.abs() <= BREATH_AMP + 1e-3);
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
