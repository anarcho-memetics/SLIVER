#![forbid(unsafe_code)]

mod collision;
mod controller;
mod motion;
mod raster;
mod raycast;
mod scene;

use controller::{WalkConfig, Walker};
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use raster::{Camera, Framebuffer, RenderConfig};
use scene::{build_demo_room, build_spinner, load_obj, tag_slivers};
use sliver_math::Vec3;
use std::env;
use std::path::Path;
use std::time::Instant;

const W: usize = 960;
const H: usize = 540;
const THRESHOLDS: [f32; 4] = [1.0, 5.0, 12.0, 25.0];

const STEP: f32 = 1.0 / 120.0; // fixed physics timestep
const EYE_HEIGHT: f32 = 1.65;

fn main() {
    let arg_path = env::args().nth(1);
    let (mut world_tris, is_custom_obj) = match arg_path {
        Some(ref p) => match load_obj(Path::new(p)) {
            Ok(tris) => {
                println!("Loaded {p}: {} triangles", tris.len());
                (tris, true)
            }
            Err(e) => {
                eprintln!("{e}");
                return;
            }
        },
        None => (build_demo_room(), false),
    };

    let mut thresh_idx = 2usize; // default threshold: 12 deg
    let mut sliver_count = tag_slivers(&mut world_tris, THRESHOLDS[thresh_idx]);

    let mut window = Window::new(
        "sliver_engine",
        W,
        H,
        WindowOptions {
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");
    window.set_target_fps(60);

    let mut fb = Framebuffer::new(W, H);

    let walk_cfg = WalkConfig::default();
    let eye_offset = Vec3::new(0.0, EYE_HEIGHT - walk_cfg.radius, 0.0);
    let start_body = if is_custom_obj {
        Vec3::new(0.0, 1.5, 8.0)
    } else {
        Vec3::new(0.0, walk_cfg.radius, 4.2)
    };
    let mut walker = Walker::new(walk_cfg, start_body);

    let mut cam = Camera {
        pos: walker.pos + eye_offset,
        yaw: 0.0,
        pitch: -0.05,
        fov_y_rad: 68.0_f32.to_radians(),
        near: 0.12,
    };

    let mut cfg = RenderConfig {
        debug_slivers: false,
        fog_enabled: true,
        fog_color: [16, 20, 26],
        fog_start: 4.5,
        fog_end: 20.0,
        light_dir: Vec3::new(0.45, 0.82, 0.35).normalize(),
    };

    let mut step_acc = 0.0_f32;
    let mut jump_queued = false;

    let mut last = Instant::now();
    let mut elapsed = 0.0_f32;
    let mut fps_timer = 0.0_f32;
    let mut fps_frames = 0u32;
    let mut render_ms_acc = 0.0_f32;

    println!("Controls: WASD move | Space jump | Arrows/Q/E look | F1 sliver debug | F2 cycle threshold | F3 fog | F4 raycast | Esc quit");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f32().min(0.05);
        last = now;
        elapsed += dt;

        // Debug and rendering toggles
        if window.is_key_pressed(Key::F1, KeyRepeat::No) {
            cfg.debug_slivers = !cfg.debug_slivers;
        }
        if window.is_key_pressed(Key::F2, KeyRepeat::No) {
            thresh_idx = (thresh_idx + 1) % THRESHOLDS.len();
            sliver_count = tag_slivers(&mut world_tris, THRESHOLDS[thresh_idx]);
        }
        if window.is_key_pressed(Key::F3, KeyRepeat::No) {
            cfg.fog_enabled = !cfg.fog_enabled;
        }
        if window.is_key_pressed(Key::F4, KeyRepeat::No) {
            let (_, _, forward) = cam.basis();
            match raycast::raycast(&world_tris, cam.pos, forward, 100.0) {
                Some((i, hit)) => println!(
                    "Hit tri #{i} at {:.2} units (front face: {})",
                    hit.t, hit.front_face
                ),
                None => println!("No hit"),
            }
        }

        // Camera rotation
        let turn_speed = 2.0 * dt;
        if window.is_key_down(Key::Left) || window.is_key_down(Key::Q) {
            cam.yaw -= turn_speed;
        }
        if window.is_key_down(Key::Right) || window.is_key_down(Key::E) {
            cam.yaw += turn_speed;
        }
        if window.is_key_down(Key::Up) {
            cam.pitch = (cam.pitch + turn_speed * 0.75).min(1.35);
        }
        if window.is_key_down(Key::Down) {
            cam.pitch = (cam.pitch - turn_speed * 0.75).max(-1.35);
        }

        // Horizontal wish direction (normalized 0..=1)
        let (sy, cy) = cam.yaw.sin_cos();
        let fwd_xz = Vec3::new(sy, 0.0, -cy);
        let right_xz = Vec3::new(cy, 0.0, sy);

        let mut wish = Vec3::ZERO;
        if window.is_key_down(Key::W) {
            wish = wish + fwd_xz;
        }
        if window.is_key_down(Key::S) {
            wish = wish - fwd_xz;
        }
        if window.is_key_down(Key::D) {
            wish = wish + right_xz;
        }
        if window.is_key_down(Key::A) {
            wish = wish - right_xz;
        }
        let wish_len = wish.len();
        let wish_dir = if wish_len > 1e-6 {
            wish * (1.0 / wish_len)
        } else {
            Vec3::ZERO
        };

        // Press-edge jump queued so it is never lost on fast frames or double-fired at 60 FPS
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            jump_queued = true;
        }

        // Fixed-timestep walking physics.
        // `dt` is already capped at 0.05 s at the top of the loop, so at most 6 ticks run per frame.
        step_acc += dt;
        while step_acc >= STEP {
            step_acc -= STEP;
            let jump = std::mem::take(&mut jump_queued);
            walker.tick(&world_tris, STEP, wish_dir, jump);

            // ---- FUTURE GAME HOOKS GO HERE (damage, audio events, shoulder bump, camera shake) ----
            // Read `walker.contacts` inside this loop, once per physics tick.
        }

        cam.pos = walker.pos + eye_offset;

        // Render pass
        let t_render = Instant::now();
        fb.clear(cfg.fog_color);
        let mut drawn = fb.draw_triangles(&world_tris, &cam, &cfg);
        if !is_custom_obj {
            let spinner = build_spinner(elapsed * 1.1, Vec3::new(0.0, 2.5, -4.0));
            drawn += fb.draw_triangles(&spinner, &cam, &cfg);
        }
        render_ms_acc += t_render.elapsed().as_secs_f32() * 1000.0;

        window.update_with_buffer(&fb.color, W, H).unwrap();

        fps_timer += dt;
        fps_frames += 1;
        if fps_timer >= 0.35 {
            let fps = fps_frames as f32 / fps_timer;
            let avg_render_ms = render_ms_acc / fps_frames as f32;
            let mode = if cfg.debug_slivers {
                "SLIVER-DEBUG"
            } else {
                "NORMAL"
            };
            window.set_title(&format!(
                "sliver_engine | {:.0} FPS ({:.2} ms) | tris: {}/{} | slivers (<{}°): {} | [F1] {}",
                fps,
                avg_render_ms,
                drawn,
                world_tris.len(),
                THRESHOLDS[thresh_idx],
                sliver_count,
                mode
            ));
            fps_timer = 0.0;
            fps_frames = 0;
            render_ms_acc = 0.0;
        }
    }
}