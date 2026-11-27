#![forbid(unsafe_code)]

mod collision;
mod controller;
mod motion;
mod raster;
mod raycast;
mod scene;
mod view_bob;
mod wad;

use controller::{WalkConfig, Walker};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use raster::{Camera, Framebuffer, RenderConfig};
use scene::{build_demo_room, build_playground, build_spinner, load_obj, tag_slivers};
use sliver_math::Vec3;
use std::env;
use std::path::Path;
use std::time::Instant;
use view_bob::{JumpFeedback, ViewBob};

const W: usize = 960;
const H: usize = 540;
const THRESHOLDS: [f32; 4] = [1.0, 5.0, 12.0, 25.0];

const STEP: f32 = 1.0 / 120.0; // fixed physics timestep
const EYE_HEIGHT: f32 = 1.65; // eye height above the floor
const MOUSE_SENS: f32 = 0.0035;

fn main() {
    let arg_path = env::args().nth(1);
    let arg_map = env::args().nth(2);
    let mut walk_cfg = WalkConfig::default();
    let (mut world_tris, world_textures, is_external_map, body_start, start_yaw, default_fog_end) =
        match arg_path {
            Some(ref p) if p.to_ascii_lowercase().ends_with(".wad") => {
                match wad::load_wad_map(Path::new(p), arg_map.as_deref()) {
                    Ok(map) => {
                        // Doom's MAXSTEPHEIGHT is 24 map units (~0.67 m at 1/36 scale).
                        walk_cfg.max_step_height = 0.70;
                        println!(
                            "Loaded {p} ({}): {} triangles, {} textures",
                            map.name,
                            map.tris.len(),
                            map.textures.len()
                        );
                        (
                            map.tris,
                            map.textures,
                            true,
                            map.spawn_pos + Vec3::new(0.0, walk_cfg.radius, 0.0),
                            map.spawn_yaw,
                            55.0,
                        )
                    }
                    Err(e) => {
                        eprintln!("{e}");
                        return;
                    }
                }
            }
            Some(ref p) => match load_obj(Path::new(p)) {
                Ok(tris) => {
                    println!("Loaded {p}: {} triangles", tris.len());
                    (tris, Vec::new(), true, Vec3::new(0.0, 1.5, 8.0), 0.0, 20.0)
                }
                Err(e) => {
                    eprintln!("{e}");
                    return;
                }
            },
            None => (
                build_demo_room(),
                Vec::new(),
                false,
                Vec3::new(0.0, walk_cfg.radius, 4.2),
                0.0,
                20.0,
            ),
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

    let eye_offset = Vec3::new(0.0, EYE_HEIGHT - walk_cfg.radius, 0.0);
    let mut walker = Walker::new(walk_cfg, body_start);
    let mut bob = ViewBob::new();
    let mut prev_body_pos = walker.pos;
    let mut prev_bob = bob;

    let mut cam = Camera {
        pos: walker.pos + eye_offset,
        yaw: start_yaw,
        pitch: -0.05,
        fov_y_rad: 68.0_f32.to_radians(),
        near: 0.12,
    };

    let mut cfg = RenderConfig {
        debug_slivers: false,
        textures_enabled: true,
        fog_enabled: true,
        fog_color: [16, 20, 26],
        fog_start: 4.5,
        fog_end: default_fog_end,
        light_dir: Vec3::new(0.45, 0.82, 0.35).normalize(),
    };

    let mut step_acc = 0.0_f32;
    let mut jump_queued = false;
    let mut in_playground = false;
    let mut slow_mo = false;
    let mut last_mouse: Option<(f32, f32)> = None;

    let mut last = Instant::now();
    let mut elapsed = 0.0_f32;
    let mut fps_timer = 0.0_f32;
    let mut fps_frames = 0u32;
    let mut render_ms_acc = 0.0_f32;

    println!("Controls: WASD move | Shift sprint | Space jump | Mouse drag or Arrows/Q/E look | F1 sliver debug | F2 cycle threshold | F3 fog | F4 raycast | F5 playground | Esc quit");

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
        if window.is_key_pressed(Key::F5, KeyRepeat::No) && !is_external_map {
            in_playground = !in_playground;
            world_tris = if in_playground {
                build_playground()
            } else {
                build_demo_room()
            };
            sliver_count = tag_slivers(&mut world_tris, THRESHOLDS[thresh_idx]);
            let spawn = if in_playground {
                Vec3::new(0.0, walk_cfg.radius, 25.0)
            } else {
                Vec3::new(0.0, walk_cfg.radius, 4.2)
            };
            walker = Walker::new(walk_cfg, spawn);
            bob = ViewBob::new();
            prev_body_pos = walker.pos;
            prev_bob = bob;
            jump_queued = false;
            cfg.fog_end = if in_playground { 48.0 } else { 20.0 };
        }
        if window.is_key_pressed(Key::F6, KeyRepeat::No) {
            slow_mo = !slow_mo;
            println!("Slow motion: {}", if slow_mo { "ON" } else { "OFF" });
        }
        if window.is_key_pressed(Key::F7, KeyRepeat::No) {
            cfg.textures_enabled = !cfg.textures_enabled;
        }
        // Jump is a press edge, consumed by the first physics tick that runs.
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            jump_queued = true;
        }

        // Camera rotation: mouse drag (LMB or RMB) + keyboard fallback
        let mouse_look =
            window.get_mouse_down(MouseButton::Left) || window.get_mouse_down(MouseButton::Right);
        if mouse_look {
            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Pass) {
                if let Some((lx, ly)) = last_mouse {
                    cam.yaw += (mx - lx) * MOUSE_SENS;
                    cam.pitch = (cam.pitch - (my - ly) * MOUSE_SENS).clamp(-1.35, 1.35);
                }
                last_mouse = Some((mx, my));
            }
        } else {
            last_mouse = None;
        }

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

        // Desired horizontal direction from input (length 0..=1)
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

        walker.sprinting =
            window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift);

        // Fixed-timestep physics.
        step_acc += if slow_mo { dt * 0.2 } else { dt };
        while step_acc >= STEP {
            step_acc -= STEP;
            prev_body_pos = walker.pos;
            prev_bob = bob;

            walker.tick(&world_tris, STEP, wish_dir, jump_queued);
            jump_queued = false;

            let speed = Vec3::new(walker.vel.x, 0.0, walker.vel.z).len();
            bob.tick(
                STEP,
                speed,
                walker.cfg.max_speed,
                walker.grounded,
                walker.last_step_y,
                JumpFeedback {
                    crouch: walker.jump_crouch,
                    just_jumped: walker.just_jumped,
                    landing_speed: walker.last_landing_speed,
                },
            );
        }

        // Interpolate between the last two physics ticks to avoid 1-vs-2 tick stutter at 60 FPS.
        let alpha = (step_acc / STEP).clamp(0.0, 1.0);
        let body_pos = prev_body_pos + (walker.pos - prev_body_pos) * alpha;
        let (right, _, _) = cam.basis();
        let prev_off = prev_bob.offset(right);
        let curr_off = bob.offset(right);
        cam.pos = body_pos + eye_offset + prev_off + (curr_off - prev_off) * alpha;

        // Render pass
        let (prev_yaw, prev_pitch) = prev_bob.angles();
        let (curr_yaw, curr_pitch) = bob.angles();
        let bob_yaw = prev_yaw + (curr_yaw - prev_yaw) * alpha;
        let bob_pitch = prev_pitch + (curr_pitch - prev_pitch) * alpha;
        let render_cam = Camera {
            yaw: cam.yaw + bob_yaw,
            pitch: (cam.pitch + bob_pitch).clamp(-1.45, 1.45),
            ..cam
        };

        let t_render = Instant::now();
        fb.clear(cfg.fog_color);
        let mut drawn = fb.draw_triangles(&world_tris, &world_textures, &render_cam, &cfg);
        if !is_external_map {
            let spinner = build_spinner(elapsed * 1.1, Vec3::new(0.0, 2.5, -4.0));
            drawn += fb.draw_triangles(&spinner, &[], &render_cam, &cfg);
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
            } else if cfg.textures_enabled {
                "TEX"
            } else {
                "FLAT"
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
