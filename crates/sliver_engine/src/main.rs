#![forbid(unsafe_code)]

use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, ScaleMode, Window, WindowOptions};
use sliver_engine::engine::{Engine, InputState, World};
use std::env;
use std::path::Path;
use std::time::Instant;

const W: usize = 960;
const H: usize = 540;
const MOUSE_SENS: f32 = 0.0035;

fn open_window(scale_2x: bool) -> Window {
    let opts = WindowOptions {
        scale: if scale_2x { Scale::X2 } else { Scale::X1 },
        scale_mode: ScaleMode::AspectRatioStretch,
        borderless: scale_2x,
        title: !scale_2x,
        resize: !scale_2x,
        ..WindowOptions::default()
    };
    let mut win = Window::new("sliver_engine", W, H, opts).expect("Failed to create window");
    if scale_2x {
        win.set_position(0, 0);
    }
    win.set_target_fps(60);
    win
}

fn main() {
    let mut scale_2x = false;
    let mut start_sbs = false;
    let mut pos_args = Vec::new();
    for arg in env::args().skip(1) {
        if arg == "--2x" {
            scale_2x = true;
        } else if arg == "--sbs" {
            start_sbs = true;
        } else {
            pos_args.push(arg);
        }
    }

    let arg_path = pos_args.first().cloned();
    let arg_map = pos_args.get(1).cloned();
    let world = match arg_path {
        Some(ref p) if p.to_ascii_lowercase().ends_with(".wad") => {
            match World::from_wad_file(Path::new(p), arg_map.as_deref()) {
                Ok(w) => {
                    println!(
                        "Loaded {p} ({}): {} triangles, {} textures",
                        w.name,
                        w.tris.len(),
                        w.textures.len()
                    );
                    w
                }
                Err(e) => {
                    eprintln!("{e}");
                    return;
                }
            }
        }
        Some(ref p) => match World::from_obj_file(Path::new(p)) {
            Ok(w) => {
                println!("Loaded {p}: {} triangles", w.tris.len());
                w
            }
            Err(e) => {
                eprintln!("{e}");
                return;
            }
        },
        None => World::demo_room(),
    };

    let mut engine = Engine::new(W, H, world);
    engine.stereo.enabled = start_sbs;

    let mut window = open_window(scale_2x);
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut last = Instant::now();
    let mut fps_timer = 0.0_f32;
    let mut fps_frames = 0u32;
    let mut render_ms_acc = 0.0_f32;

    println!(
        "Controls: WASD move | Shift sprint | Space jump | Mouse drag or Arrows/Q/E look | F1 sliver | F2 thresh | F3 fog | F4 raycast | F5 playground | F6 slow-mo | F7 tex | F8 SBS | F10 warp | [/] IPD | F9 perf | F11 2x | Esc quit"
    );

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let raw_dt = now.duration_since(last).as_secs_f32();
        let dt = raw_dt.min(0.05);
        last = now;

        // Toggles and debug actions
        if window.is_key_pressed(Key::F1, KeyRepeat::No) {
            engine.toggle_sliver_debug();
        }
        if window.is_key_pressed(Key::F2, KeyRepeat::No) {
            engine.cycle_sliver_threshold();
        }
        if window.is_key_pressed(Key::F3, KeyRepeat::No) {
            engine.toggle_fog();
        }
        if window.is_key_pressed(Key::F4, KeyRepeat::No) {
            match engine.raycast_center() {
                Some((i, t, front)) => {
                    println!("Hit tri #{i} at {t:.2} units (front face: {front})")
                }
                None => println!("No hit"),
            }
        }
        if window.is_key_pressed(Key::F5, KeyRepeat::No) {
            engine.toggle_playground();
        }
        if window.is_key_pressed(Key::F6, KeyRepeat::No) {
            let on = engine.toggle_slow_mo();
            println!("Slow motion: {}", if on { "ON" } else { "OFF" });
        }
        if window.is_key_pressed(Key::F7, KeyRepeat::No) {
            engine.toggle_textures();
        }
        if window.is_key_pressed(Key::F8, KeyRepeat::No) {
            let on = engine.toggle_sbs();
            println!("Cardboard SBS: {}", if on { "ON" } else { "OFF" });
        }
        if window.is_key_pressed(Key::F10, KeyRepeat::No) {
            let mode = engine.cycle_sbs_distortion();
            println!("SBS lens warp: {mode}");
        }
        if window.is_key_pressed(Key::LeftBracket, KeyRepeat::Yes) {
            let ipd = engine.adjust_ipd(-0.002);
            println!("IPD: {:.1} mm", ipd * 1000.0);
        }
        if window.is_key_pressed(Key::RightBracket, KeyRepeat::Yes) {
            let ipd = engine.adjust_ipd(0.002);
            println!("IPD: {:.1} mm", ipd * 1000.0);
        }
        if window.is_key_pressed(Key::F9, KeyRepeat::No) {
            let on = engine.toggle_perf();
            println!("Perf monitor: {}", if on { "ON" } else { "OFF" });
        }
        if window.is_key_pressed(Key::F11, KeyRepeat::No) {
            scale_2x = !scale_2x;
            window = open_window(scale_2x);
            last_mouse = None;
        }

        // Camera look (mouse drag + keyboard fallback)
        let mut yaw_delta = 0.0_f32;
        let mut pitch_delta = 0.0_f32;
        let mouse_look =
            window.get_mouse_down(MouseButton::Left) || window.get_mouse_down(MouseButton::Right);
        if mouse_look {
            if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Pass) {
                if let Some((lx, ly)) = last_mouse {
                    yaw_delta += (mx - lx) * MOUSE_SENS;
                    pitch_delta -= (my - ly) * MOUSE_SENS;
                }
                last_mouse = Some((mx, my));
            }
        } else {
            last_mouse = None;
        }

        let turn_speed = 2.0 * dt;
        if window.is_key_down(Key::Left) || window.is_key_down(Key::Q) {
            yaw_delta -= turn_speed;
        }
        if window.is_key_down(Key::Right) || window.is_key_down(Key::E) {
            yaw_delta += turn_speed;
        }
        if window.is_key_down(Key::Up) {
            pitch_delta += turn_speed * 0.75;
        }
        if window.is_key_down(Key::Down) {
            pitch_delta -= turn_speed * 0.75;
        }

        let move_forward =
            (window.is_key_down(Key::W) as i32 - window.is_key_down(Key::S) as i32) as f32;
        let move_right =
            (window.is_key_down(Key::D) as i32 - window.is_key_down(Key::A) as i32) as f32;

        let input = InputState {
            move_forward,
            move_right,
            yaw_delta,
            pitch_delta,
            sprint: window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift),
            jump_pressed: window.is_key_pressed(Key::Space, KeyRepeat::No),
        };

        let report = engine.step(raw_dt, input);
        render_ms_acc += report.render_ms;

        let t_pres = Instant::now();
        window
            .update_with_buffer(engine.framebuffer(), W, H)
            .unwrap();
        let present_ms = t_pres.elapsed().as_secs_f32() * 1000.0;
        engine.record_present_ms(present_ms);

        fps_timer += dt;
        fps_frames += 1;
        if fps_timer >= 0.35 {
            let fps = fps_frames as f32 / fps_timer;
            let avg_render_ms = render_ms_acc / fps_frames as f32;
            let title = engine.format_title(fps, avg_render_ms, report.drawn_tris);
            window.set_title(&title);
            if scale_2x {
                println!("{title}");
            }
            fps_timer = 0.0;
            fps_frames = 0;
            render_ms_acc = 0.0;
        }
    }
}
