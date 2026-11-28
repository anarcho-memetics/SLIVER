#![cfg(target_os = "android")]

use android_activity::input::{InputEvent, MotionAction};
use android_activity::{AndroidApp, InputStatus, MainEvent, PollEvent};
use ndk::native_window::NativeWindow;
use sliver_engine::engine::{Engine, InputState, World};
use std::time::{Duration, Instant};

const WINDOW_FORMAT_RGBX_8888: i32 = 2;
const TARGET_RENDER_H: usize = 480;
const LOOK_SENS: f32 = 0.0045;

#[derive(Default)]
struct TouchControls {
    screen_w: f32,
    screen_h: f32,
    /// `(pointer_id, start_x, start_y, cur_x, cur_y)` for left-half virtual stick.
    left_stick: Option<(i32, f32, f32, f32, f32)>,
    /// `(pointer_id, last_x, last_y)` for right-half look drag.
    right_look: Option<(i32, f32, f32)>,
    pending_yaw: f32,
    pending_pitch: f32,
    pending_jump: bool,
    toggle_sbs: bool,
    toggle_warp: bool,
    toggle_perf: bool,
}

impl TouchControls {
    fn set_screen_size(&mut self, w: f32, h: f32) {
        self.screen_w = w.max(1.0);
        self.screen_h = h.max(1.0);
    }

    fn handle_event(&mut self, event: &InputEvent<'_>) -> InputStatus {
        let InputEvent::MotionEvent(motion) = event else {
            return InputStatus::Unhandled;
        };

        match motion.action() {
            MotionAction::Down | MotionAction::PointerDown => {
                let p = motion.pointer_at_index(motion.pointer_index());
                let (id, x, y) = (p.pointer_id(), p.x(), p.y());

                if y < self.screen_h * 0.14 {
                    if x < self.screen_w * 0.22 {
                        self.toggle_perf = true;
                    } else if x > self.screen_w * 0.78 {
                        self.toggle_sbs = true;
                    } else if x > self.screen_w * 0.56 {
                        self.toggle_warp = true;
                    }
                } else if x < self.screen_w * 0.5 {
                    if self.left_stick.is_none() {
                        self.left_stick = Some((id, x, y, x, y));
                    }
                } else if y > self.screen_h * 0.78 && x > self.screen_w * 0.80 {
                    self.pending_jump = true;
                } else if self.right_look.is_none() {
                    self.right_look = Some((id, x, y));
                }
                InputStatus::Handled
            }
            MotionAction::Move => {
                for p in motion.pointers() {
                    let id = p.pointer_id();
                    if let Some((sid, _, _, cx, cy)) = &mut self.left_stick
                        && id == *sid
                    {
                        *cx = p.x();
                        *cy = p.y();
                    }
                    if let Some((lid, lx, ly)) = &mut self.right_look
                        && id == *lid
                    {
                        self.pending_yaw += (p.x() - *lx) * LOOK_SENS;
                        self.pending_pitch -= (p.y() - *ly) * LOOK_SENS;
                        *lx = p.x();
                        *ly = p.y();
                    }
                }
                InputStatus::Handled
            }
            MotionAction::Up | MotionAction::PointerUp => {
                let id = motion
                    .pointer_at_index(motion.pointer_index())
                    .pointer_id();
                if self.left_stick.is_some_and(|s| s.0 == id) {
                    self.left_stick = None;
                }
                if self.right_look.is_some_and(|r| r.0 == id) {
                    self.right_look = None;
                }
                InputStatus::Handled
            }
            MotionAction::Cancel => {
                self.left_stick = None;
                self.right_look = None;
                InputStatus::Handled
            }
            _ => InputStatus::Unhandled,
        }
    }

    fn take_input(&mut self) -> InputState {
        let (move_forward, move_right, sprint) = if let Some((_, x0, y0, cx, cy)) = self.left_stick
        {
            let radius = (self.screen_h * 0.22).max(40.0);
            let rx = ((cx - x0) / radius).clamp(-1.0, 1.0);
            let ry = ((y0 - cy) / radius).clamp(-1.0, 1.0);
            let mag = rx.hypot(ry);
            if mag < 0.12 {
                (0.0, 0.0, false)
            } else {
                (ry, rx, mag > 0.85)
            }
        } else {
            (0.0, 0.0, false)
        };

        let input = InputState {
            move_forward,
            move_right,
            yaw_delta: self.pending_yaw,
            pitch_delta: self.pending_pitch,
            sprint,
            jump_pressed: self.pending_jump,
        };
        self.pending_yaw = 0.0;
        self.pending_pitch = 0.0;
        self.pending_jump = false;
        input
    }
}

fn load_world_from_assets(app: &AndroidApp) -> World {
    let am = app.asset_manager();
    if let Some(mut asset) = am.open(c"DOOM.WAD")
        && let Ok(bytes) = asset.buffer()
    {
        match World::from_wad_bytes(bytes, Some("E1M1")) {
            Ok(w) => {
                println!(
                    "Loaded DOOM.WAD ({}): {} tris, {} textures",
                    w.name,
                    w.tris.len(),
                    w.textures.len()
                );
                return w;
            }
            Err(e) => eprintln!("Failed to parse DOOM.WAD: {e}"),
        }
    }
    println!("DOOM.WAD asset not found; falling back to demo_room");
    World::demo_room()
}

fn compute_render_size(win_w: i32, win_h: i32) -> (usize, usize) {
    let w = (win_w.max(320)) as usize;
    let h = (win_h.max(240)) as usize;
    let render_h = h.min(TARGET_RENDER_H);
    let render_w = ((w * render_h) / h).clamp(320, 1280) & !1;
    (render_w.max(2), render_h.max(2))
}

fn configure_window(win: &NativeWindow, width: usize, height: usize) {
    // SAFETY: `win.ptr()` is a valid non-null ANativeWindow owned by AndroidApp for this lifecycle.
    unsafe {
        ndk_sys::ANativeWindow_setBuffersGeometry(
            win.ptr().as_ptr(),
            width as i32,
            height as i32,
            WINDOW_FORMAT_RGBX_8888,
        );
    }
}

fn blit_to_window(win: &NativeWindow, src: &[u32], src_w: usize, src_h: usize) {
    // SAFETY: Locks the ANativeWindow surface buffer, writes within `copy_h * stride` bounds,
    // and unconditionally calls `ANativeWindow_unlockAndPost` when lock succeeds.
    unsafe {
        let mut buf: ndk_sys::ANativeWindow_Buffer = std::mem::zeroed();
        if ndk_sys::ANativeWindow_lock(win.ptr().as_ptr(), &mut buf, std::ptr::null_mut()) != 0 {
            return;
        }
        if !buf.bits.is_null() && buf.width > 0 && buf.height > 0 && buf.stride >= buf.width {
            let stride = buf.stride as usize;
            let copy_w = src_w.min(buf.width as usize);
            let copy_h = src_h.min(buf.height as usize);
            let dst = std::slice::from_raw_parts_mut(
                buf.bits.cast::<u32>(),
                stride * (buf.height as usize),
            );
            for y in 0..copy_h {
                let s_row = &src[y * src_w..y * src_w + copy_w];
                let d_row = &mut dst[y * stride..y * stride + copy_w];
                for (d, &px) in d_row.iter_mut().zip(s_row.iter()) {
                    // Engine outputs 0x00RRGGBB; Android RGBX_8888 (little-endian) expects 0xFFBBGGRR.
                    *d = 0xFF00_0000
                        | ((px & 0x0000_00FF) << 16)
                        | (px & 0x0000_FF00)
                        | ((px & 0x00FF_0000) >> 16);
                }
            }
        }
        ndk_sys::ANativeWindow_unlockAndPost(win.ptr().as_ptr());
    }
}

#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    let world = load_world_from_assets(&app);
    let mut engine = Engine::new(854, TARGET_RENDER_H, world);
    engine.perf.hud_enabled = true;
    engine.perf.log_enabled = true;

    let mut touch = TouchControls::default();
    let mut active_window: Option<NativeWindow> = None;
    let mut running = true;
    let mut resumed = false;
    let mut last = Instant::now();

    while running {
        let timeout = if resumed && active_window.is_some() {
            Some(Duration::ZERO)
        } else {
            None
        };

        let mut init_window = false;
        let mut drop_window = false;
        app.poll_events(timeout, |event| {
            if let PollEvent::Main(main_ev) = event {
                match main_ev {
                    MainEvent::InitWindow { .. } => init_window = true,
                    MainEvent::TerminateWindow { .. } => drop_window = true,
                    MainEvent::Resume { .. } => {
                        resumed = true;
                        last = Instant::now();
                    }
                    MainEvent::Pause => resumed = false,
                    MainEvent::Destroy => running = false,
                    _ => {}
                }
            }
        });

        if drop_window {
            active_window = None;
        }
        if init_window && let Some(win) = app.native_window() {
            let (sw, sh) = (win.width(), win.height());
            touch.set_screen_size(sw as f32, sh as f32);
            let (rw, rh) = compute_render_size(sw, sh);
            configure_window(&win, rw, rh);
            engine.resize(rw, rh);
            println!("Android window ready: physical {sw}x{sh} -> render {rw}x{rh}");
            active_window = Some(win);
            last = Instant::now();
        }

        if let Ok(mut iter) = app.input_events_iter() {
            while iter.next(|event| touch.handle_event(event)) {}
        }

        if touch.toggle_sbs {
            touch.toggle_sbs = false;
            let on = engine.toggle_sbs();
            println!("Cardboard SBS: {}", if on { "ON" } else { "OFF" });
        }
        if touch.toggle_warp {
            touch.toggle_warp = false;
            let mode = engine.cycle_sbs_distortion();
            println!("SBS lens warp: {mode}");
        }
        if touch.toggle_perf {
            touch.toggle_perf = false;
            let on = engine.toggle_perf();
            println!("Perf HUD: {}", if on { "ON" } else { "OFF" });
        }

        if !resumed {
            continue;
        }
        let Some(ref win) = active_window else {
            continue;
        };

        let now = Instant::now();
        let raw_dt = now.duration_since(last).as_secs_f32();
        last = now;

        let input = touch.take_input();
        engine.step(raw_dt, input);

        let t_pres = Instant::now();
        blit_to_window(win, engine.framebuffer(), engine.width(), engine.height());
        engine.record_present_ms(t_pres.elapsed().as_secs_f32() * 1000.0);
    }
}