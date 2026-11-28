use crate::controller::{WalkConfig, Walker};
use crate::perf::{FrameSample, PerfMonitor};
use crate::raster::{Camera, Framebuffer, RenderConfig};
use crate::raycast;
use crate::scene::{
    Texture, Tri, build_demo_room, build_playground, build_spinner, load_obj, tag_slivers,
};
use crate::stereo::{StereoConfig, StereoRenderer, StereoScene};
use crate::view_bob::{JumpFeedback, ViewBob};
use crate::wad;
use sliver_math::Vec3;
use std::path::Path;
use std::time::Instant;

const THRESHOLDS: [f32; 4] = [1.0, 5.0, 12.0, 25.0];
const STEP: f32 = 1.0 / 120.0;
pub const DEFAULT_EYE_HEIGHT: f32 = 1.65;

/// Platform-agnostic world data loaded from a Doom WAD, Wavefront OBJ, or built-in test scene.
pub struct World {
    pub name: String,
    pub tris: Vec<Tri>,
    pub textures: Vec<Texture>,
    pub sky_tex_id: Option<u16>,
    pub is_external_map: bool,
    pub body_start: Vec3,
    pub start_yaw: f32,
    pub default_fog_end: f32,
    pub eye_height: f32,
    pub walk_cfg: WalkConfig,
}

impl World {
    pub fn demo_room() -> Self {
        let walk_cfg = WalkConfig::default();
        Self {
            name: "demo_room".into(),
            tris: build_demo_room(),
            textures: Vec::new(),
            sky_tex_id: None,
            is_external_map: false,
            body_start: Vec3::new(0.0, walk_cfg.radius, 4.2),
            start_yaw: 0.0,
            default_fog_end: 20.0,
            eye_height: DEFAULT_EYE_HEIGHT,
            walk_cfg,
        }
    }

    pub fn from_wad_file(path: &Path, map_name: Option<&str>) -> Result<Self, String> {
        let map = wad::load_wad_map(path, map_name)?;
        Ok(Self::from_wad_map(map))
    }

    pub fn from_wad_bytes(bytes: &[u8], map_name: Option<&str>) -> Result<Self, String> {
        let map = wad::load_wad_map_bytes(bytes, map_name)?;
        Ok(Self::from_wad_map(map))
    }

    fn from_wad_map(map: wad::WadMap) -> Self {
        // Doom's MAXSTEPHEIGHT is 24 units (0.80 m at 1.2/36 Y scale) and VIEWHEIGHT is 41 (~1.37 m).
        let walk_cfg = WalkConfig {
            max_step_height: 0.82,
            ..WalkConfig::default()
        };
        let body_start = map.spawn_pos + Vec3::new(0.0, walk_cfg.radius, 0.0);
        Self {
            name: map.name,
            tris: map.tris,
            textures: map.textures,
            sky_tex_id: map.sky_tex_id,
            is_external_map: true,
            body_start,
            start_yaw: map.spawn_yaw,
            default_fog_end: 55.0,
            eye_height: 41.0 * (1.2 / 36.0),
            walk_cfg,
        }
    }

    pub fn from_obj_file(path: &Path) -> Result<Self, String> {
        let tris = load_obj(path)?;
        Ok(Self {
            name: path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model.obj")
                .to_string(),
            tris,
            textures: Vec::new(),
            sky_tex_id: None,
            is_external_map: true,
            body_start: Vec3::new(0.0, 1.5, 8.0),
            start_yaw: 0.0,
            default_fog_end: 20.0,
            eye_height: DEFAULT_EYE_HEIGHT,
            walk_cfg: WalkConfig::default(),
        })
    }
}

/// Host-supplied per-frame controls (keyboard/mouse on desktop, touch/gamepad/IMU on Android).
#[derive(Clone, Copy, Default, Debug)]
pub struct InputState {
    /// Forward/backward movement in `[-1.0, 1.0]`.
    pub move_forward: f32,
    /// Strafe right/left movement in `[-1.0, 1.0]`.
    pub move_right: f32,
    /// Incremental yaw delta in radians.
    pub yaw_delta: f32,
    /// Incremental pitch delta in radians.
    pub pitch_delta: f32,
    pub sprint: bool,
    pub jump_pressed: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameReport {
    pub drawn_tris: usize,
    pub total_tris: usize,
    pub render_ms: f32,
}

struct PendingSample {
    phys_ms: f32,
    clear_ms: f32,
    rast_ms: f32,
    frame_ms: f32,
    drawn: usize,
}

/// Headless engine core: physics, camera, rasterizer, stereo SBS, and rolling profiler.
pub struct Engine {
    world_tris: Vec<Tri>,
    world_textures: Vec<Texture>,
    is_external_map: bool,
    thresh_idx: usize,
    sliver_count: usize,
    fb: Framebuffer,
    stereo_renderer: StereoRenderer,
    pub stereo: StereoConfig,
    pub perf: PerfMonitor,
    walk_cfg: WalkConfig,
    eye_offset: Vec3,
    walker: Walker,
    bob: ViewBob,
    prev_body_pos: Vec3,
    prev_bob: ViewBob,
    pub cam: Camera,
    pub cfg: RenderConfig,
    step_acc: f32,
    jump_queued: bool,
    in_playground: bool,
    slow_mo: bool,
    elapsed: f32,
    pending_sample: Option<PendingSample>,
}

impl Engine {
    pub fn new(width: usize, height: usize, mut world: World) -> Self {
        let thresh_idx = 2usize; // default threshold: 12 deg
        let sliver_count = tag_slivers(&mut world.tris, THRESHOLDS[thresh_idx]);

        let eye_offset = Vec3::new(0.0, world.eye_height - world.walk_cfg.radius, 0.0);
        let walker = Walker::new(world.walk_cfg, world.body_start);
        let bob = ViewBob::new();
        let prev_body_pos = walker.pos;
        let prev_bob = bob;

        let cam = Camera {
            pos: walker.pos + eye_offset,
            yaw: world.start_yaw,
            pitch: -0.05,
            fov_y_rad: 68.0_f32.to_radians(),
            near: 0.12,
        };

        let cfg = RenderConfig {
            debug_slivers: false,
            textures_enabled: true,
            fog_enabled: true,
            fog_color: [16, 20, 26],
            fog_start: 4.5,
            fog_end: world.default_fog_end,
            light_dir: Vec3::new(0.45, 0.82, 0.35).normalize(),
            sky_tex_id: world.sky_tex_id,
        };

        Self {
            world_tris: world.tris,
            world_textures: world.textures,
            is_external_map: world.is_external_map,
            thresh_idx,
            sliver_count,
            fb: Framebuffer::new(width, height),
            stereo_renderer: StereoRenderer::new(),
            stereo: StereoConfig::cardboard(),
            perf: PerfMonitor::new(),
            walk_cfg: world.walk_cfg,
            eye_offset,
            walker,
            bob,
            prev_body_pos,
            prev_bob,
            cam,
            cfg,
            step_acc: 0.0,
            jump_queued: false,
            in_playground: false,
            slow_mo: false,
            elapsed: 0.0,
            pending_sample: None,
        }
    }

    pub fn width(&self) -> usize {
        self.fb.width
    }

    pub fn height(&self) -> usize {
        self.fb.height
    }

    /// Packed `0x00RRGGBB` output framebuffer (`width() * height()`).
    pub fn framebuffer(&self) -> &[u32] {
        &self.fb.color
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if width > 0 && height > 0 && (width != self.fb.width || height != self.fb.height) {
            self.fb = Framebuffer::new(width, height);
        }
    }

    /// Sets camera orientation directly (e.g., from an Android gyroscope / rotation vector).
    pub fn set_look(&mut self, yaw: f32, pitch: f32) {
        self.cam.yaw = yaw;
        self.cam.pitch = pitch.clamp(-1.35, 1.35);
    }

    pub fn toggle_sliver_debug(&mut self) {
        self.cfg.debug_slivers = !self.cfg.debug_slivers;
    }

    pub fn cycle_sliver_threshold(&mut self) {
        self.thresh_idx = (self.thresh_idx + 1) % THRESHOLDS.len();
        self.sliver_count = tag_slivers(&mut self.world_tris, THRESHOLDS[self.thresh_idx]);
    }

    pub fn toggle_fog(&mut self) {
        self.cfg.fog_enabled = !self.cfg.fog_enabled;
    }

    pub fn raycast_center(&self) -> Option<(usize, f32, bool)> {
        let (_, _, forward) = self.cam.basis();
        raycast::raycast(&self.world_tris, self.cam.pos, forward, 100.0)
            .map(|(i, hit)| (i, hit.t, hit.front_face))
    }

    pub fn toggle_playground(&mut self) {
        if self.is_external_map {
            return;
        }
        self.in_playground = !self.in_playground;
        self.world_tris = if self.in_playground {
            build_playground()
        } else {
            build_demo_room()
        };
        self.sliver_count = tag_slivers(&mut self.world_tris, THRESHOLDS[self.thresh_idx]);
        let spawn = if self.in_playground {
            Vec3::new(0.0, self.walk_cfg.radius, 25.0)
        } else {
            Vec3::new(0.0, self.walk_cfg.radius, 4.2)
        };
        self.walker = Walker::new(self.walk_cfg, spawn);
        self.bob = ViewBob::new();
        self.prev_body_pos = self.walker.pos;
        self.prev_bob = self.bob;
        self.jump_queued = false;
        self.cfg.fog_end = if self.in_playground { 48.0 } else { 20.0 };
    }

    pub fn toggle_slow_mo(&mut self) -> bool {
        self.slow_mo = !self.slow_mo;
        self.slow_mo
    }

    pub fn toggle_textures(&mut self) {
        self.cfg.textures_enabled = !self.cfg.textures_enabled;
    }

    pub fn toggle_sbs(&mut self) -> bool {
        self.stereo.enabled = !self.stereo.enabled;
        self.stereo.enabled
    }

    pub fn cycle_sbs_distortion(&mut self) -> &'static str {
        if self.stereo.has_warp() {
            self.stereo.k1 = 0.0;
            self.stereo.k2 = 0.0;
            self.stereo.zoom = 1.0;
            "FLAT"
        } else {
            let def = StereoConfig::cardboard();
            self.stereo.k1 = def.k1;
            self.stereo.k2 = def.k2;
            self.stereo.zoom = def.zoom;
            "CARDBOARD"
        }
    }

    pub fn adjust_ipd(&mut self, delta_m: f32) -> f32 {
        self.stereo.ipd_m = (self.stereo.ipd_m + delta_m).clamp(0.0, 0.12);
        self.stereo.ipd_m
    }

    pub fn toggle_perf(&mut self) -> bool {
        self.perf.hud_enabled = !self.perf.hud_enabled;
        self.perf.log_enabled = self.perf.hud_enabled;
        self.perf.hud_enabled
    }

    /// Advances physics and renders one frame (mono or SBS) into `self.framebuffer()`.
    pub fn step(&mut self, raw_dt: f32, input: InputState) -> FrameReport {
        let frame_ms = raw_dt * 1000.0;
        let dt = raw_dt.min(0.05);
        self.elapsed += dt;

        if input.jump_pressed {
            self.jump_queued = true;
        }

        self.cam.yaw += input.yaw_delta;
        self.cam.pitch = (self.cam.pitch + input.pitch_delta).clamp(-1.35, 1.35);

        // Horizontal desired velocity direction from camera yaw.
        let (sy, cy) = self.cam.yaw.sin_cos();
        let fwd_xz = Vec3::new(sy, 0.0, -cy);
        let right_xz = Vec3::new(cy, 0.0, sy);
        let wish = fwd_xz * input.move_forward + right_xz * input.move_right;
        let wish_len = wish.len();
        let wish_dir = if wish_len > 1.0 {
            wish * (1.0 / wish_len)
        } else if wish_len > 1e-6 {
            wish
        } else {
            Vec3::ZERO
        };

        self.walker.sprinting = input.sprint;

        // 1) Fixed-timestep physics.
        let t_phys = Instant::now();
        self.step_acc += if self.slow_mo { dt * 0.2 } else { dt };
        while self.step_acc >= STEP {
            self.step_acc -= STEP;
            self.prev_body_pos = self.walker.pos;
            self.prev_bob = self.bob;

            self.walker
                .tick(&self.world_tris, STEP, wish_dir, self.jump_queued);
            self.jump_queued = false;

            let speed = Vec3::new(self.walker.vel.x, 0.0, self.walker.vel.z).len();
            self.bob.tick(
                STEP,
                speed,
                self.walker.cfg.max_speed,
                self.walker.grounded,
                self.walker.last_step_y,
                JumpFeedback {
                    crouch: self.walker.jump_crouch,
                    just_jumped: self.walker.just_jumped,
                    landing_speed: self.walker.last_landing_speed,
                },
            );
        }
        let phys_ms = t_phys.elapsed().as_secs_f32() * 1000.0;

        // Interpolate between the last two physics ticks to avoid 1-vs-2 tick stutter at 60 FPS.
        let alpha = (self.step_acc / STEP).clamp(0.0, 1.0);
        let body_pos = self.prev_body_pos + (self.walker.pos - self.prev_body_pos) * alpha;
        let (right, _, _) = self.cam.basis();
        let prev_off = self.prev_bob.offset(right);
        let curr_off = self.bob.offset(right);
        self.cam.pos = body_pos + self.eye_offset + prev_off + (curr_off - prev_off) * alpha;

        let (prev_yaw, prev_pitch) = self.prev_bob.angles();
        let (curr_yaw, curr_pitch) = self.bob.angles();
        let bob_yaw = prev_yaw + (curr_yaw - prev_yaw) * alpha;
        let bob_pitch = prev_pitch + (curr_pitch - prev_pitch) * alpha;
        let render_cam = Camera {
            pos: self.cam.pos,
            yaw: self.cam.yaw + bob_yaw,
            pitch: (self.cam.pitch + bob_pitch).clamp(-1.45, 1.45),
            fov_y_rad: self.cam.fov_y_rad,
            near: self.cam.near,
        };

        let spinner_buf;
        let extra_tris: &[Tri] = if !self.is_external_map {
            spinner_buf = build_spinner(self.elapsed * 1.1, Vec3::new(0.0, 2.5, -4.0));
            &spinner_buf
        } else {
            &[]
        };

        // 2) Clear + rasterize (mono or SBS).
        let (clear_ms, rast_ms, drawn) = if self.stereo.enabled {
            let t_rast = Instant::now();
            let w = self.fb.width;
            let h = self.fb.height;
            let drawn = self.stereo_renderer.render(
                &mut self.fb.color,
                w,
                h,
                &StereoScene {
                    world_tris: &self.world_tris,
                    extra_tris,
                    textures: &self.world_textures,
                    cam: &render_cam,
                    cfg: &self.cfg,
                },
                &self.stereo,
            );
            (0.0, t_rast.elapsed().as_secs_f32() * 1000.0, drawn)
        } else {
            let t_clear = Instant::now();
            self.fb.clear(self.cfg.fog_color);
            let clear_ms = t_clear.elapsed().as_secs_f32() * 1000.0;

            let t_rast = Instant::now();
            let mut drawn = self.fb.draw_triangles(
                &self.world_tris,
                &self.world_textures,
                &render_cam,
                &self.cfg,
            );
            if !extra_tris.is_empty() {
                drawn += self
                    .fb
                    .draw_triangles(extra_tris, &[], &render_cam, &self.cfg);
            }
            (clear_ms, t_rast.elapsed().as_secs_f32() * 1000.0, drawn)
        };

        let w = self.fb.width;
        let h = self.fb.height;
        self.perf.draw_hud(&mut self.fb.color, w, h);

        self.pending_sample = Some(PendingSample {
            phys_ms,
            clear_ms,
            rast_ms,
            frame_ms,
            drawn,
        });

        FrameReport {
            drawn_tris: drawn,
            total_tris: self.world_tris.len(),
            render_ms: clear_ms + rast_ms,
        }
    }

    /// Completes the profiler record after the host presents the framebuffer.
    pub fn record_present_ms(&mut self, present_ms: f32) {
        if let Some(p) = self.pending_sample.take() {
            self.perf.push(FrameSample {
                phys_ms: p.phys_ms,
                clear_ms: p.clear_ms,
                rast_ms: p.rast_ms,
                present_ms,
                frame_ms: p.frame_ms,
            });
            self.perf.maybe_log(p.drawn, self.world_tris.len());
        }
    }

    pub fn format_title(&self, fps: f32, avg_render_ms: f32, drawn: usize) -> String {
        let mode = if self.cfg.debug_slivers {
            "SLIVER-DEBUG"
        } else if self.cfg.textures_enabled {
            "TEX"
        } else {
            "FLAT"
        };
        let view = if !self.stereo.enabled {
            ""
        } else if self.stereo.has_warp() {
            " | SBS-VR"
        } else {
            " | SBS-FLAT"
        };

        if self.perf.hud_enabled {
            let s = self.perf.summary();
            format!(
                "sliver_engine | {:.0} FPS | work {:.2} ms (phys {:.2} clr {:.2} rast {:.2}) | p95 {:.1} max {:.1} | tris {}/{} | {}{}",
                fps,
                s.work_avg,
                s.phys_avg,
                s.clear_avg,
                s.rast_avg,
                s.frame_p95,
                s.frame_max,
                drawn,
                self.world_tris.len(),
                mode,
                view
            )
        } else {
            format!(
                "sliver_engine | {:.0} FPS ({:.2} ms) | tris: {}/{} | slivers (<{}°): {} | [F1] {}{}",
                fps,
                avg_render_ms,
                drawn,
                self.world_tris.len(),
                THRESHOLDS[self.thresh_idx],
                self.sliver_count,
                mode,
                view
            )
        }
    }
}
