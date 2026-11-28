use crate::scene::{Texture, Tri, NO_TEX, SKY_TEX};
use sliver_math::Vec3;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Perspective-correct texture span length in pixels (Quake-style 16 px subdivision).
const SPAN_SIZE: usize = 16;
const INV_SPAN_SIZE: f32 = 1.0 / SPAN_SIZE as f32;
/// Minimum projected triangle count before waking worker bands.
const MT_TRI_THRESHOLD: usize = 16;
/// Cap total threads (main + persistent workers) to keep band sizes L2-friendly.
const MAX_THREADS: usize = 8;

pub struct Camera {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_rad: f32,
    pub near: f32,
}

impl Camera {
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);
        let up = right.cross(forward);
        (right, up, forward)
    }

    /// Transforms a world-space point into view space (+X right, +Y up, +Z forward).
    #[inline(always)]
    pub fn world_to_view(&self, p: Vec3, right: Vec3, up: Vec3, forward: Vec3) -> Vec3 {
        let rel = p - self.pos;
        Vec3::new(rel.dot(right), rel.dot(up), rel.dot(forward))
    }
}

pub struct RenderConfig {
    pub debug_slivers: bool,
    pub textures_enabled: bool,
    pub fog_enabled: bool,
    pub fog_color: [u8; 3],
    pub fog_start: f32,
    pub fog_end: f32,
    pub light_dir: Vec3,
    pub sky_tex_id: Option<u16>,
}

#[derive(Clone, Copy)]
struct FogParams {
    packed: u32,
    start: f32,
    iz_near: f32,
    iz_far: f32,
    inv_iz_span: f32,
    enabled: bool,
}

#[derive(Clone, Copy)]
struct ClipVert {
    pos: Vec3,
    uv: [f32; 2],
}

#[derive(Clone, Copy)]
struct ProjCtx {
    width: usize,
    height: usize,
    wf: f32,
    hf: f32,
    cx: f32,
    cy: f32,
    focal: f32,
    fog_enabled: bool,
    fog_start: f32,
}

#[derive(Clone, Copy)]
struct TriStyle {
    tex_id: u16,
    shade256: u32,
    solid_packed: u32,
    allow_fog: bool,
}
#[derive(Clone, Copy)]
struct SkyRow {
    tex_y: usize,
    cap_color: u32,
    fade256: u32,
}

#[derive(Clone, Default)]
struct SkyLut {
    tex_id: u16,
    u_cols: Vec<usize>,
    v_rows: Vec<SkyRow>,
}

#[derive(Clone, Copy)]
struct ProjectedTri {
    min_x: usize,
    max_x: usize,
    min_y: usize,
    max_y: usize,
    e0_base: f32,
    e1_base: f32,
    e2_base: f32,
    dx0: f32,
    dy0: f32,
    inv_dx0: f32,
    dx1: f32,
    dy1: f32,
    inv_dx1: f32,
    dx2: f32,
    dy2: f32,
    inv_dx2: f32,
    iz_base: f32,
    diz_dx: f32,
    diz_dy: f32,
    uz_base: f32,
    duz_dx: f32,
    duz_dy: f32,
    vz_base: f32,
    dvz_dx: f32,
    dvz_dy: f32,
    /// Descending sort key (`iz0 + iz1 + iz2`) for front-to-back overdraw rejection.
    sort_iz: f32,
    solid_packed: u32,
    shade256: u32,
    tex_id: u16,
    need_fog: bool,
}

struct Band<'a> {
    color: &'a mut [u32],
    inv_z: &'a mut [f32],
    width: usize,
    y0: usize,
}

struct WorkerBuf {
    color: Vec<u32>,
    inv_z: Vec<f32>,
    px_start: usize,
    px_end: usize,
}

struct PoolJob {
    generation: u64,
    projected: Arc<Vec<ProjectedTri>>,
    textures: Arc<Vec<Texture>>,
    sky_lut: Arc<SkyLut>,
    fog: FogParams,
    clear_packed: Option<u32>,
}

struct PoolPass<'a> {
    projected: &'a Arc<Vec<ProjectedTri>>,
    textures: &'a Arc<Vec<Texture>>,
    sky_lut: &'a Arc<SkyLut>,
    fog: FogParams,
    clear_packed: Option<u32>,
}

struct PoolState {
    job: Option<PoolJob>,
    remaining: usize,
    shutdown: bool,
}

struct PoolShared {
    state: Mutex<PoolState>,
    work_cv: Condvar,
    done_cv: Condvar,
}

struct BandPool {
    num_threads: usize,
    width: usize,
    band0_px: usize,
    generation: u64,
    shared: Arc<PoolShared>,
    worker_bufs: Vec<Arc<Mutex<WorkerBuf>>>,
    handles: Vec<JoinHandle<()>>,
}

impl BandPool {
    fn new(num_threads: usize, width: usize, height: usize) -> Self {
        let rows_per_band = height.div_ceil(num_threads);
        let band0_rows = height.min(rows_per_band);
        let band0_px = band0_rows * width;

        let shared = Arc::new(PoolShared {
            state: Mutex::new(PoolState {
                job: None,
                remaining: 0,
                shutdown: false,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
        });

        let mut worker_bufs = Vec::new();
        let mut handles = Vec::new();

        let mut band_idx = 1usize;
        while band_idx * rows_per_band < height {
            let band_y0 = band_idx * rows_per_band;
            let band_rows = (height - band_y0).min(rows_per_band);
            let px_start = band_y0 * width;
            let px_len = band_rows * width;

            let buf = Arc::new(Mutex::new(WorkerBuf {
                color: vec![0; px_len],
                inv_z: vec![0.0; px_len],
                px_start,
                px_end: px_start + px_len,
            }));
            worker_bufs.push(Arc::clone(&buf));

            let shared_w = Arc::clone(&shared);
            let handle = thread::spawn(move || {
                let mut last_gen = 0u64;
                loop {
                    let (projected, textures, sky_lut, fog, clear_packed) = {
                        let mut guard = shared_w.state.lock().unwrap();
                        while !guard.shutdown
                            && guard
                                .job
                                .as_ref()
                                .is_none_or(|j| j.generation == last_gen)
                        {
                            guard = shared_w.work_cv.wait(guard).unwrap();
                        }
                        if guard.shutdown {
                            break;
                        }
                        let j = guard.job.as_ref().unwrap();
                        last_gen = j.generation;
                        (
                            Arc::clone(&j.projected),
                            Arc::clone(&j.textures),
                            Arc::clone(&j.sky_lut),
                            j.fog,
                            j.clear_packed,
                        )
                    };

                    {
                        let mut b = buf.lock().unwrap();
                        let wb = &mut *b;
                        if let Some(packed) = clear_packed {
                            wb.color.fill(packed);
                            wb.inv_z.fill(0.0);
                        }
                        let mut band = Band {
                            color: &mut wb.color,
                            inv_z: &mut wb.inv_z,
                            width,
                            y0: band_y0,
                        };
                        rasterize_band(&mut band, &projected, &textures, &sky_lut, fog);
                    }

                    // Drop Arc clones before signaling done so main thread's Arc::make_mut reuses buffers.
                    drop(projected);
                    drop(textures);
                    drop(sky_lut);

                    let mut guard = shared_w.state.lock().unwrap();
                    guard.remaining -= 1;
                    if guard.remaining == 0 {
                        shared_w.done_cv.notify_one();
                    }
                }
            });
            handles.push(handle);
            band_idx += 1;
        }

        Self {
            num_threads,
            width,
            band0_px,
            generation: 0,
            shared,
            worker_bufs,
            handles,
        }
    }

    fn dispatch(&mut self, color: &mut [u32], inv_z: &mut [f32], pass: PoolPass<'_>) {
        let n_workers = self.worker_bufs.len();
        if n_workers == 0 {
            let mut band = Band {
                color,
                inv_z,
                width: self.width,
                y0: 0,
            };
            rasterize_band(
                &mut band,
                pass.projected,
                pass.textures,
                pass.sky_lut,
                pass.fog,
            );
            return;
        }

        if pass.clear_packed.is_none() {
            for wb in &self.worker_bufs {
                let mut b = wb.lock().unwrap();
                let (s, e) = (b.px_start, b.px_end);
                b.color.copy_from_slice(&color[s..e]);
                b.inv_z.copy_from_slice(&inv_z[s..e]);
            }
        }

        self.generation = self.generation.wrapping_add(1).max(1);
        {
            let mut guard = self.shared.state.lock().unwrap();
            guard.job = Some(PoolJob {
                generation: self.generation,
                projected: Arc::clone(pass.projected),
                textures: Arc::clone(pass.textures),
                sky_lut: Arc::clone(pass.sky_lut),
                fog: pass.fog,
                clear_packed: pass.clear_packed,
            });
            guard.remaining = n_workers;
            self.shared.work_cv.notify_all();
        }

        // Main thread renders band 0 directly in place.
        {
            let mut band0 = Band {
                color: &mut color[..self.band0_px],
                inv_z: &mut inv_z[..self.band0_px],
                width: self.width,
                y0: 0,
            };
            rasterize_band(
                &mut band0,
                pass.projected,
                pass.textures,
                pass.sky_lut,
                pass.fog,
            );
        }

        // Wait for all workers and drop the job's Arc clones.
        {
            let mut guard = self.shared.state.lock().unwrap();
            while guard.remaining > 0 {
                guard = self.shared.done_cv.wait(guard).unwrap();
            }
            guard.job = None;
        }

        for wb in &self.worker_bufs {
            let b = wb.lock().unwrap();
            let (s, e) = (b.px_start, b.px_end);
            color[s..e].copy_from_slice(&b.color);
            inv_z[s..e].copy_from_slice(&b.inv_z);
        }
    }
}

impl Drop for BandPool {
    fn drop(&mut self) {
        {
            let mut guard = self.shared.state.lock().unwrap();
            guard.shutdown = true;
            guard.job = None;
            self.shared.work_cv.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    pub color: Vec<u32>,
    pub inv_z: Vec<f32>,
    projected: Arc<Vec<ProjectedTri>>,
    sky_lut: Arc<SkyLut>,
    tex_cache_key: (usize, usize),
    tex_cache: Arc<Vec<Texture>>,
    clean_clear: Option<u32>,
    num_threads: usize,
    pool: Option<BandPool>,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        let num_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, MAX_THREADS);
        Self {
            width,
            height,
            color: vec![0; width * height],
            inv_z: vec![0.0; width * height],
            projected: Arc::new(Vec::with_capacity(1024)),
            sky_lut: Arc::new(SkyLut::default()),
            tex_cache_key: (0, 0),
            tex_cache: Arc::new(Vec::new()),
            clean_clear: None,
            num_threads,
            pool: None,
        }
    }

    pub fn clear(&mut self, rgb: [u8; 3]) {
        let packed = pack_rgb(rgb[0], rgb[1], rgb[2]);
        self.color.fill(packed);
        self.inv_z.fill(0.0);
        self.clean_clear = Some(packed);
    }

    fn cached_textures(&mut self, textures: &[Texture]) -> Arc<Vec<Texture>> {
        let key = (textures.as_ptr() as usize, textures.len());
        if key != self.tex_cache_key {
            self.tex_cache = Arc::new(textures.to_vec());
            self.tex_cache_key = key;
        }
        Arc::clone(&self.tex_cache)
    }

    pub fn draw_triangles(
        &mut self,
        tris: &[Tri],
        textures: &[Texture],
        cam: &Camera,
        cfg: &RenderConfig,
    ) -> usize {
        if self.width == 0 || self.height == 0 {
            return 0;
        }

        let (right, up, forward) = cam.basis();
        let wf = self.width as f32;
        let hf = self.height as f32;
        let focal = (hf * 0.5) / (cam.fov_y_rad * 0.5).tan();
        let iz_near = 1.0 / cfg.fog_start.max(1e-4);
        let iz_far = 1.0 / cfg.fog_end.max(cfg.fog_start + 1e-4);
        let fog = FogParams {
            packed: pack_rgb(cfg.fog_color[0], cfg.fog_color[1], cfg.fog_color[2]),
            start: cfg.fog_start,
            iz_near,
            iz_far,
            inv_iz_span: 256.0 / (iz_near - iz_far),
            enabled: cfg.fog_enabled,
        };
        let pctx = ProjCtx {
            width: self.width,
            height: self.height,
            wf,
            hf,
            cx: wf * 0.5,
            cy: hf * 0.5,
            focal,
            fog_enabled: fog.enabled,
            fog_start: fog.start,
        };

        let active_sky_tex = if cfg.textures_enabled && !cfg.debug_slivers {
            cfg.sky_tex_id.filter(|&id| {
                textures
                    .get(id as usize)
                    .is_some_and(|t| t.width > 0 && t.height > 0)
            })
        } else {
            None
        };
        let mut has_sky = false;

        // Phase 1: cull, clip against near plane, shade, and project into a reusable buffer.
        let projected_vec = Arc::make_mut(&mut self.projected);
        projected_vec.clear();
        for tri in tris {
            let n = tri.normal();
            // World-space backface culling
            if n.dot(cam.pos - tri.p0) <= 0.0 {
                continue;
            }

            let v0 = ClipVert {
                pos: cam.world_to_view(tri.p0, right, up, forward),
                uv: tri.uv0,
            };
            let v1 = ClipVert {
                pos: cam.world_to_view(tri.p1, right, up, forward),
                uv: tri.uv1,
            };
            let v2 = ClipVert {
                pos: cam.world_to_view(tri.p2, right, up, forward),
                uv: tri.uv2,
            };

            let (clipped, count) = clip_near(v0, v1, v2, cam.near);
            if count == 0 {
                continue;
            }

            let ndotl = n.dot(cfg.light_dir).max(0.0);
            let is_sky_tri = tri.tex_id == SKY_TEX && !cfg.debug_slivers;
            let tex_id = if is_sky_tri {
                if active_sky_tex.is_some() {
                    SKY_TEX
                } else {
                    NO_TEX
                }
            } else if cfg.textures_enabled
                && !cfg.debug_slivers
                && tri.tex_id != NO_TEX
                && (tri.tex_id as usize) < textures.len()
            {
                tri.tex_id
            } else {
                NO_TEX
            };

            let (shade256, solid_packed) = if tex_id == SKY_TEX {
                (256, 0)
            } else if is_sky_tri {
                (256, pack_rgb(tri.color[0], tri.color[1], tri.color[2]))
            } else if tex_id != NO_TEX {
                let light_norm = (tri.light as f32) / 255.0;
                let sector_scale = 0.22 + 0.78 * light_norm;
                let dir_shade = (0.74 + 0.08 * (n.y * 0.5 + 0.5) + 0.18 * ndotl).min(1.0);
                let s256 = ((sector_scale * dir_shade * 256.0).round() as u32).clamp(0, 256);
                (s256, 0)
            } else {
                // Hemispherical ambient + directional diffuse shading
                let hemi = 0.40 + 0.12 * (n.y * 0.5 + 0.5);
                let shade = (hemi + 0.48 * ndotl).min(1.0);
                let lit = if cfg.debug_slivers {
                    if tri.is_sliver {
                        scale_rgb([250, 42, 68], 0.65 + 0.35 * ndotl)
                    } else {
                        scale_rgb([112, 128, 146], shade)
                    }
                } else {
                    scale_rgb(tri.color, shade)
                };
                (256, pack_rgb(lit[0], lit[1], lit[2]))
            };

            let style = TriStyle {
                tex_id,
                shade256,
                solid_packed,
                allow_fog: !is_sky_tri,
            };

            for c in &clipped[..count] {
                if let Some(proj) = project_tri(c.0, c.1, c.2, &pctx, style) {
                    if tex_id == SKY_TEX {
                        has_sky = true;
                    }
                    projected_vec.push(proj);
                }
            }
        }

        let drawn = projected_vec.len();
        if drawn == 0 {
            return 0;
        }

        // Front-to-back sort (largest 1/z sum first) so nearer walls reject background overdraw.
        projected_vec.sort_unstable_by(|a, b| b.sort_iz.total_cmp(&a.sort_iz));

        if has_sky {
            if let Some(sky_id) = active_sky_tex {
                let lut = Arc::make_mut(&mut self.sky_lut);
                build_sky_lut(lut, sky_id, &textures[sky_id as usize], cam, &pctx);
            }
        }

        // Phase 2: rasterize horizontal bands in parallel using the persistent thread pool.
        let width = self.width;
        let height = self.height;
        let n_threads = self.num_threads.min(height);
        let clear_packed = self.clean_clear.take();

        if n_threads <= 1 || drawn < MT_TRI_THRESHOLD {
            let mut band = Band {
                color: &mut self.color,
                inv_z: &mut self.inv_z,
                width,
                y0: 0,
            };
            rasterize_band(&mut band, &self.projected, textures, &self.sky_lut, fog);
        } else {
            let tex_arc = self.cached_textures(textures);
            let need_init = self
                .pool
                .as_ref()
                .is_none_or(|p| p.num_threads != n_threads);
            if need_init {
                self.pool = Some(BandPool::new(n_threads, width, height));
            }
            if let Some(pool) = self.pool.as_mut() {
                pool.dispatch(
                    &mut self.color,
                    &mut self.inv_z,
                    PoolPass {
                        projected: &self.projected,
                        textures: &tex_arc,
                        sky_lut: &self.sky_lut,
                        fog,
                        clear_packed,
                    },
                );
            }
        }

        drawn
    }
}

fn project_tri(
    v0: ClipVert,
    v1: ClipVert,
    v2: ClipVert,
    ctx: &ProjCtx,
    style: TriStyle,
) -> Option<ProjectedTri> {
    let iz0 = 1.0 / v0.pos.z;
    let iz1 = 1.0 / v1.pos.z;
    let iz2 = 1.0 / v2.pos.z;

    let sx0 = ctx.cx + v0.pos.x * iz0 * ctx.focal;
    let sy0 = ctx.cy - v0.pos.y * iz0 * ctx.focal;
    let sx1 = ctx.cx + v1.pos.x * iz1 * ctx.focal;
    let sy1 = ctx.cy - v1.pos.y * iz1 * ctx.focal;
    let sx2 = ctx.cx + v2.pos.x * iz2 * ctx.focal;
    let sy2 = ctx.cy - v2.pos.y * iz2 * ctx.focal;

    let area = edge(sx0, sy0, sx1, sy1, sx2, sy2);
    if area <= 0.0 {
        return None;
    }

    let min_xf = sx0.min(sx1).min(sx2).floor();
    let max_xf = sx0.max(sx1).max(sx2).ceil();
    let min_yf = sy0.min(sy1).min(sy2).floor();
    let max_yf = sy0.max(sy1).max(sy2).ceil();

    if max_xf < 0.0 || min_xf >= ctx.wf || max_yf < 0.0 || min_yf >= ctx.hf {
        return None;
    }

    let min_x = min_xf.max(0.0) as usize;
    let max_x = (max_xf as usize).min(ctx.width - 1);
    let min_y = min_yf.max(0.0) as usize;
    let max_y = (max_yf as usize).min(ctx.height - 1);
    if min_x > max_x || min_y > max_y {
        return None;
    }

    let start_px = min_x as f32 + 0.5;
    let start_py = min_y as f32 + 0.5;

    let e0_base = edge(sx1, sy1, sx2, sy2, start_px, start_py);
    let e1_base = edge(sx2, sy2, sx0, sy0, start_px, start_py);
    let e2_base = edge(sx0, sy0, sx1, sy1, start_px, start_py);

    let dx0 = sy2 - sy1;
    let dy0 = sx1 - sx2;
    let dx1 = sy0 - sy2;
    let dy1 = sx2 - sx0;
    let dx2 = sy1 - sy0;
    let dy2 = sx0 - sx1;

    let inv_dx0 = if dx0.abs() > 1e-7 { 1.0 / dx0 } else { 0.0 };
    let inv_dx1 = if dx1.abs() > 1e-7 { 1.0 / dx1 } else { 0.0 };
    let inv_dx2 = if dx2.abs() > 1e-7 { 1.0 / dx2 } else { 0.0 };

    let inv_area = 1.0 / area;
    let iz_base = (e0_base * iz0 + e1_base * iz1 + e2_base * iz2) * inv_area;
    let diz_dx = (dx0 * iz0 + dx1 * iz1 + dx2 * iz2) * inv_area;
    let diz_dy = (dy0 * iz0 + dy1 * iz1 + dy2 * iz2) * inv_area;

    let (uz_base, duz_dx, duz_dy, vz_base, dvz_dx, dvz_dy) =
        if style.tex_id != NO_TEX && style.tex_id != SKY_TEX {
            let u0_iz = v0.uv[0] * iz0;
            let u1_iz = v1.uv[0] * iz1;
            let u2_iz = v2.uv[0] * iz2;
            let uz_b = (e0_base * u0_iz + e1_base * u1_iz + e2_base * u2_iz) * inv_area;
            let duz_x = (dx0 * u0_iz + dx1 * u1_iz + dx2 * u2_iz) * inv_area;
            let duz_y = (dy0 * u0_iz + dy1 * u1_iz + dy2 * u2_iz) * inv_area;

            let v0_iz = v0.uv[1] * iz0;
            let v1_iz = v1.uv[1] * iz1;
            let v2_iz = v2.uv[1] * iz2;
            let vz_b = (e0_base * v0_iz + e1_base * v1_iz + e2_base * v2_iz) * inv_area;
            let dvz_x = (dx0 * v0_iz + dx1 * v1_iz + dx2 * v2_iz) * inv_area;
            let dvz_y = (dy0 * v0_iz + dy1 * v1_iz + dy2 * v2_iz) * inv_area;

            (uz_b, duz_x, duz_y, vz_b, dvz_x, dvz_y)
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };

    let need_fog =
        style.allow_fog && ctx.fog_enabled && v0.pos.z.max(v1.pos.z).max(v2.pos.z) > ctx.fog_start;

    Some(ProjectedTri {
        min_x,
        max_x,
        min_y,
        max_y,
        e0_base,
        e1_base,
        e2_base,
        dx0,
        dy0,
        inv_dx0,
        dx1,
        dy1,
        inv_dx1,
        dx2,
        dy2,
        inv_dx2,
        iz_base,
        diz_dx,
        diz_dy,
        uz_base,
        duz_dx,
        duz_dy,
        vz_base,
        dvz_dx,
        dvz_dy,
        sort_iz: iz0 + iz1 + iz2,
        solid_packed: style.solid_packed,
        shade256: style.shade256,
        tex_id: style.tex_id,
        need_fog,
    })
}

fn avg_rows_color(tex: &Texture, y_start: usize, y_end: usize) -> u32 {
    let mut r_sum = 0u64;
    let mut g_sum = 0u64;
    let mut b_sum = 0u64;
    let mut count = 0u64;
    for y in y_start..y_end {
        let off = y * tex.width;
        for &c in &tex.pixels[off..off + tex.width] {
            r_sum += ((c >> 16) & 0xFF) as u64;
            g_sum += ((c >> 8) & 0xFF) as u64;
            b_sum += (c & 0xFF) as u64;
            count += 1;
        }
    }
    if count == 0 {
        return 0;
    }
    let half = count / 2;
    (((r_sum + half) / count) as u32) << 16
        | (((g_sum + half) / count) as u32) << 8
        | (((b_sum + half) / count) as u32)
}

fn build_sky_lut(
    lut: &mut SkyLut,
    sky_tex_id: u16,
    tex: &Texture,
    cam: &Camera,
    ctx: &ProjCtx,
) {
    lut.tex_id = sky_tex_id;
    lut.u_cols.clear();
    lut.u_cols.reserve(ctx.width);
    lut.v_rows.clear();
    lut.v_rows.reserve(ctx.height);

    let inv_focal = 1.0 / ctx.focal;
    let tw = tex.width as i32;

    // Classic Doom maps 1024 sky texels to 360 deg (4x wrap for 256-wide, 2x for 512, 1x for 1024).
    let u_scale = 1024.0 / std::f32::consts::TAU;
    for x in 0..ctx.width {
        let angle_x = cam.yaw + ((x as f32 + 0.5 - ctx.cx) * inv_focal).atan();
        let u = (angle_x * u_scale).floor() as i32;
        lut.u_cols.push(u.rem_euclid(tw) as usize);
    }

    // Horizon at ~100/128 of texture height with classic 4:3 aspect (~136 texels/rad).
    // Fade the top/bottom edges into their average cap color so mouselook never stretches row 0.
    let th_f = tex.height as f32;
    let horizon_v = th_f * (100.0 / 128.0);
    let v_scale = th_f * (136.0 / 128.0);
    let max_v = tex.height - 1;
    let max_v_i = max_v as i32;
    let max_v_f = max_v as f32;

    let cap_rows = (tex.height / 32).clamp(1, 4).min(tex.height);
    let top_cap = avg_rows_color(tex, 0, cap_rows);
    let bot_cap = avg_rows_color(tex, tex.height - cap_rows, tex.height);
    let top_fade = (th_f * 0.22).max(1.0);
    let bot_fade = (th_f * 0.08).max(1.0);

    for y in 0..ctx.height {
        let elev = cam.pitch + ((ctx.cy - (y as f32 + 0.5)) * inv_focal).atan();
        let v_f = horizon_v - elev * v_scale;
        let row = if v_f <= 0.0 {
            SkyRow {
                tex_y: 0,
                cap_color: top_cap,
                fade256: 256,
            }
        } else if v_f < top_fade {
            let t = 1.0 - v_f / top_fade;
            let smooth = t * t * (3.0 - 2.0 * t);
            SkyRow {
                tex_y: (v_f.floor() as i32).clamp(0, max_v_i) as usize,
                cap_color: top_cap,
                fade256: (smooth * 256.0).round().clamp(0.0, 256.0) as u32,
            }
        } else if v_f >= max_v_f {
            SkyRow {
                tex_y: max_v,
                cap_color: bot_cap,
                fade256: 256,
            }
        } else if v_f > max_v_f - bot_fade {
            let t = 1.0 - (max_v_f - v_f) / bot_fade;
            let smooth = t * t * (3.0 - 2.0 * t);
            SkyRow {
                tex_y: (v_f.floor() as i32).clamp(0, max_v_i) as usize,
                cap_color: bot_cap,
                fade256: (smooth * 256.0).round().clamp(0.0, 256.0) as u32,
            }
        } else {
            SkyRow {
                tex_y: (v_f.floor() as i32).clamp(0, max_v_i) as usize,
                cap_color: 0,
                fade256: 0,
            }
        };
        lut.v_rows.push(row);
    }
}

fn rasterize_band(
    band: &mut Band<'_>,
    projected: &[ProjectedTri],
    textures: &[Texture],
    sky_lut: &SkyLut,
    fog: FogParams,
) {
    let band_rows = band.color.len() / band.width;
    if band_rows == 0 {
        return;
    }
    let band_y0 = band.y0;
    let band_y1 = band_y0 + band_rows - 1;

    for tri in projected {
        if tri.max_y < band_y0 || tri.min_y > band_y1 {
            continue;
        }
        let y0 = tri.min_y.max(band_y0);
        let y1 = tri.max_y.min(band_y1);

        if tri.tex_id == SKY_TEX {
            let tex = &textures[sky_lut.tex_id as usize];
            scan_sky(band, y0, y1, tri, tex, sky_lut);
        } else if tri.tex_id != NO_TEX {
            let tex = &textures[tri.tex_id as usize];
            match (tex.pow2.is_some(), tri.need_fog) {
                (true, true) => scan_textured::<true, true>(band, y0, y1, tri, tex, fog),
                (true, false) => scan_textured::<true, false>(band, y0, y1, tri, tex, fog),
                (false, true) => scan_textured::<false, true>(band, y0, y1, tri, tex, fog),
                (false, false) => scan_textured::<false, false>(band, y0, y1, tri, tex, fog),
            }
        } else if tri.need_fog {
            scan_solid::<true>(band, y0, y1, tri, fog);
        } else {
            scan_solid::<false>(band, y0, y1, tri, fog);
        }
    }
}

/// Computes the exact covered horizontal pixel range `[x_start, x_end]` for one scanline
/// by intersecting the three linear half-spaces `e_i + dx_i * t >= 0`.
#[inline(always)]
fn scanline_x_range(tri: &ProjectedTri, e0: f32, e1: f32, e2: f32) -> Option<(usize, usize)> {
    let max_t = (tri.max_x - tri.min_x) as f32;
    let mut t_lo = 0.0_f32;
    let mut t_hi = max_t;

    for (e, dx, inv_dx) in [
        (e0, tri.dx0, tri.inv_dx0),
        (e1, tri.dx1, tri.inv_dx1),
        (e2, tri.dx2, tri.inv_dx2),
    ] {
        if dx > 1e-7 {
            t_lo = t_lo.max(-e * inv_dx);
        } else if dx < -1e-7 {
            t_hi = t_hi.min(-e * inv_dx);
        } else if e < 0.0 {
            return None;
        }
    }

    if t_lo > t_hi {
        return None;
    }
    let i0 = t_lo.ceil().max(0.0) as usize;
    let i1 = t_hi.floor().min(max_t) as usize;
    if i0 > i1 {
        return None;
    }
    Some((tri.min_x + i0, tri.min_x + i1))
}

fn scan_sky(
    band: &mut Band<'_>,
    y0: usize,
    y1: usize,
    tri: &ProjectedTri,
    tex: &Texture,
    sky_lut: &SkyLut,
) {
    let tw = tex.width;
    for y in y0..=y1 {
        let dy = (y - tri.min_y) as f32;
        let e0_row = tri.e0_base + tri.dy0 * dy;
        let e1_row = tri.e1_base + tri.dy1 * dy;
        let e2_row = tri.e2_base + tri.dy2 * dy;
        if let Some((x_start, x_end)) = scanline_x_range(tri, e0_row, e1_row, e2_row) {
            let iz_row = tri.iz_base + tri.diz_dy * dy;
            let local_row = (y - band.y0) * band.width;
            let inv_z_slice = &mut band.inv_z[local_row + x_start..=local_row + x_end];
            let color_slice = &mut band.color[local_row + x_start..=local_row + x_end];
            let sky_row = sky_lut.v_rows[y];
            let mut iz = iz_row + tri.diz_dx * ((x_start - tri.min_x) as f32);

            if sky_row.fade256 >= 256 {
                let cap = sky_row.cap_color;
                for (iz_cell, col_cell) in inv_z_slice.iter_mut().zip(color_slice.iter_mut()) {
                    if iz > *iz_cell {
                        *iz_cell = iz;
                        *col_cell = cap;
                    }
                    iz += tri.diz_dx;
                }
            } else {
                let u_slice = &sky_lut.u_cols[x_start..=x_end];
                let row_off = sky_row.tex_y * tw;
                let row_pixels = &tex.pixels[row_off..row_off + tw];
                if sky_row.fade256 == 0 {
                    for ((iz_cell, col_cell), &tx) in inv_z_slice
                        .iter_mut()
                        .zip(color_slice.iter_mut())
                        .zip(u_slice.iter())
                    {
                        if iz > *iz_cell {
                            *iz_cell = iz;
                            *col_cell = row_pixels[tx];
                        }
                        iz += tri.diz_dx;
                    }
                } else {
                    let cap = sky_row.cap_color;
                    let fade = sky_row.fade256;
                    for ((iz_cell, col_cell), &tx) in inv_z_slice
                        .iter_mut()
                        .zip(color_slice.iter_mut())
                        .zip(u_slice.iter())
                    {
                        if iz > *iz_cell {
                            *iz_cell = iz;
                            *col_cell = lerp_packed(row_pixels[tx], cap, fade);
                        }
                        iz += tri.diz_dx;
                    }
                }
            }
        }
    }
}

fn scan_solid<const FOG: bool>(
    band: &mut Band<'_>,
    y0: usize,
    y1: usize,
    tri: &ProjectedTri,
    fog: FogParams,
) {
    let solid_packed = tri.solid_packed;

    for y in y0..=y1 {
        let dy = (y - tri.min_y) as f32;
        let e0_row = tri.e0_base + tri.dy0 * dy;
        let e1_row = tri.e1_base + tri.dy1 * dy;
        let e2_row = tri.e2_base + tri.dy2 * dy;
        if let Some((x_start, x_end)) = scanline_x_range(tri, e0_row, e1_row, e2_row) {
            let iz_row = tri.iz_base + tri.diz_dy * dy;
            let local_row = (y - band.y0) * band.width;
            let inv_z_slice = &mut band.inv_z[local_row + x_start..=local_row + x_end];
            let color_slice = &mut band.color[local_row + x_start..=local_row + x_end];
            let mut iz = iz_row + tri.diz_dx * ((x_start - tri.min_x) as f32);

            for (iz_cell, col_cell) in inv_z_slice.iter_mut().zip(color_slice.iter_mut()) {
                if iz > *iz_cell {
                    *iz_cell = iz;
                    *col_cell = if FOG {
                        if iz >= fog.iz_near {
                            solid_packed
                        } else if iz <= fog.iz_far {
                            fog.packed
                        } else {
                            let t256 = ((fog.iz_near - iz) * fog.inv_iz_span) as u32;
                            lerp_packed(solid_packed, fog.packed, t256.min(256))
                        }
                    } else {
                        solid_packed
                    };
                }
                iz += tri.diz_dx;
            }
        }
    }
}

fn scan_textured<const POW2: bool, const FOG: bool>(
    band: &mut Band<'_>,
    y0: usize,
    y1: usize,
    tri: &ProjectedTri,
    tex: &Texture,
    fog: FogParams,
) {
    let shade256 = tri.shade256;
    let tw = tex.width;
    let tw_i32 = tex.width as i32;
    let th_i32 = tex.height as i32;

    for y in y0..=y1 {
        let dy = (y - tri.min_y) as f32;
        let e0_row = tri.e0_base + tri.dy0 * dy;
        let e1_row = tri.e1_base + tri.dy1 * dy;
        let e2_row = tri.e2_base + tri.dy2 * dy;
        if let Some((x_start, x_end)) = scanline_x_range(tri, e0_row, e1_row, e2_row) {
            let iz_row = tri.iz_base + tri.diz_dy * dy;
            let uz_row = tri.uz_base + tri.duz_dy * dy;
            let vz_row = tri.vz_base + tri.dvz_dy * dy;
            let local_row = (y - band.y0) * band.width;
            let off = (x_start - tri.min_x) as f32;
            let mut iz = iz_row + tri.diz_dx * off;
            let mut uz = uz_row + tri.duz_dx * off;
            let mut vz = vz_row + tri.dvz_dx * off;

            let mut z0 = 1.0 / iz.max(1e-7);
            let mut u0 = uz * z0;
            let mut v0 = vz * z0;

            let mut x = x_start;
            let total_end = x_end + 1;
            while x < total_end {
                let remaining = total_end - x;
                let chunk = remaining.min(SPAN_SIZE);
                let end_x = x + chunk;

                let (du, dv, next_state) = if remaining > SPAN_SIZE {
                    let iz_n = iz + tri.diz_dx * (SPAN_SIZE as f32);
                    let uz_n = uz + tri.duz_dx * (SPAN_SIZE as f32);
                    let vz_n = vz + tri.dvz_dx * (SPAN_SIZE as f32);
                    let z1 = 1.0 / iz_n.max(1e-7);
                    let u1 = uz_n * z1;
                    let v1 = vz_n * z1;
                    (
                        (u1 - u0) * INV_SPAN_SIZE,
                        (v1 - v0) * INV_SPAN_SIZE,
                        Some((iz_n, uz_n, vz_n, z1, u1, v1)),
                    )
                } else if chunk > 1 {
                    let steps = (chunk - 1) as f32;
                    let iz_e = iz + tri.diz_dx * steps;
                    let uz_e = uz + tri.duz_dx * steps;
                    let vz_e = vz + tri.dvz_dx * steps;
                    let z1 = 1.0 / iz_e.max(1e-7);
                    let u1 = uz_e * z1;
                    let v1 = vz_e * z1;
                    let inv_s = 1.0 / steps;
                    ((u1 - u0) * inv_s, (v1 - v0) * inv_s, None)
                } else {
                    (0.0, 0.0, None)
                };

                // Pick mipmap level once per 16-pixel span from exact screen-space texel footprint.
                let (pixels, w_mask, h_mask, w_shift, uv_shift) = if POW2 {
                    let du_dy = (tri.duz_dy - u0 * tri.diz_dy) * z0;
                    let dv_dy = (tri.dvz_dy - v0 * tri.diz_dy) * z0;
                    let rho2 = (du * du + dv * dv).max(du_dy * du_dy + dv_dy * dv_dy);
                    let mip = mip_from_rho2(rho2);
                    let (px, wm, hm, ws, actual_mip) = tex.pow2_mip(mip);
                    (px, wm, hm, ws, 16 + actual_mip)
                } else {
                    (tex.pixels.as_slice(), 0, 0, 0, 16)
                };

                // 48.16 fixed-point coordinates in i64: immune to large Doom map UV coordinates.
                let mut u_fp = (u0 * 65536.0).floor() as i64;
                let mut v_fp = (v0 * 65536.0).floor() as i64;
                let du_fp = (du * 65536.0).round() as i64;
                let dv_fp = (dv * 65536.0).round() as i64;

                let inv_z_slice = &mut band.inv_z[local_row + x..local_row + end_x];
                let color_slice = &mut band.color[local_row + x..local_row + end_x];
                let mut cur_iz = iz;

                for (iz_cell, col_cell) in inv_z_slice.iter_mut().zip(color_slice.iter_mut()) {
                    if cur_iz > *iz_cell {
                        *iz_cell = cur_iz;
                        if FOG && cur_iz <= fog.iz_far {
                            *col_cell = fog.packed;
                        } else {
                            let idx = if POW2 {
                                let tx = ((u_fp >> uv_shift) as usize) & w_mask;
                                let ty = ((v_fp >> uv_shift) as usize) & h_mask;
                                (ty << w_shift) | tx
                            } else {
                                let tx = ((u_fp >> 16) as i32).rem_euclid(tw_i32) as usize;
                                let ty = ((v_fp >> 16) as i32).rem_euclid(th_i32) as usize;
                                ty * tw + tx
                            };
                            let lit = scale_packed(pixels[idx], shade256);
                            *col_cell = if FOG && cur_iz < fog.iz_near {
                                let t256 = ((fog.iz_near - cur_iz) * fog.inv_iz_span) as u32;
                                lerp_packed(lit, fog.packed, t256.min(256))
                            } else {
                                lit
                            };
                        }
                    }
                    cur_iz += tri.diz_dx;
                    u_fp = u_fp.wrapping_add(du_fp);
                    v_fp = v_fp.wrapping_add(dv_fp);
                }

                if let Some((iz_n, uz_n, vz_n, z1, u1, v1)) = next_state {
                    iz = iz_n;
                    uz = uz_n;
                    vz = vz_n;
                    z0 = z1;
                    u0 = u1;
                    v0 = v1;
                }
                x = end_x;
            }
        }
    }
}

/// Fast `floor(log2(sqrt(rho2)))` using IEEE-754 exponent bits (zero transcendental calls).
#[inline(always)]
fn mip_from_rho2(rho2: f32) -> usize {
    if rho2 <= 1.0 || !rho2.is_finite() {
        return 0;
    }
    let exp = ((rho2.to_bits() >> 23) & 0xFF) as i32 - 127;
    (exp >> 1).max(0) as usize
}

#[inline(always)]
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (px - ax) * (by - ay) - (py - ay) * (bx - ax)
}

#[inline(always)]
fn intersect_near(a: ClipVert, b: ClipVert, near: f32) -> ClipVert {
    let t = (near - a.pos.z) / (b.pos.z - a.pos.z);
    let mut pos = a.pos.lerp(b.pos, t);
    pos.z = near;
    let uv = [
        a.uv[0] + (b.uv[0] - a.uv[0]) * t,
        a.uv[1] + (b.uv[1] - a.uv[1]) * t,
    ];
    ClipVert { pos, uv }
}

/// Clips a view-space triangle against the near plane (z = near), interpolating UVs.
fn clip_near(
    v0: ClipVert,
    v1: ClipVert,
    v2: ClipVert,
    near: f32,
) -> ([(ClipVert, ClipVert, ClipVert); 2], usize) {
    let i0 = v0.pos.z >= near;
    let i1 = v1.pos.z >= near;
    let i2 = v2.pos.z >= near;
    let inside = i0 as usize + i1 as usize + i2 as usize;

    let z = ClipVert {
        pos: Vec3::ZERO,
        uv: [0.0, 0.0],
    };
    let dummy = (z, z, z);
    match inside {
        0 => ([dummy, dummy], 0),
        3 => ([(v0, v1, v2), dummy], 1),
        1 => {
            let (a, b, c) = if i0 {
                (v0, v1, v2)
            } else if i1 {
                (v1, v2, v0)
            } else {
                (v2, v0, v1)
            };
            let ab = intersect_near(a, b, near);
            let ac = intersect_near(a, c, near);
            ([(a, ab, ac), dummy], 1)
        }
        _ => {
            // Two vertices inside: rotate so `c` is the clipped vertex
            let (a, b, c) = if !i2 {
                (v0, v1, v2)
            } else if !i0 {
                (v1, v2, v0)
            } else {
                (v2, v0, v1)
            };
            let bc = intersect_near(b, c, near);
            let ac = intersect_near(a, c, near);
            ([(a, b, bc), (a, bc, ac)], 2)
        }
    }
}

#[inline(always)]
fn scale_rgb(c: [u8; 3], k: f32) -> [u8; 3] {
    [
        (c[0] as f32 * k).clamp(0.0, 255.0) as u8,
        (c[1] as f32 * k).clamp(0.0, 255.0) as u8,
        (c[2] as f32 * k).clamp(0.0, 255.0) as u8,
    ]
}

/// Scales a packed 0x00RRGGBB color by `shade256` in 0..=256 using integer arithmetic.
#[inline(always)]
fn scale_packed(c: u32, shade256: u32) -> u32 {
    let rb = (((c & 0x00FF_00FF) * shade256) >> 8) & 0x00FF_00FF;
    let g = (((c & 0x0000_FF00) * shade256) >> 8) & 0x0000_FF00;
    rb | g
}

/// Blends two packed 0x00RRGGBB colors with t256 in 0..=256 using integer arithmetic.
#[inline(always)]
fn lerp_packed(a: u32, b: u32, t256: u32) -> u32 {
    let inv = 256 - t256;
    let rb = (((a & 0x00FF_00FF) * inv + (b & 0x00FF_00FF) * t256) >> 8) & 0x00FF_00FF;
    let g = (((a & 0x0000_FF00) * inv + (b & 0x0000_FF00) * t256) >> 8) & 0x0000_FF00;
    rb | g
}

#[inline(always)]
fn pack_rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::build_demo_room;

    fn test_cam() -> Camera {
        Camera {
            pos: Vec3::new(0.0, 1.6, 4.0),
            yaw: 0.0,
            pitch: -0.08,
            fov_y_rad: 60.0_f32.to_radians(),
            near: 0.05,
        }
    }

    fn test_cfg() -> RenderConfig {
        RenderConfig {
            debug_slivers: false,
            textures_enabled: true,
            fog_enabled: true,
            fog_color: [16, 20, 26],
            fog_start: 4.0,
            fog_end: 20.0,
            light_dir: Vec3::new(0.4, 0.8, 0.3).normalize(),
            sky_tex_id: None,
        }
    }

    #[test]
    fn multithreaded_bands_match_singlethreaded_bit_for_bit() {
        let mut tris = build_demo_room();
        let tex = Texture::new(
            64,
            64,
            (0..64 * 64)
                .map(|i| {
                    let x = (i & 63) as u32;
                    let y = (i >> 6) as u32;
                    ((x * 4) << 16) | ((y * 4) << 8) | (((x ^ y) * 4) & 0xFF)
                })
                .collect(),
        );
        // Texture half of the room triangles with negative and large UVs to stress span + mip code.
        for (idx, t) in tris.iter_mut().enumerate() {
            if idx % 2 == 0 {
                t.tex_id = 0;
                t.uv0 = [-4000.0, -4000.0];
                t.uv1 = [4000.0, -4000.0];
                t.uv2 = [4000.0, 4000.0];
                t.light = 220;
            }
        }

        let cam = test_cam();
        let cfg = test_cfg();

        let mut fb_st = Framebuffer::new(320, 180);
        fb_st.num_threads = 1;
        fb_st.clear(cfg.fog_color);
        let drawn_st = fb_st.draw_triangles(&tris, std::slice::from_ref(&tex), &cam, &cfg);

        let mut fb_mt = Framebuffer::new(320, 180);
        fb_mt.num_threads = 4;
        fb_mt.clear(cfg.fog_color);
        let drawn_mt = fb_mt.draw_triangles(&tris, std::slice::from_ref(&tex), &cam, &cfg);

        assert!(drawn_st >= MT_TRI_THRESHOLD);
        assert_eq!(drawn_st, drawn_mt);
        assert_eq!(fb_st.color, fb_mt.color);
        assert_eq!(fb_st.inv_z, fb_mt.inv_z);
    }

    #[test]
    fn sky_triangles_draw_panorama_and_occlude_behind() {
        let sky_color = 0x0044_88CC;
        let sky_tex = Texture::new(64, 64, vec![sky_color; 64 * 64]);

        // Closer sky quad at z = -2 occluding a farther red wall at z = -6.
        let tris = vec![
            Tri::new_textured(
                Vec3::new(-2.0, -2.0, -2.0),
                Vec3::new(2.0, -2.0, -2.0),
                Vec3::new(2.0, 2.0, -2.0),
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [100, 140, 180],
                SKY_TEX,
                255,
            ),
            Tri::new_textured(
                Vec3::new(-2.0, -2.0, -2.0),
                Vec3::new(2.0, 2.0, -2.0),
                Vec3::new(-2.0, 2.0, -2.0),
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [100, 140, 180],
                SKY_TEX,
                255,
            ),
            Tri::new(
                Vec3::new(-4.0, -4.0, -6.0),
                Vec3::new(4.0, -4.0, -6.0),
                Vec3::new(4.0, 4.0, -6.0),
                [255, 0, 0],
            ),
        ];

        let cam = Camera {
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fov_y_rad: 60.0_f32.to_radians(),
            near: 0.1,
        };
        let mut cfg = test_cfg();
        cfg.fog_start = 0.5;
        cfg.fog_end = 1.5; // Beyond fog_end: sky must still stay un-fogged.
        cfg.sky_tex_id = Some(0);

        let mut fb = Framebuffer::new(120, 80);
        fb.clear(cfg.fog_color);
        fb.draw_triangles(&tris, std::slice::from_ref(&sky_tex), &cam, &cfg);

        let center = (80 / 2) * 120 + (120 / 2);
        assert_eq!(fb.color[center], sky_color);
        assert!(fb.inv_z[center] > 0.45);
    }

    #[test]
    fn mip_selector_matches_expected_thresholds() {
        assert_eq!(mip_from_rho2(0.25), 0);
        assert_eq!(mip_from_rho2(1.0), 0);
        assert_eq!(mip_from_rho2(3.9), 0);
        assert_eq!(mip_from_rho2(4.0), 1);
        assert_eq!(mip_from_rho2(16.0), 2);
        assert_eq!(mip_from_rho2(64.0), 3);
        assert_eq!(mip_from_rho2(f32::NAN), 0);
    }
}