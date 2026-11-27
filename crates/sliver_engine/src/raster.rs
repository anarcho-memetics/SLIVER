use crate::scene::{Texture, Tri, NO_TEX};
use sliver_math::Vec3;

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

pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    pub color: Vec<u32>,
    pub inv_z: Vec<f32>,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            color: vec![0; width * height],
            inv_z: vec![0.0; width * height],
        }
    }

    pub fn clear(&mut self, rgb: [u8; 3]) {
        let packed = pack_rgb(rgb[0], rgb[1], rgb[2]);
        self.color.fill(packed);
        self.inv_z.fill(0.0);
    }

    pub fn draw_triangles(
        &mut self,
        tris: &[Tri],
        textures: &[Texture],
        cam: &Camera,
        cfg: &RenderConfig,
    ) -> usize {
        let (right, up, forward) = cam.basis();
        let focal = (self.height as f32 * 0.5) / (cam.fov_y_rad * 0.5).tan();
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
        let mut drawn = 0;

        for tri in tris {
            let n = tri.normal();
            // World-space backface culling
            if n.dot(cam.pos - tri.p0) <= 0.0 {
                continue;
            }

            let ndotl = n.dot(cfg.light_dir).max(0.0);
            let tex = if cfg.textures_enabled && !cfg.debug_slivers && tri.tex_id != NO_TEX {
                textures.get(tri.tex_id as usize)
            } else {
                None
            };

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

            if let Some(tex) = tex {
                let light_norm = (tri.light as f32) / 255.0;
                let sector_scale = 0.22 + 0.78 * light_norm;
                let dir_shade = (0.74 + 0.08 * (n.y * 0.5 + 0.5) + 0.18 * ndotl).min(1.0);
                let shade256 = ((sector_scale * dir_shade * 256.0).round() as u32).clamp(0, 256);

                for c in &clipped[..count] {
                    if self.rasterize_view_tri_textured(c.0, c.1, c.2, focal, tex, shade256, fog) {
                        drawn += 1;
                    }
                }
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
                let solid_packed = pack_rgb(lit[0], lit[1], lit[2]);

                for c in &clipped[..count] {
                    if self.rasterize_view_tri_solid(
                        c.0.pos,
                        c.1.pos,
                        c.2.pos,
                        focal,
                        solid_packed,
                        fog,
                    ) {
                        drawn += 1;
                    }
                }
            }
        }
        drawn
    }

    fn rasterize_view_tri_solid(
        &mut self,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        focal: f32,
        solid_packed: u32,
        fog: FogParams,
    ) -> bool {
        let wf = self.width as f32;
        let hf = self.height as f32;
        let cx = wf * 0.5;
        let cy = hf * 0.5;

        let iz0 = 1.0 / v0.z;
        let iz1 = 1.0 / v1.z;
        let iz2 = 1.0 / v2.z;

        let sx0 = cx + v0.x * iz0 * focal;
        let sy0 = cy - v0.y * iz0 * focal;
        let sx1 = cx + v1.x * iz1 * focal;
        let sy1 = cy - v1.y * iz1 * focal;
        let sx2 = cx + v2.x * iz2 * focal;
        let sy2 = cy - v2.y * iz2 * focal;

        let area = edge(sx0, sy0, sx1, sy1, sx2, sy2);
        if area <= 0.0 {
            return false;
        }

        let min_xf = sx0.min(sx1).min(sx2).floor();
        let max_xf = sx0.max(sx1).max(sx2).ceil();
        let min_yf = sy0.min(sy1).min(sy2).floor();
        let max_yf = sy0.max(sy1).max(sy2).ceil();

        if max_xf < 0.0 || min_xf >= wf || max_yf < 0.0 || min_yf >= hf {
            return false;
        }

        let min_x = min_xf.max(0.0) as usize;
        let max_x = (max_xf as usize).min(self.width - 1);
        let min_y = min_yf.max(0.0) as usize;
        let max_y = (max_yf as usize).min(self.height - 1);
        if min_x > max_x || min_y > max_y {
            return false;
        }

        let start_px = min_x as f32 + 0.5;
        let start_py = min_y as f32 + 0.5;

        let mut e0_row = edge(sx1, sy1, sx2, sy2, start_px, start_py);
        let mut e1_row = edge(sx2, sy2, sx0, sy0, start_px, start_py);
        let mut e2_row = edge(sx0, sy0, sx1, sy1, start_px, start_py);

        let dx0 = sy2 - sy1;
        let dy0 = sx1 - sx2;
        let dx1 = sy0 - sy2;
        let dy1 = sx2 - sx0;
        let dx2 = sy1 - sy0;
        let dy2 = sx0 - sx1;

        let inv_area = 1.0 / area;
        let mut iz_row = (e0_row * iz0 + e1_row * iz1 + e2_row * iz2) * inv_area;
        let diz_dx = (dx0 * iz0 + dx1 * iz1 + dx2 * iz2) * inv_area;
        let diz_dy = (dy0 * iz0 + dy1 * iz1 + dy2 * iz2) * inv_area;

        let need_fog = fog.enabled && v0.z.max(v1.z).max(v2.z) > fog.start;

        if need_fog {
            for y in min_y..=max_y {
                let row = y * self.width;
                let inv_z_slice = &mut self.inv_z[row + min_x..=row + max_x];
                let color_slice = &mut self.color[row + min_x..=row + max_x];

                let mut e0 = e0_row;
                let mut e1 = e1_row;
                let mut e2 = e2_row;
                let mut iz = iz_row;
                let mut entered = false;

                for (iz_cell, col_cell) in inv_z_slice.iter_mut().zip(color_slice.iter_mut()) {
                    if e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0 {
                        entered = true;
                        if iz > *iz_cell {
                            *iz_cell = iz;
                            *col_cell = if iz >= fog.iz_near {
                                solid_packed
                            } else if iz <= fog.iz_far {
                                fog.packed
                            } else {
                                let t256 = ((fog.iz_near - iz) * fog.inv_iz_span) as u32;
                                lerp_packed(solid_packed, fog.packed, t256.min(256))
                            };
                        }
                    } else if entered {
                        break;
                    }
                    e0 += dx0;
                    e1 += dx1;
                    e2 += dx2;
                    iz += diz_dx;
                }

                e0_row += dy0;
                e1_row += dy1;
                e2_row += dy2;
                iz_row += diz_dy;
            }
        } else {
            for y in min_y..=max_y {
                let row = y * self.width;
                let inv_z_slice = &mut self.inv_z[row + min_x..=row + max_x];
                let color_slice = &mut self.color[row + min_x..=row + max_x];

                let mut e0 = e0_row;
                let mut e1 = e1_row;
                let mut e2 = e2_row;
                let mut iz = iz_row;
                let mut entered = false;

                for (iz_cell, col_cell) in inv_z_slice.iter_mut().zip(color_slice.iter_mut()) {
                    if e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0 {
                        entered = true;
                        if iz > *iz_cell {
                            *iz_cell = iz;
                            *col_cell = solid_packed;
                        }
                    } else if entered {
                        break;
                    }
                    e0 += dx0;
                    e1 += dx1;
                    e2 += dx2;
                    iz += diz_dx;
                }

                e0_row += dy0;
                e1_row += dy1;
                e2_row += dy2;
                iz_row += diz_dy;
            }
        }

        true
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize_view_tri_textured(
        &mut self,
        v0: ClipVert,
        v1: ClipVert,
        v2: ClipVert,
        focal: f32,
        tex: &Texture,
        shade256: u32,
        fog: FogParams,
    ) -> bool {
        let wf = self.width as f32;
        let hf = self.height as f32;
        let cx = wf * 0.5;
        let cy = hf * 0.5;

        let iz0 = 1.0 / v0.pos.z;
        let iz1 = 1.0 / v1.pos.z;
        let iz2 = 1.0 / v2.pos.z;

        let sx0 = cx + v0.pos.x * iz0 * focal;
        let sy0 = cy - v0.pos.y * iz0 * focal;
        let sx1 = cx + v1.pos.x * iz1 * focal;
        let sy1 = cy - v1.pos.y * iz1 * focal;
        let sx2 = cx + v2.pos.x * iz2 * focal;
        let sy2 = cy - v2.pos.y * iz2 * focal;

        let area = edge(sx0, sy0, sx1, sy1, sx2, sy2);
        if area <= 0.0 {
            return false;
        }

        let min_xf = sx0.min(sx1).min(sx2).floor();
        let max_xf = sx0.max(sx1).max(sx2).ceil();
        let min_yf = sy0.min(sy1).min(sy2).floor();
        let max_yf = sy0.max(sy1).max(sy2).ceil();

        if max_xf < 0.0 || min_xf >= wf || max_yf < 0.0 || min_yf >= hf {
            return false;
        }

        let min_x = min_xf.max(0.0) as usize;
        let max_x = (max_xf as usize).min(self.width - 1);
        let min_y = min_yf.max(0.0) as usize;
        let max_y = (max_yf as usize).min(self.height - 1);
        if min_x > max_x || min_y > max_y {
            return false;
        }

        let start_px = min_x as f32 + 0.5;
        let start_py = min_y as f32 + 0.5;

        let e0_row = edge(sx1, sy1, sx2, sy2, start_px, start_py);
        let e1_row = edge(sx2, sy2, sx0, sy0, start_px, start_py);
        let e2_row = edge(sx0, sy0, sx1, sy1, start_px, start_py);

        let dx0 = sy2 - sy1;
        let dy0 = sx1 - sx2;
        let dx1 = sy0 - sy2;
        let dy1 = sx2 - sx0;
        let dx2 = sy1 - sy0;
        let dy2 = sx0 - sx1;

        let inv_area = 1.0 / area;
        let iz_row = (e0_row * iz0 + e1_row * iz1 + e2_row * iz2) * inv_area;
        let diz_dx = (dx0 * iz0 + dx1 * iz1 + dx2 * iz2) * inv_area;
        let diz_dy = (dy0 * iz0 + dy1 * iz1 + dy2 * iz2) * inv_area;

        let u0_iz = v0.uv[0] * iz0;
        let u1_iz = v1.uv[0] * iz1;
        let u2_iz = v2.uv[0] * iz2;
        let uz_row = (e0_row * u0_iz + e1_row * u1_iz + e2_row * u2_iz) * inv_area;
        let duz_dx = (dx0 * u0_iz + dx1 * u1_iz + dx2 * u2_iz) * inv_area;
        let duz_dy = (dy0 * u0_iz + dy1 * u1_iz + dy2 * u2_iz) * inv_area;

        let v0_iz = v0.uv[1] * iz0;
        let v1_iz = v1.uv[1] * iz1;
        let v2_iz = v2.uv[1] * iz2;
        let vz_row = (e0_row * v0_iz + e1_row * v1_iz + e2_row * v2_iz) * inv_area;
        let dvz_dx = (dx0 * v0_iz + dx1 * v1_iz + dx2 * v2_iz) * inv_area;
        let dvz_dy = (dy0 * v0_iz + dy1 * v1_iz + dy2 * v2_iz) * inv_area;

        let need_fog = fog.enabled && v0.pos.z.max(v1.pos.z).max(v2.pos.z) > fog.start;

        let span = TexSpan {
            min_x,
            max_x,
            min_y,
            max_y,
            e0_row,
            e1_row,
            e2_row,
            dx0,
            dy0,
            dx1,
            dy1,
            dx2,
            dy2,
            iz_row,
            diz_dx,
            diz_dy,
            uz_row,
            duz_dx,
            duz_dy,
            vz_row,
            dvz_dx,
            dvz_dy,
        };

        match (tex.pow2.is_some(), need_fog) {
            (true, true) => self.scan_textured::<true, true>(span, tex, shade256, fog),
            (true, false) => self.scan_textured::<true, false>(span, tex, shade256, fog),
            (false, true) => self.scan_textured::<false, true>(span, tex, shade256, fog),
            (false, false) => self.scan_textured::<false, false>(span, tex, shade256, fog),
        }

        true
    }

    fn scan_textured<const POW2: bool, const FOG: bool>(
        &mut self,
        mut s: TexSpan,
        tex: &Texture,
        shade256: u32,
        fog: FogParams,
    ) {
        let (w_mask, h_mask, w_shift) = tex.pow2.unwrap_or((0, 0, 0));
        let tw_i32 = tex.width as i32;
        let th_i32 = tex.height as i32;
        let tw = tex.width;
        let pixels = &tex.pixels;

        for y in s.min_y..=s.max_y {
            let row = y * self.width;
            let inv_z_slice = &mut self.inv_z[row + s.min_x..=row + s.max_x];
            let color_slice = &mut self.color[row + s.min_x..=row + s.max_x];

            let mut e0 = s.e0_row;
            let mut e1 = s.e1_row;
            let mut e2 = s.e2_row;
            let mut iz = s.iz_row;
            let mut uz = s.uz_row;
            let mut vz = s.vz_row;
            let mut entered = false;

            for (iz_cell, col_cell) in inv_z_slice.iter_mut().zip(color_slice.iter_mut()) {
                if e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0 {
                    entered = true;
                    if iz > *iz_cell {
                        *iz_cell = iz;
                        if FOG && iz <= fog.iz_far {
                            *col_cell = fog.packed;
                        } else {
                            let z = 1.0 / iz;
                            let iu = fast_floor_i32(uz * z);
                            let iv = fast_floor_i32(vz * z);
                            let idx = if POW2 {
                                let tx = (iu as usize) & w_mask;
                                let ty = (iv as usize) & h_mask;
                                (ty << w_shift) | tx
                            } else {
                                let tx = iu.rem_euclid(tw_i32) as usize;
                                let ty = iv.rem_euclid(th_i32) as usize;
                                ty * tw + tx
                            };
                            let lit = scale_packed(pixels[idx], shade256);
                            *col_cell = if FOG && iz < fog.iz_near {
                                let t256 = ((fog.iz_near - iz) * fog.inv_iz_span) as u32;
                                lerp_packed(lit, fog.packed, t256.min(256))
                            } else {
                                lit
                            };
                        }
                    }
                } else if entered {
                    break;
                }
                e0 += s.dx0;
                e1 += s.dx1;
                e2 += s.dx2;
                iz += s.diz_dx;
                uz += s.duz_dx;
                vz += s.dvz_dx;
            }

            s.e0_row += s.dy0;
            s.e1_row += s.dy1;
            s.e2_row += s.dy2;
            s.iz_row += s.diz_dy;
            s.uz_row += s.duz_dy;
            s.vz_row += s.dvz_dy;
        }
    }
}

#[derive(Clone, Copy)]
struct TexSpan {
    min_x: usize,
    max_x: usize,
    min_y: usize,
    max_y: usize,
    e0_row: f32,
    e1_row: f32,
    e2_row: f32,
    dx0: f32,
    dy0: f32,
    dx1: f32,
    dy1: f32,
    dx2: f32,
    dy2: f32,
    iz_row: f32,
    diz_dx: f32,
    diz_dy: f32,
    uz_row: f32,
    duz_dx: f32,
    duz_dy: f32,
    vz_row: f32,
    dvz_dx: f32,
    dvz_dy: f32,
}

#[inline(always)]
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (px - ax) * (by - ay) - (py - ay) * (bx - ax)
}

#[inline(always)]
fn fast_floor_i32(x: f32) -> i32 {
    let i = x as i32;
    i - ((x < i as f32) as i32)
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
