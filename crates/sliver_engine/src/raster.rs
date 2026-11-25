use crate::scene::Tri;
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
    pub fog_enabled: bool,
    pub fog_color: [u8; 3],
    pub fog_start: f32,
    pub fog_end: f32,
    pub light_dir: Vec3,
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

    pub fn draw_triangles(&mut self, tris: &[Tri], cam: &Camera, cfg: &RenderConfig) -> usize {
        let (right, up, forward) = cam.basis();
        let focal = (self.height as f32 * 0.5) / (cam.fov_y_rad * 0.5).tan();
        let fog_packed = pack_rgb(cfg.fog_color[0], cfg.fog_color[1], cfg.fog_color[2]);
        let iz_near = 1.0 / cfg.fog_start.max(1e-4);
        let iz_far = 1.0 / cfg.fog_end.max(cfg.fog_start + 1e-4);
        let inv_iz_span = 256.0 / (iz_near - iz_far);
        let mut drawn = 0;

        for tri in tris {
            let n = tri.normal();
            // World-space backface culling
            if n.dot(cam.pos - tri.p0) <= 0.0 {
                continue;
            }

            // Hemispherical ambient + directional diffuse shading
            let ndotl = n.dot(cfg.light_dir).max(0.0);
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

            let v0 = cam.world_to_view(tri.p0, right, up, forward);
            let v1 = cam.world_to_view(tri.p1, right, up, forward);
            let v2 = cam.world_to_view(tri.p2, right, up, forward);

            let (clipped, count) = clip_near(v0, v1, v2, cam.near);
            for c in &clipped[..count] {
                if self.rasterize_view_tri(
                    c.0,
                    c.1,
                    c.2,
                    focal,
                    solid_packed,
                    fog_packed,
                    cfg.fog_enabled,
                    cfg.fog_start,
                    iz_near,
                    iz_far,
                    inv_iz_span,
                ) {
                    drawn += 1;
                }
            }
        }
        drawn
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize_view_tri(
        &mut self,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        focal: f32,
        solid_packed: u32,
        fog_packed: u32,
        fog_enabled: bool,
        fog_start: f32,
        iz_near: f32,
        iz_far: f32,
        inv_iz_span: f32,
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

        let need_fog = fog_enabled && v0.z.max(v1.z).max(v2.z) > fog_start;

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
                            *col_cell = if iz >= iz_near {
                                solid_packed
                            } else if iz <= iz_far {
                                fog_packed
                            } else {
                                let t256 = ((iz_near - iz) * inv_iz_span) as u32;
                                lerp_packed(solid_packed, fog_packed, t256.min(256))
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
}

#[inline(always)]
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    (px - ax) * (by - ay) - (py - ay) * (bx - ax)
}

#[inline(always)]
fn intersect_near(a: Vec3, b: Vec3, near: f32) -> Vec3 {
    let t = (near - a.z) / (b.z - a.z);
    let mut p = a.lerp(b, t);
    p.z = near;
    p
}

/// Clips a view-space triangle against the near plane (z = near).
fn clip_near(v0: Vec3, v1: Vec3, v2: Vec3, near: f32) -> ([(Vec3, Vec3, Vec3); 2], usize) {
    let i0 = v0.z >= near;
    let i1 = v1.z >= near;
    let i2 = v2.z >= near;
    let inside = i0 as usize + i1 as usize + i2 as usize;

    let dummy = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
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
