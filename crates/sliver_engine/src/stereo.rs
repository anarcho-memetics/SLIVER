use crate::raster::{Camera, Framebuffer, RenderConfig};
use crate::scene::{Texture, Tri};

const INVALID_PX: u32 = u32::MAX;

/// Configuration for side-by-side (SBS) stereoscopic rendering and Cardboard barrel warp.
#[derive(Clone, Copy, Debug)]
pub struct StereoConfig {
    pub enabled: bool,
    /// Interpupillary distance in meters (typical adult: ~0.064 m).
    pub ipd_m: f32,
    /// Quadratic radial barrel distortion coefficient (`0.0` disables lens warp).
    pub k1: f32,
    /// Quartic radial barrel distortion coefficient.
    pub k2: f32,
    /// Uniform scale applied inside the lens viewport (`1.0` = 1:1 center texels).
    pub zoom: f32,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self::cardboard()
    }
}

impl StereoConfig {
    /// Cardboard-oriented defaults (barrel warp enabled when `enabled` is toggled on).
    pub const fn cardboard() -> Self {
        Self {
            enabled: false,
            ipd_m: 0.064,
            k1: 0.24,
            k2: 0.04,
            zoom: 1.0,
        }
    }

    /// Flat side-by-side without lens distortion.
    pub const fn flat() -> Self {
        Self {
            enabled: false,
            ipd_m: 0.064,
            k1: 0.0,
            k2: 0.0,
            zoom: 1.0,
        }
    }

    #[inline]
    pub fn has_warp(&self) -> bool {
        self.k1 != 0.0 || self.k2 != 0.0 || (self.zoom - 1.0).abs() > 1e-5
    }
}

/// Borrowed scene data rendered twice (left and right eye) by `StereoRenderer`.
pub struct StereoScene<'a> {
    pub world_tris: &'a [Tri],
    pub extra_tris: &'a [Tri],
    pub textures: &'a [Texture],
    pub cam: &'a Camera,
    pub cfg: &'a RenderConfig,
}

/// Reusable half-width stereo renderer with cached inverse barrel-distortion LUT.
pub struct StereoRenderer {
    eye_fb: Option<Framebuffer>,
    lut_key: (usize, usize, u32, u32, u32),
    lut: Vec<u32>,
}

impl Default for StereoRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoRenderer {
    pub fn new() -> Self {
        Self {
            eye_fb: None,
            lut_key: (0, 0, 0, 0, 0),
            lut: Vec::new(),
        }
    }

    /// Renders left and right eyes into `out_color` (`out_w x out_h`, `0x00RRGGBB`).
    /// Returns the total number of projected triangles across both eyes.
    pub fn render(
        &mut self,
        out_color: &mut [u32],
        out_w: usize,
        out_h: usize,
        scene: &StereoScene<'_>,
        stereo: &StereoConfig,
    ) -> usize {
        if out_w < 2 || out_h == 0 || out_color.len() < out_w * out_h {
            return 0;
        }
        let eye_w = out_w / 2;
        let need_fb = self
            .eye_fb
            .as_ref()
            .is_none_or(|fb| fb.width != eye_w || fb.height != out_h);
        if need_fb {
            self.eye_fb = Some(Framebuffer::new(eye_w, out_h));
        }

        let use_warp = stereo.has_warp();
        if use_warp {
            let key = (
                eye_w,
                out_h,
                stereo.k1.to_bits(),
                stereo.k2.to_bits(),
                stereo.zoom.to_bits(),
            );
            if key != self.lut_key {
                build_barrel_lut(&mut self.lut, eye_w, out_h, stereo);
                self.lut_key = key;
            }
        }

        // Clear trailing 1 px column if output width is odd.
        if out_w > eye_w * 2 {
            for y in 0..out_h {
                out_color[y * out_w + eye_w * 2] = 0;
            }
        }

        let (right, _, _) = scene.cam.basis();
        let half_ipd = stereo.ipd_m.clamp(0.0, 0.15) * 0.5;
        let eye_fb = self.eye_fb.as_mut().unwrap();
        let mut total_drawn = 0usize;

        for (eye_idx, sign) in [(-1.0_f32), 1.0_f32].into_iter().enumerate() {
            let eye_cam = Camera {
                pos: scene.cam.pos + right * (half_ipd * sign),
                yaw: scene.cam.yaw,
                pitch: scene.cam.pitch,
                fov_y_rad: scene.cam.fov_y_rad,
                near: scene.cam.near,
            };

            eye_fb.clear(scene.cfg.fog_color);
            total_drawn +=
                eye_fb.draw_triangles(scene.world_tris, scene.textures, &eye_cam, scene.cfg);
            if !scene.extra_tris.is_empty() {
                total_drawn += eye_fb.draw_triangles(scene.extra_tris, &[], &eye_cam, scene.cfg);
            }

            let dst_x0 = eye_idx * eye_w;
            if use_warp {
                blit_eye_warped(
                    &eye_fb.color,
                    &self.lut,
                    out_color,
                    out_w,
                    eye_w,
                    out_h,
                    dst_x0,
                );
            } else {
                blit_eye_flat(&eye_fb.color, out_color, out_w, eye_w, out_h, dst_x0);
            }
        }

        total_drawn
    }
}

/// Precomputes an inverse Brown-Conrady barrel-distortion address map for one eye.
fn build_barrel_lut(lut: &mut Vec<u32>, eye_w: usize, eye_h: usize, stereo: &StereoConfig) {
    lut.clear();
    lut.reserve(eye_w * eye_h);

    let wf = eye_w as f32;
    let hf = eye_h as f32;
    let cx = wf * 0.5;
    let cy = hf * 0.5;
    let inv_r = 1.0 / cx.min(cy).max(1.0);
    let inv_zoom = 1.0 / stereo.zoom.max(0.25);

    for y in 0..eye_h {
        let py = (y as f32 + 0.5) - cy;
        let ny = py * inv_r;
        let ny2 = ny * ny;
        for x in 0..eye_w {
            let px = (x as f32 + 0.5) - cx;
            let nx = px * inv_r;
            let r2 = nx * nx + ny2;
            let scale = (1.0 + stereo.k1 * r2 + stereo.k2 * r2 * r2) * inv_zoom;
            let sx = cx + px * scale;
            let sy = cy + py * scale;
            if sx >= 0.0 && sx < wf && sy >= 0.0 && sy < hf {
                let ix = (sx as usize).min(eye_w - 1);
                let iy = (sy as usize).min(eye_h - 1);
                lut.push((iy * eye_w + ix) as u32);
            } else {
                lut.push(INVALID_PX);
            }
        }
    }
}

fn blit_eye_flat(
    src: &[u32],
    dst: &mut [u32],
    out_w: usize,
    eye_w: usize,
    eye_h: usize,
    dst_x0: usize,
) {
    for y in 0..eye_h {
        let src_row = &src[y * eye_w..(y + 1) * eye_w];
        let dst_off = y * out_w + dst_x0;
        dst[dst_off..dst_off + eye_w].copy_from_slice(src_row);
    }
}

fn blit_eye_warped(
    src: &[u32],
    lut: &[u32],
    dst: &mut [u32],
    out_w: usize,
    eye_w: usize,
    eye_h: usize,
    dst_x0: usize,
) {
    for y in 0..eye_h {
        let lut_row = &lut[y * eye_w..(y + 1) * eye_w];
        let dst_off = y * out_w + dst_x0;
        let dst_row = &mut dst[dst_off..dst_off + eye_w];
        for (dst_px, &src_idx) in dst_row.iter_mut().zip(lut_row.iter()) {
            *dst_px = if src_idx == INVALID_PX {
                0
            } else {
                src[src_idx as usize]
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sliver_math::Vec3;

    fn front_quad() -> Vec<Tri> {
        let col = [220, 80, 60];
        vec![
            Tri::new(
                Vec3::new(-0.4, -0.5, -2.0),
                Vec3::new(0.4, -0.5, -2.0),
                Vec3::new(0.4, 0.5, -2.0),
                col,
            ),
            Tri::new(
                Vec3::new(-0.4, -0.5, -2.0),
                Vec3::new(0.4, 0.5, -2.0),
                Vec3::new(-0.4, 0.5, -2.0),
                col,
            ),
        ]
    }

    fn base_cam() -> Camera {
        Camera {
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fov_y_rad: 68.0_f32.to_radians(),
            near: 0.1,
        }
    }

    fn base_cfg() -> RenderConfig {
        RenderConfig {
            debug_slivers: false,
            textures_enabled: false,
            fog_enabled: false,
            fog_color: [10, 12, 16],
            fog_start: 4.0,
            fog_end: 20.0,
            light_dir: Vec3::new(0.0, 0.0, 1.0),
            sky_tex_id: None,
        }
    }

    #[test]
    fn barrel_lut_preserves_center_clips_corners_and_is_symmetric() {
        let cfg = StereoConfig::cardboard();
        let (w, h) = (120, 120);
        let mut lut = Vec::new();
        build_barrel_lut(&mut lut, w, h, &cfg);
        assert_eq!(lut.len(), w * h);

        // Center pixels map cleanly onto themselves.
        let center_idx = (h / 2) * w + (w / 2);
        assert_eq!(lut[center_idx], center_idx as u32);

        // Extreme corners are vignetted outside the pre-warped lens circle.
        assert_eq!(lut[0], INVALID_PX);
        assert_eq!(lut[w - 1], INVALID_PX);
        assert_eq!(lut[(h - 1) * w], INVALID_PX);
        assert_eq!(lut[h * w - 1], INVALID_PX);

        // Horizontal and vertical symmetry of valid mask.
        for y in 0..h {
            for x in 0..w {
                let a = lut[y * w + x] == INVALID_PX;
                let b = lut[y * w + (w - 1 - x)] == INVALID_PX;
                let c = lut[(h - 1 - y) * w + x] == INVALID_PX;
                assert_eq!(a, b);
                assert_eq!(a, c);
            }
        }
    }

    #[test]
    fn sbs_produces_horizontal_parallax_between_eyes() {
        let tris = front_quad();
        let cam = base_cam();
        let cfg = base_cfg();
        let mut stereo = StereoConfig::flat();
        stereo.enabled = true;
        stereo.ipd_m = 0.20;

        let (out_w, out_h) = (240, 120);
        let eye_w = out_w / 2;
        let mut out = vec![0u32; out_w * out_h];
        let mut renderer = StereoRenderer::new();
        let drawn = renderer.render(
            &mut out,
            out_w,
            out_h,
            &StereoScene {
                world_tris: &tris,
                extra_tris: &[],
                textures: &[],
                cam: &cam,
                cfg: &cfg,
            },
            &stereo,
        );
        assert_eq!(drawn, 4);

        let bg = (10u32 << 16) | (12u32 << 8) | 16u32;
        let mid_row = (out_h / 2) * out_w;
        let left_row = &out[mid_row..mid_row + eye_w];
        let right_row = &out[mid_row + eye_w..mid_row + out_w];

        let left_first = left_row.iter().position(|&c| c != bg).unwrap();
        let right_first = right_row.iter().position(|&c| c != bg).unwrap();
        // Left eye is shifted left, so the object appears farther right than in the right eye.
        assert!(left_first > right_first);
    }

    #[test]
    fn sbs_zero_ipd_matches_both_eyes_bit_for_bit() {
        let tris = front_quad();
        let cam = base_cam();
        let cfg = base_cfg();
        let mut stereo = StereoConfig::cardboard();
        stereo.enabled = true;
        stereo.ipd_m = 0.0;

        let (out_w, out_h) = (240, 120);
        let eye_w = out_w / 2;
        let mut out = vec![0u32; out_w * out_h];
        let mut renderer = StereoRenderer::new();
        renderer.render(
            &mut out,
            out_w,
            out_h,
            &StereoScene {
                world_tris: &tris,
                extra_tris: &[],
                textures: &[],
                cam: &cam,
                cfg: &cfg,
            },
            &stereo,
        );

        for y in 0..out_h {
            let row = y * out_w;
            assert_eq!(
                &out[row..row + eye_w],
                &out[row + eye_w..row + out_w],
                "row {y} mismatch between eyes at zero IPD"
            );
        }
    }
}
