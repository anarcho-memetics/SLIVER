//! Lightweight, zero-allocation rolling frame profiler and framebuffer HUD overlay.

use std::time::Instant;

const WINDOW: usize = 60;

#[derive(Clone, Copy, Default, Debug)]
pub struct FrameSample {
    pub phys_ms: f32,
    pub clear_ms: f32,
    pub rast_ms: f32,
    pub present_ms: f32,
    pub frame_ms: f32,
}

impl FrameSample {
    #[inline]
    pub fn work_ms(self) -> f32 {
        self.phys_ms + self.clear_ms + self.rast_ms
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct PerfSummary {
    pub fps: f32,
    pub phys_avg: f32,
    pub clear_avg: f32,
    pub rast_avg: f32,
    pub present_avg: f32,
    pub work_avg: f32,
    pub work_max: f32,
    pub frame_p95: f32,
    pub frame_max: f32,
}

pub struct PerfMonitor {
    samples: [FrameSample; WINDOW],
    cursor: usize,
    filled: usize,
    last_log: Instant,
    pub hud_enabled: bool,
    pub log_enabled: bool,
}

impl PerfMonitor {
    pub fn new() -> Self {
        Self {
            samples: [FrameSample::default(); WINDOW],
            cursor: 0,
            filled: 0,
            last_log: Instant::now(),
            hud_enabled: false,
            log_enabled: false,
        }
    }

    pub fn push(&mut self, sample: FrameSample) {
        self.samples[self.cursor] = sample;
        self.cursor = (self.cursor + 1) % WINDOW;
        if self.filled < WINDOW {
            self.filled += 1;
        }
    }

    pub fn summary(&self) -> PerfSummary {
        let n = self.filled;
        if n == 0 {
            return PerfSummary::default();
        }
        let inv_n = 1.0 / (n as f32);
        let mut phys = 0.0;
        let mut clear = 0.0;
        let mut rast = 0.0;
        let mut present = 0.0;
        let mut frame_sum = 0.0;
        let mut work_max = 0.0f32;
        let mut frames = [0.0f32; WINDOW];

        for (i, s) in self.samples[..n].iter().enumerate() {
            phys += s.phys_ms;
            clear += s.clear_ms;
            rast += s.rast_ms;
            present += s.present_ms;
            frame_sum += s.frame_ms;
            work_max = work_max.max(s.work_ms());
            frames[i] = s.frame_ms;
        }

        let slice = &mut frames[..n];
        slice.sort_unstable_by(|a, b| a.total_cmp(b));
        let p95_idx = ((n - 1) * 95) / 100;
        let frame_avg = frame_sum * inv_n;

        PerfSummary {
            fps: if frame_avg > 0.001 {
                1000.0 / frame_avg
            } else {
                0.0
            },
            phys_avg: phys * inv_n,
            clear_avg: clear * inv_n,
            rast_avg: rast * inv_n,
            present_avg: present * inv_n,
            work_avg: (phys + clear + rast) * inv_n,
            work_max,
            frame_p95: slice[p95_idx],
            frame_max: slice[n - 1],
        }
    }

    /// Prints a one-line breakdown to stdout once per second when `log_enabled` is true.
    pub fn maybe_log(&mut self, drawn_tris: usize, total_tris: usize) {
        if !self.log_enabled || self.last_log.elapsed().as_secs_f32() < 1.0 {
            return;
        }
        self.last_log = Instant::now();
        let s = self.summary();
        println!(
            "[PERF] {:4.1} FPS | work {:5.2} ms (max {:5.2}) | phys {:4.2} | clr {:4.2} | rast {:5.2} | pres {:5.2} | dt p95 {:5.2} max {:5.2} | tris {}/{}",
            s.fps,
            s.work_avg,
            s.work_max,
            s.phys_avg,
            s.clear_avg,
            s.rast_avg,
            s.present_avg,
            s.frame_p95,
            s.frame_max,
            drawn_tris,
            total_tris,
        );
    }

    /// Draws a compact stacked frame-time graph in the bottom-left corner of the framebuffer.
    /// Colors: cyan = phys, yellow = clear, red = raster. White line = 16.67 ms (60 FPS budget).
    pub fn draw_hud(&self, color: &mut [u32], fb_w: usize, fb_h: usize) {
        if !self.hud_enabled || self.filled == 0 {
            return;
        }
        const COL_W: usize = 3;
        const GRAPH_W: usize = WINDOW * COL_W;
        const GRAPH_H: usize = 68;
        const PAD: usize = 10;
        const MAX_MS: f32 = 33.33;

        if fb_w < GRAPH_W + PAD * 2 || fb_h < GRAPH_H + PAD * 2 {
            return;
        }
        let x0 = PAD;
        let y0 = fb_h - PAD - GRAPH_H;

        // Darken background panel (~50% brightness) so bars stay readable over bright flats.
        for y in y0..y0 + GRAPH_H {
            let row = y * fb_w;
            for px in &mut color[row + x0..row + x0 + GRAPH_W] {
                *px = (*px >> 1) & 0x007F_7F7F;
            }
        }

        // Draw oldest -> newest from left to right.
        let oldest = if self.filled < WINDOW { 0 } else { self.cursor };
        let to_px = |ms: f32| -> usize {
            ((ms / MAX_MS) * (GRAPH_H as f32))
                .round()
                .clamp(0.0, GRAPH_H as f32) as usize
        };

        for i in 0..self.filled {
            let s = self.samples[(oldest + i) % WINDOW];
            let h_phys = to_px(s.phys_ms);
            let h_clear = to_px(s.phys_ms + s.clear_ms);
            let h_work = to_px(s.work_ms());
            let cx = x0 + i * COL_W;

            for dy in 0..h_work {
                let py = y0 + GRAPH_H - 1 - dy;
                let c = if dy < h_phys {
                    0x003E_86BD // phys (blue)
                } else if dy < h_clear {
                    0x00F2_C94C // clear (yellow)
                } else {
                    0x00EB_5757 // raster (red)
                };
                let row = py * fb_w;
                for dx in 0..COL_W.saturating_sub(1) {
                    color[row + cx + dx] = c;
                }
            }
        }

        // 16.67 ms (60 FPS) budget reference line.
        let budget_y = y0 + GRAPH_H - 1 - to_px(16.67);
        let row = budget_y * fb_w;
        for px in &mut color[row + x0..row + x0 + GRAPH_W] {
            *px = 0x00FF_FFFF;
        }
    }
}

impl Default for PerfMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_computes_avg_max_and_p95() {
        let mut mon = PerfMonitor::new();
        for i in 0..60 {
            let spike = if i == 59 { 20.0 } else { 4.0 };
            mon.push(FrameSample {
                phys_ms: 1.0,
                clear_ms: 0.5,
                rast_ms: spike,
                present_ms: 8.0,
                frame_ms: 16.0 + if i == 59 { 10.0 } else { 0.0 },
            });
        }
        let s = mon.summary();
        assert!((s.work_max - 21.5).abs() < 1e-4);
        assert!((s.frame_max - 26.0).abs() < 1e-4);
        assert!((s.frame_p95 - 16.0).abs() < 1e-4);
    }
}
