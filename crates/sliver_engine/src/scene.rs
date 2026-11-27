use sliver_math::{ChordSliver, Vec3};
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct Tri {
    pub p0: Vec3,
    pub p1: Vec3,
    pub p2: Vec3,
    pub color: [u8; 3],
    pub is_sliver: bool,
}

impl Tri {
    pub fn new(p0: Vec3, p1: Vec3, p2: Vec3, color: [u8; 3]) -> Self {
        Self {
            p0,
            p1,
            p2,
            color,
            is_sliver: false,
        }
    }

    #[inline]
    pub fn normal(&self) -> Vec3 {
        (self.p1 - self.p0).cross(self.p2 - self.p0).normalize()
    }
}

/// Evaluates the entire mesh with ChordSliver and returns the number of slivers found.
pub fn tag_slivers(tris: &mut [Tri], limit_deg: f32) -> usize {
    let det = ChordSliver::from_degrees(limit_deg);
    let mut count = 0;
    for t in tris.iter_mut() {
        let s = det.is_sliver_points(t.p0, t.p1, t.p2);
        t.is_sliver = s;
        count += s as usize;
    }
    count
}

/// Appends a quad with counter-clockwise (CCW) front-face winding.
fn push_quad(out: &mut Vec<Tri>, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: [u8; 3]) {
    out.push(Tri::new(a, b, c, color));
    out.push(Tri::new(a, c, d, color));
}

/// Appends an axis-aligned box with outward-facing normals.
fn push_box(out: &mut Vec<Tri>, min: Vec3, max: Vec3, color: [u8; 3]) {
    let (x0, y0, z0) = (min.x, min.y, min.z);
    let (x1, y1, z1) = (max.x, max.y, max.z);

    // +Z (front)
    push_quad(
        out,
        Vec3::new(x0, y0, z1),
        Vec3::new(x1, y0, z1),
        Vec3::new(x1, y1, z1),
        Vec3::new(x0, y1, z1),
        color,
    );
    // -Z (back)
    push_quad(
        out,
        Vec3::new(x1, y0, z0),
        Vec3::new(x0, y0, z0),
        Vec3::new(x0, y1, z0),
        Vec3::new(x1, y1, z0),
        color,
    );
    // +X (right)
    push_quad(
        out,
        Vec3::new(x1, y0, z1),
        Vec3::new(x1, y0, z0),
        Vec3::new(x1, y1, z0),
        Vec3::new(x1, y1, z1),
        color,
    );
    // -X (left)
    push_quad(
        out,
        Vec3::new(x0, y0, z0),
        Vec3::new(x0, y0, z1),
        Vec3::new(x0, y1, z1),
        Vec3::new(x0, y1, z0),
        color,
    );
    // +Y (top)
    push_quad(
        out,
        Vec3::new(x0, y1, z0),
        Vec3::new(x0, y1, z1),
        Vec3::new(x1, y1, z1),
        Vec3::new(x1, y1, z0),
        color,
    );
    // -Y (bottom)
    push_quad(
        out,
        Vec3::new(x0, y0, z0),
        Vec3::new(x1, y0, z0),
        Vec3::new(x1, y0, z1),
        Vec3::new(x0, y0, z1),
        color,
    );
}

/// Appends a straight staircase rising toward +X: `steps` treads, each `tread` deep and `rise` higher
/// than the previous one. Each step is a solid box from the floor up, so there are no gaps under treads.
fn push_stairs(
    out: &mut Vec<Tri>,
    origin: Vec3,
    width_z: f32,
    steps: usize,
    tread: f32,
    rise: f32,
    color: [u8; 3],
) {
    for i in 0..steps {
        let x0 = origin.x + i as f32 * tread;
        push_box(
            out,
            Vec3::new(x0, origin.y, origin.z),
            Vec3::new(
                x0 + tread,
                origin.y + rise * (i + 1) as f32,
                origin.z + width_z,
            ),
            color,
        );
    }
}

/// Appends an enclosed room with inward-facing normals.
fn push_room(
    out: &mut Vec<Tri>,
    min: Vec3,
    max: Vec3,
    floor: [u8; 3],
    wall: [u8; 3],
    ceil: [u8; 3],
) {
    let (x0, y0, z0) = (min.x, min.y, min.z);
    let (x1, y1, z1) = (max.x, max.y, max.z);

    // Subdivided checkerboard floor (+Y normal) to keep floor triangles well-conditioned
    let nx = 6;
    let nz = 8;
    let dx = (x1 - x0) / nx as f32;
    let dz = (z1 - z0) / nz as f32;
    for ix in 0..nx {
        for iz in 0..nz {
            let ax = x0 + ix as f32 * dx;
            let bx = ax + dx;
            let az = z0 + iz as f32 * dz;
            let bz = az + dz;
            let col = if (ix + iz) % 2 == 0 {
                floor
            } else {
                [
                    floor[0].saturating_sub(14),
                    floor[1].saturating_sub(14),
                    floor[2].saturating_sub(14),
                ]
            };
            push_quad(
                out,
                Vec3::new(ax, y0, az),
                Vec3::new(ax, y0, bz),
                Vec3::new(bx, y0, bz),
                Vec3::new(bx, y0, az),
                col,
            );
        }
    }

    // Ceiling (-Y normal)
    push_quad(
        out,
        Vec3::new(x0, y1, z0),
        Vec3::new(x1, y1, z0),
        Vec3::new(x1, y1, z1),
        Vec3::new(x0, y1, z1),
        ceil,
    );
    // North wall (-Z, normal +Z)
    push_quad(
        out,
        Vec3::new(x0, y0, z0),
        Vec3::new(x1, y0, z0),
        Vec3::new(x1, y1, z0),
        Vec3::new(x0, y1, z0),
        wall,
    );
    // South wall (+Z, normal -Z)
    push_quad(
        out,
        Vec3::new(x1, y0, z1),
        Vec3::new(x0, y0, z1),
        Vec3::new(x0, y1, z1),
        Vec3::new(x1, y1, z1),
        wall,
    );
    // West wall (-X, normal +X)
    push_quad(
        out,
        Vec3::new(x0, y0, z1),
        Vec3::new(x0, y0, z0),
        Vec3::new(x0, y1, z0),
        Vec3::new(x0, y1, z1),
        wall,
    );
    // East wall (+X, normal -X)
    push_quad(
        out,
        Vec3::new(x1, y0, z0),
        Vec3::new(x1, y0, z1),
        Vec3::new(x1, y1, z1),
        Vec3::new(x1, y1, z0),
        wall,
    );
}

/// Builds a retro 3D test room with pillars, steps, and intentional sliver trims for F1/F2 testing.
pub fn build_demo_room() -> Vec<Tri> {
    let mut tris = Vec::new();

    // Main hall: 12 x 5 x 16 units
    push_room(
        &mut tris,
        Vec3::new(-6.0, 0.0, -10.0),
        Vec3::new(6.0, 5.0, 6.0),
        [92, 102, 114],  // stone floor
        [124, 134, 146], // concrete walls
        [68, 76, 86],    // dark ceiling
    );

    // 4 symmetric pillars
    let pillar_col = [156, 148, 132];
    for &px in &[-3.6, 3.6] {
        for &pz in &[-6.5, -1.5] {
            push_box(
                &mut tris,
                Vec3::new(px - 0.45, 0.0, pz - 0.45),
                Vec3::new(px + 0.45, 5.0, pz + 0.45),
                pillar_col,
            );
        }
    }

    // Back platform and step
    push_box(
        &mut tris,
        Vec3::new(-4.0, 0.0, -9.8),
        Vec3::new(4.0, 0.7, -7.2),
        [116, 128, 142],
    );
    push_box(
        &mut tris,
        Vec3::new(-2.2, 0.0, -7.2),
        Vec3::new(2.2, 0.35, -6.3),
        [130, 140, 152],
    );

    // Central pedestal
    push_box(
        &mut tris,
        Vec3::new(-0.9, 0.0, -4.9),
        Vec3::new(0.9, 1.0, -3.1),
        [138, 148, 160],
    );
    // Test staircase on the east side: 5 steps, 0.40 m deep and 0.20 m high each (top at 1.0 m).
    // It sits at x 3.4..5.4, z 1.2..2.8: clear of the pillars, the platform and every coordinate used
    // by the existing collision, controller and raycast tests.
    push_stairs(
        &mut tris,
        Vec3::new(3.4, 0.0, 1.2),
        1.6,
        5,
        0.4,
        0.2,
        [140, 150, 162],
    );

    // --- Intentional slivers to verify sliver_math live detection (F1 / F2) ---
    // 1) Long thin trim on the north wall: 11.8 x 0.32 -> min angle ~1.55 deg
    push_quad(
        &mut tris,
        Vec3::new(-5.9, 3.8, -9.95),
        Vec3::new(5.9, 3.8, -9.95),
        Vec3::new(5.9, 4.12, -9.95),
        Vec3::new(-5.9, 4.12, -9.95),
        [170, 155, 110],
    );
    // 2) Ultra-thin baseboard on the west wall: 15.6 x 0.14 -> min angle ~0.51 deg
    push_quad(
        &mut tris,
        Vec3::new(-5.95, 0.05, 5.8),
        Vec3::new(-5.95, 0.05, -9.8),
        Vec3::new(-5.95, 0.19, -9.8),
        Vec3::new(-5.95, 0.19, 5.8),
        [150, 140, 120],
    );
    // 3) Medium strip on the east wall: 10.0 x 1.2 -> min angle ~6.84 deg
    push_quad(
        &mut tris,
        Vec3::new(5.95, 1.6, -8.5),
        Vec3::new(5.95, 1.6, 1.5),
        Vec3::new(5.95, 2.8, 1.5),
        Vec3::new(5.95, 2.8, -8.5),
        [118, 144, 162],
    );

    tris
}

/// Spinning octahedron above the central pedestal.
pub fn build_spinner(angle: f32, center: Vec3) -> [Tri; 8] {
    let top = center + Vec3::new(0.0, 1.15, 0.0);
    let bot = center + Vec3::new(0.0, -1.15, 0.0);
    let r = 0.65;
    let ring = [
        center + Vec3::new(r, 0.0, 0.0).rotate_y(angle),
        center + Vec3::new(0.0, 0.0, -r).rotate_y(angle),
        center + Vec3::new(-r, 0.0, 0.0).rotate_y(angle),
        center + Vec3::new(0.0, 0.0, r).rotate_y(angle),
    ];
    let c1 = [86, 172, 224];
    let c2 = [62, 134, 189];
    [
        Tri::new(ring[0], ring[1], top, c1),
        Tri::new(ring[1], ring[2], top, c2),
        Tri::new(ring[2], ring[3], top, c1),
        Tri::new(ring[3], ring[0], top, c2),
        Tri::new(ring[1], ring[0], bot, c2),
        Tri::new(ring[2], ring[1], bot, c1),
        Tri::new(ring[3], ring[2], bot, c2),
        Tri::new(ring[0], ring[3], bot, c1),
    ]
}

/// Minimal Wavefront .obj loader (parses v and f with fan triangulation) with auto-centering and scaling.
pub fn load_obj(path: &Path) -> Result<Vec<Tri>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("Failed to read {:?}: {e}", path))?;
    let mut verts = Vec::new();
    let mut raw_faces: Vec<(usize, usize, usize)> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("v ") {
            let mut nums = rest
                .split_whitespace()
                .filter_map(|s| s.parse::<f32>().ok());
            if let (Some(x), Some(y), Some(z)) = (nums.next(), nums.next(), nums.next()) {
                verts.push(Vec3::new(x, y, z));
            }
        } else if let Some(rest) = line.strip_prefix("f ") {
            let mut idx = Vec::new();
            for tok in rest.split_whitespace() {
                if let Some(first) = tok.split('/').next() {
                    if let Ok(raw) = first.parse::<isize>() {
                        let resolved = if raw > 0 {
                            (raw - 1) as usize
                        } else {
                            verts.len().saturating_sub((-raw) as usize)
                        };
                        if resolved < verts.len() {
                            idx.push(resolved);
                        }
                    }
                }
            }
            for k in 1..idx.len().saturating_sub(1) {
                raw_faces.push((idx[0], idx[k], idx[k + 1]));
            }
        }
    }

    if verts.is_empty() || raw_faces.is_empty() {
        return Err("The .obj file contains no valid vertices or faces.".into());
    }

    // Center at the origin and normalize maximum extent to ~6 world units
    let mut vmin = verts[0];
    let mut vmax = verts[0];
    for &v in &verts {
        vmin.x = vmin.x.min(v.x);
        vmin.y = vmin.y.min(v.y);
        vmin.z = vmin.z.min(v.z);
        vmax.x = vmax.x.max(v.x);
        vmax.y = vmax.y.max(v.y);
        vmax.z = vmax.z.max(v.z);
    }
    let center = (vmin + vmax) * 0.5;
    let extent = (vmax.x - vmin.x)
        .max(vmax.y - vmin.y)
        .max(vmax.z - vmin.z)
        .max(1e-5);
    let scale = 6.0 / extent;
    for v in &mut verts {
        *v = (*v - center) * scale;
    }

    let mut tris = Vec::with_capacity(raw_faces.len());
    for (i0, i1, i2) in raw_faces {
        tris.push(Tri::new(verts[i0], verts[i1], verts[i2], [168, 178, 190]));
    }
    Ok(tris)
}

/// Large 80x80 m test arena for high-speed sprinting, long stairs, jump gaps, and wall impacts.
pub fn build_playground() -> Vec<Tri> {
    let mut tris = Vec::new();

    // Tiled floor (80x80 m in 10 m tiles) for clear visual speed reference.
    for ix in -4..4 {
        for iz in -4..4 {
            let x0 = ix as f32 * 10.0;
            let z0 = iz as f32 * 10.0;
            let x1 = x0 + 10.0;
            let z1 = z0 + 10.0;
            let color = if (ix + iz) & 1 == 0 {
                [76, 82, 92]
            } else {
                [66, 72, 82]
            };
            let a = Vec3::new(x0, 0.0, z0);
            let b = Vec3::new(x0, 0.0, z1);
            let c = Vec3::new(x1, 0.0, z1);
            let d = Vec3::new(x1, 0.0, z0);
            tris.push(Tri::new(a, b, c, color));
            tris.push(Tri::new(a, c, d, color));
        }
    }

    // Perimeter walls (height 6 m).
    let wall_col = [95, 102, 114];
    push_box(
        &mut tris,
        Vec3::new(-40.5, 0.0, -40.5),
        Vec3::new(40.5, 6.0, -40.0),
        wall_col,
    );
    push_box(
        &mut tris,
        Vec3::new(-40.5, 0.0, 40.0),
        Vec3::new(40.5, 6.0, 40.5),
        wall_col,
    );
    push_box(
        &mut tris,
        Vec3::new(-40.5, 0.0, -40.0),
        Vec3::new(-40.0, 6.0, 40.0),
        wall_col,
    );
    push_box(
        &mut tris,
        Vec3::new(40.0, 0.0, -40.0),
        Vec3::new(40.5, 6.0, 40.0),
        wall_col,
    );

    // Sprint runway markers along -Z (pillars every 10 m on both sides of the central lane).
    for i in -3..=3 {
        let z = i as f32 * 10.0;
        push_box(
            &mut tris,
            Vec3::new(-6.0, 0.0, z - 0.3),
            Vec3::new(-5.4, 3.0, z + 0.3),
            [130, 140, 155],
        );
        push_box(
            &mut tris,
            Vec3::new(5.4, 0.0, z - 0.3),
            Vec3::new(6.0, 3.0, z + 0.3),
            [130, 140, 155],
        );
    }

    // Long staircase (+X): 10 steps (0.45 m deep x 0.20 m high -> top at 2.0 m) + upper deck.
    push_stairs(
        &mut tris,
        Vec3::new(8.0, 0.0, -4.0),
        4.0,
        10,
        0.45,
        0.20,
        [140, 150, 162],
    );
    push_box(
        &mut tris,
        Vec3::new(12.5, 0.0, -4.0),
        Vec3::new(22.0, 2.0, 0.0),
        [122, 132, 146],
    );

    // Jump gap course off the upper deck (+Z): gaps of 2.0 m, 3.2 m, and 4.5 m.
    push_box(
        &mut tris,
        Vec3::new(14.0, 0.0, 2.0),
        Vec3::new(20.0, 2.0, 6.0),
        [122, 132, 146],
    );
    push_box(
        &mut tris,
        Vec3::new(14.0, 0.0, 9.2),
        Vec3::new(20.0, 2.0, 13.2),
        [122, 132, 146],
    );
    push_box(
        &mut tris,
        Vec3::new(14.0, 0.0, 17.7),
        Vec3::new(20.0, 2.0, 22.0),
        [122, 132, 146],
    );

    // Obstacle test line on the west side (-X):
    // 1) Climbable curb (0.20 m)
    push_box(
        &mut tris,
        Vec3::new(-14.0, 0.0, -10.0),
        Vec3::new(-10.0, 0.20, -6.0),
        [110, 145, 120],
    );
    // 2) Blocking ledge (0.55 m — above max_step_height, should stop without climbing)
    push_box(
        &mut tris,
        Vec3::new(-14.0, 0.0, -2.0),
        Vec3::new(-10.0, 0.55, 2.0),
        [165, 130, 95],
    );
    // 3) Tall impact wall (3.0 m — full sprint bounce test)
    push_box(
        &mut tris,
        Vec3::new(-14.0, 0.0, 6.0),
        Vec3::new(-10.0, 3.0, 12.0),
        [160, 100, 100],
    );

    // Low ceiling over a 0.20 m step (clearance 0.60 m: body of diameter 0.70 m cannot climb).
    push_box(
        &mut tris,
        Vec3::new(-22.0, 0.0, -4.0),
        Vec3::new(-18.0, 0.20, 2.0),
        [135, 135, 150],
    );
    push_box(
        &mut tris,
        Vec3::new(-22.0, 0.80, -4.0),
        Vec3::new(-18.0, 1.40, 2.0),
        [115, 115, 130],
    );

    tris
}
