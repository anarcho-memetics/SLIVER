//! Minimal Doom IWAD/PWAD map and texture loader (zero dependencies, pure safe Rust).
//!
//! Parses classic Doom map lumps (`THINGS`, `LINEDEFS`, `SIDEDEFS`, `VERTEXES`, `SEGS`,
//! `SSECTORS`, `NODES`, `SECTORS`) plus `PLAYPAL`, flats (`F_START..F_END`), `PNAMES`,
//! and `TEXTURE1`/`TEXTURE2`, converting them into `Vec<Tri>`, `Vec<Texture>`, and Player 1 spawn.

use crate::scene::{NO_TEX, SKY_TEX, Texture, Tri};
use sliver_math::Vec3;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Scale from Doom map units to Sliver meters (~36 units = 1 m).
const DOOM_SCALE: f32 = 1.0 / 36.0;
/// Classic Doom stretched 320x200 to 4:3 CRT, making pixels 1.2x taller than wide.
const DOOM_Y_SCALE: f32 = DOOM_SCALE * 1.2;
const NO_SIDEDEF: u16 = 0xFFFF;
const SUBSECTOR_BIT: u16 = 0x8000;
const ML_DONTPEGTOP: u16 = 0x0008;
const ML_DONTPEGBOTTOM: u16 = 0x0010;
const SKY_FALLBACK_COLOR: [u8; 3] = [104, 140, 184];

#[derive(Debug)]
pub struct WadMap {
    pub name: String,
    pub tris: Vec<Tri>,
    pub textures: Vec<Texture>,
    /// Index into `textures` for the map's panoramic sky (`SKY1`..`SKY4`), if present.
    pub sky_tex_id: Option<u16>,
    /// Floor position (x, floor_y, z) of Player 1 in Sliver coordinates.
    pub spawn_pos: Vec3,
    /// Initial camera yaw (radians) matching Sliver's `fwd = (sin(yaw), 0, -cos(yaw))`.
    pub spawn_yaw: f32,
}

#[derive(Clone, Copy)]
struct Lump<'a> {
    name: [u8; 8],
    data: &'a [u8],
}

#[derive(Clone, Copy)]
struct Thing {
    x: i16,
    y: i16,
    angle: i16,
    kind: i16,
}

#[derive(Clone, Copy)]
struct Linedef {
    v1: u16,
    v2: u16,
    flags: u16,
    special: u16,
    front: u16,
    back: u16,
}

#[derive(Clone, Copy)]
struct Sidedef {
    x_off: i16,
    y_off: i16,
    upper_tex: [u8; 8],
    lower_tex: [u8; 8],
    mid_tex: [u8; 8],
    sector: u16,
}

#[derive(Clone, Copy)]
struct Seg {
    linedef: u16,
    direction: i16,
}

#[derive(Clone, Copy)]
struct Subsector {
    num_segs: u16,
    first_seg: u16,
}

#[derive(Clone, Copy)]
struct Node {
    x: i16,
    y: i16,
    dx: i16,
    dy: i16,
    right_child: u16,
    left_child: u16,
}

#[derive(Clone, Copy)]
struct Sector {
    floor_h: i16,
    ceil_h: i16,
    floor_tex: [u8; 8],
    ceil_tex: [u8; 8],
    light: i16,
}

#[derive(Clone)]
struct WallPatchRef {
    origin_x: i16,
    origin_y: i16,
    patch_idx: u16,
}

#[derive(Clone)]
struct WallTexDef {
    width: u16,
    height: u16,
    patches: Vec<WallPatchRef>,
}

struct TexLoader<'a> {
    palette: Option<[u32; 256]>,
    flat_lumps: HashMap<[u8; 8], &'a [u8]>,
    patch_lumps: HashMap<[u8; 8], &'a [u8]>,
    pnames: Vec<[u8; 8]>,
    wall_defs: HashMap<[u8; 8], WallTexDef>,
    flat_cache: HashMap<[u8; 8], u16>,
    wall_cache: HashMap<[u8; 8], (u16, u16)>,
    textures: Vec<Texture>,
}

impl<'a> TexLoader<'a> {
    fn from_lumps(lumps: &[Lump<'a>]) -> Self {
        let mut palette = None;
        let mut flat_lumps = HashMap::new();
        let mut patch_lumps = HashMap::new();
        let mut pnames_data: Option<&'a [u8]> = None;
        let mut tex1_data: Option<&'a [u8]> = None;
        let mut tex2_data: Option<&'a [u8]> = None;

        let mut in_flats = false;
        for lump in lumps {
            let n = name_str(&lump.name);
            if n == "F_START" || n == "FF_START" {
                in_flats = true;
                continue;
            }
            if n == "F_END" || n == "FF_END" {
                in_flats = false;
                continue;
            }
            if in_flats {
                if lump.data.len() == 4096 {
                    flat_lumps.insert(lump.name, lump.data);
                }
                continue;
            }
            if n == "PLAYPAL" && lump.data.len() >= 768 {
                let mut pal = [0u32; 256];
                for (i, slot) in pal.iter_mut().enumerate() {
                    let r = lump.data[i * 3] as u32;
                    let g = lump.data[i * 3 + 1] as u32;
                    let b = lump.data[i * 3 + 2] as u32;
                    *slot = (r << 16) | (g << 8) | b;
                }
                palette = Some(pal);
            } else if n == "PNAMES" {
                pnames_data = Some(lump.data);
            } else if n == "TEXTURE1" {
                tex1_data = Some(lump.data);
            } else if n == "TEXTURE2" {
                tex2_data = Some(lump.data);
            }
            if !lump.data.is_empty() {
                patch_lumps.insert(lump.name, lump.data);
            }
        }

        let mut pnames = Vec::new();
        if let Some(b) = pnames_data
            && b.len() >= 4
        {
            let count = read_i32(b, 0).max(0) as usize;
            let max_entries = (b.len() - 4) / 8;
            for i in 0..count.min(max_entries) {
                pnames.push(read_name8(b, 4 + i * 8));
            }
        }

        let mut wall_defs = HashMap::new();
        if let Some(b) = tex1_data {
            parse_texture_lump(b, &mut wall_defs);
        }
        if let Some(b) = tex2_data {
            parse_texture_lump(b, &mut wall_defs);
        }

        Self {
            palette,
            flat_lumps,
            patch_lumps,
            pnames,
            wall_defs,
            flat_cache: HashMap::new(),
            wall_cache: HashMap::new(),
            textures: Vec::new(),
        }
    }

    fn get_flat(&mut self, name: &[u8; 8]) -> u16 {
        if let Some(&id) = self.flat_cache.get(name) {
            return id;
        }
        let (Some(pal), Some(&raw)) = (self.palette, self.flat_lumps.get(name)) else {
            return NO_TEX;
        };
        if self.textures.len() >= SKY_TEX as usize {
            return NO_TEX;
        }
        let pixels: Vec<u32> = raw.iter().map(|&idx| pal[idx as usize]).collect();
        let id = self.textures.len() as u16;
        self.textures.push(Texture::new(64, 64, pixels));
        self.flat_cache.insert(*name, id);
        id
    }

    /// Returns `(tex_id, tex_height)` for a wall texture, composing its patches on demand.
    fn get_wall(&mut self, name: &[u8; 8]) -> (u16, f32) {
        if name_str(name) == "-" || name_str(name).is_empty() {
            return (NO_TEX, 128.0);
        }
        if let Some(&(id, h)) = self.wall_cache.get(name) {
            return (id, h as f32);
        }
        let (Some(pal), Some(def)) = (self.palette, self.wall_defs.get(name).cloned()) else {
            return (NO_TEX, 128.0);
        };
        if def.width == 0 || def.height == 0 || self.textures.len() >= SKY_TEX as usize {
            return (NO_TEX, 128.0);
        }

        let w = def.width as usize;
        let h = def.height as usize;
        let mut pixels = vec![0x0018_1818u32; w * h];

        for p in &def.patches {
            let Some(pname) = self.pnames.get(p.patch_idx as usize) else {
                continue;
            };
            let Some(&pdata) = self.patch_lumps.get(pname) else {
                continue;
            };
            blit_patch(
                pdata,
                &pal,
                p.origin_x as i32,
                p.origin_y as i32,
                w,
                h,
                &mut pixels,
            );
        }

        let id = self.textures.len() as u16;
        self.textures.push(Texture::new(w, h, pixels));
        self.wall_cache.insert(*name, (id, def.height));
        (id, def.height as f32)
    }
}

fn parse_texture_lump(b: &[u8], out: &mut HashMap<[u8; 8], WallTexDef>) {
    if b.len() < 4 {
        return;
    }
    let count = read_i32(b, 0).max(0) as usize;
    let max_offsets = (b.len() - 4) / 4;
    for i in 0..count.min(max_offsets) {
        let off_raw = read_i32(b, 4 + i * 4);
        if off_raw < 0 {
            continue;
        }
        let off = off_raw as usize;
        if off + 22 > b.len() {
            continue;
        }
        let name = read_name8(b, off);
        let width = read_u16(b, off + 12);
        let height = read_u16(b, off + 14);
        let patch_count = read_u16(b, off + 20) as usize;
        let mut patches = Vec::with_capacity(patch_count);
        for k in 0..patch_count {
            let po = off + 22 + k * 10;
            if po + 10 > b.len() {
                break;
            }
            patches.push(WallPatchRef {
                origin_x: read_i16(b, po),
                origin_y: read_i16(b, po + 2),
                patch_idx: read_u16(b, po + 4),
            });
        }
        out.insert(
            name,
            WallTexDef {
                width,
                height,
                patches,
            },
        );
    }
}

fn blit_patch(
    patch: &[u8],
    pal: &[u32; 256],
    origin_x: i32,
    origin_y: i32,
    dst_w: usize,
    dst_h: usize,
    dst: &mut [u32],
) {
    if patch.len() < 8 {
        return;
    }
    let pw = read_u16(patch, 0) as usize;
    if 8 + pw * 4 > patch.len() {
        return;
    }
    let dst_wi = dst_w as i32;
    let dst_hi = dst_h as i32;

    for col in 0..pw {
        let dst_x = origin_x + col as i32;
        if dst_x < 0 || dst_x >= dst_wi {
            continue;
        }
        let mut pos = read_u32(patch, 8 + col * 4) as usize;
        while pos < patch.len() {
            let topdelta = patch[pos];
            if topdelta == 0xFF {
                break;
            }
            if pos + 2 >= patch.len() {
                break;
            }
            let len = patch[pos + 1] as usize;
            let data_start = pos + 3;
            let data_end = data_start.saturating_add(len);
            if data_end > patch.len() {
                break;
            }
            for (row, &pal_idx) in patch[data_start..data_end].iter().enumerate() {
                let dst_y = origin_y + topdelta as i32 + row as i32;
                if dst_y >= 0 && dst_y < dst_hi {
                    dst[(dst_y as usize) * dst_w + (dst_x as usize)] = pal[pal_idx as usize];
                }
            }
            pos = data_end.saturating_add(1);
        }
    }
}

fn read_i16(b: &[u8], off: usize) -> i16 {
    i16::from_le_bytes([b[off], b[off + 1]])
}

fn read_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn read_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn read_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn read_name8(b: &[u8], off: usize) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (dst, &src) in out.iter_mut().zip(&b[off..off + 8]) {
        *dst = src.to_ascii_uppercase();
    }
    out
}

fn name8(s: &str) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (dst, &src) in out.iter_mut().zip(s.as_bytes()) {
        *dst = src.to_ascii_uppercase();
    }
    out
}

fn name_str(name: &[u8; 8]) -> &str {
    let len = name.iter().position(|&c| c == 0).unwrap_or(8);
    std::str::from_utf8(&name[..len]).unwrap_or("")
}

fn is_sky(name: &[u8; 8]) -> bool {
    name_str(name) == "F_SKY1"
}

fn sky_texture_name(map_name: &str) -> [u8; 8] {
    let up = map_name.to_ascii_uppercase();
    if let Some(rest) = up.strip_prefix("MAP") {
        if let Ok(num) = rest.parse::<u32>() {
            return match num {
                1..=11 => name8("SKY1"),
                12..=20 => name8("SKY2"),
                _ => name8("SKY3"),
            };
        }
    } else if up.starts_with("E2") {
        return name8("SKY2");
    } else if up.starts_with("E3") {
        return name8("SKY3");
    } else if up.starts_with("E4") {
        return name8("SKY4");
    }
    name8("SKY1")
}

fn is_manual_door_special(special: u16) -> bool {
    matches!(special, 1 | 26 | 27 | 28 | 31 | 32 | 33 | 34 | 117 | 118)
}

#[inline]
fn doom_to_sliver(dx: f32, dy: f32, dz: f32) -> Vec3 {
    Vec3::new(dx * DOOM_SCALE, dz * DOOM_Y_SCALE, -dy * DOOM_SCALE)
}

fn parse_directory(bytes: &[u8]) -> Result<Vec<Lump<'_>>, String> {
    if bytes.len() < 12 {
        return Err("WAD file is too short to contain a header.".into());
    }
    let magic = &bytes[0..4];
    if magic != b"IWAD" && magic != b"PWAD" {
        return Err("Invalid WAD header (expected IWAD or PWAD).".into());
    }
    let num_lumps = read_i32(bytes, 4);
    let dir_off = read_i32(bytes, 8);
    if num_lumps < 0 || dir_off < 0 {
        return Err("Negative lump count or directory offset in WAD header.".into());
    }
    let num_lumps = num_lumps as usize;
    let dir_off = dir_off as usize;
    let dir_end = dir_off
        .checked_add(num_lumps.checked_mul(16).ok_or("WAD directory overflow.")?)
        .ok_or("WAD directory overflow.")?;
    if dir_end > bytes.len() {
        return Err("WAD directory extends past end of file.".into());
    }

    let mut lumps = Vec::with_capacity(num_lumps);
    for i in 0..num_lumps {
        let off = dir_off + i * 16;
        let filepos = read_i32(bytes, off);
        let size = read_i32(bytes, off + 4);
        let name = read_name8(bytes, off + 8);
        if filepos < 0 || size < 0 {
            return Err(format!("Invalid lump bounds for '{}'.", name_str(&name)));
        }
        let start = filepos as usize;
        let end = start
            .checked_add(size as usize)
            .ok_or("WAD lump size overflow.")?;
        if end > bytes.len() {
            return Err(format!(
                "Lump '{}' extends past end of file.",
                name_str(&name)
            ));
        }
        lumps.push(Lump {
            name,
            data: &bytes[start..end],
        });
    }
    Ok(lumps)
}

fn find_map_index(lumps: &[Lump<'_>], requested: Option<&str>) -> Result<usize, String> {
    let target = requested.map(|s| s.to_ascii_uppercase());
    for i in 0..lumps.len().saturating_sub(1) {
        if name_str(&lumps[i + 1].name) != "THINGS" {
            continue;
        }
        let map_name = name_str(&lumps[i].name);
        match &target {
            Some(want) if want == map_name => return Ok(i),
            None => return Ok(i),
            _ => {}
        }
    }
    match target {
        Some(want) => Err(format!("Map '{want}' not found in WAD.")),
        None => Err("No Doom maps found in WAD.".into()),
    }
}

fn find_map_lump<'a>(lumps: &[Lump<'a>], map_idx: usize, want: &str) -> Result<&'a [u8], String> {
    let limit = (map_idx + 12).min(lumps.len());
    for lump in &lumps[map_idx + 1..limit] {
        if name_str(&lump.name) == want {
            return Ok(lump.data);
        }
    }
    Err(format!("Missing required lump '{want}' in map."))
}

/// Loads a map from a Doom `.wad` file (`map_name` defaults to the first map, e.g. `E1M1`).
pub fn load_wad_map(path: &Path, map_name: Option<&str>) -> Result<WadMap, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {:?}: {e}", path))?;
    load_wad_map_bytes(&bytes, map_name)
}

/// Parses a Doom map directly from WAD bytes (used by both the file loader and unit tests).
pub fn load_wad_map_bytes(bytes: &[u8], map_name: Option<&str>) -> Result<WadMap, String> {
    let lumps = parse_directory(bytes)?;
    let map_idx = find_map_index(&lumps, map_name)?;
    let resolved_name = name_str(&lumps[map_idx].name).to_string();

    let mut tex_loader = TexLoader::from_lumps(&lumps);
    let sky_name = sky_texture_name(&resolved_name);
    let (mut sky_id, _) = tex_loader.get_wall(&sky_name);
    if sky_id == NO_TEX && sky_name != name8("SKY1") {
        sky_id = tex_loader.get_wall(&name8("SKY1")).0;
    }
    let sky_tex_id = (sky_id != NO_TEX).then_some(sky_id);

    let things_b = find_map_lump(&lumps, map_idx, "THINGS")?;
    let linedefs_b = find_map_lump(&lumps, map_idx, "LINEDEFS")?;
    let sidedefs_b = find_map_lump(&lumps, map_idx, "SIDEDEFS")?;
    let vertexes_b = find_map_lump(&lumps, map_idx, "VERTEXES")?;
    let segs_b = find_map_lump(&lumps, map_idx, "SEGS")?;
    let ssectors_b = find_map_lump(&lumps, map_idx, "SSECTORS")?;
    let nodes_b = find_map_lump(&lumps, map_idx, "NODES")?;
    let sectors_b = find_map_lump(&lumps, map_idx, "SECTORS")?;

    let things: Vec<Thing> = things_b
        .as_chunks::<10>()
        .0
        .iter()
        .map(|c| Thing {
            x: read_i16(c, 0),
            y: read_i16(c, 2),
            angle: read_i16(c, 4),
            kind: read_i16(c, 6),
        })
        .collect();

    let verts: Vec<(f32, f32)> = vertexes_b
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| (read_i16(c, 0) as f32, read_i16(c, 2) as f32))
        .collect();

    let linedefs: Vec<Linedef> = linedefs_b
        .as_chunks::<14>()
        .0
        .iter()
        .map(|c| Linedef {
            v1: read_u16(c, 0),
            v2: read_u16(c, 2),
            flags: read_u16(c, 4),
            special: read_u16(c, 6),
            front: read_u16(c, 10),
            back: read_u16(c, 12),
        })
        .collect();

    let sidedefs: Vec<Sidedef> = sidedefs_b
        .as_chunks::<30>()
        .0
        .iter()
        .map(|c| Sidedef {
            x_off: read_i16(c, 0),
            y_off: read_i16(c, 2),
            upper_tex: read_name8(c, 4),
            lower_tex: read_name8(c, 12),
            mid_tex: read_name8(c, 20),
            sector: read_u16(c, 28),
        })
        .collect();

    let segs: Vec<Seg> = segs_b
        .as_chunks::<12>()
        .0
        .iter()
        .map(|c| Seg {
            linedef: read_u16(c, 6),
            direction: read_i16(c, 8),
        })
        .collect();

    let ssectors: Vec<Subsector> = ssectors_b
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| Subsector {
            num_segs: read_u16(c, 0),
            first_seg: read_u16(c, 2),
        })
        .collect();

    let nodes: Vec<Node> = nodes_b
        .as_chunks::<28>()
        .0
        .iter()
        .map(|c| Node {
            x: read_i16(c, 0),
            y: read_i16(c, 2),
            dx: read_i16(c, 4),
            dy: read_i16(c, 6),
            right_child: read_u16(c, 24),
            left_child: read_u16(c, 26),
        })
        .collect();

    let mut sectors: Vec<Sector> = sectors_b
        .as_chunks::<26>()
        .0
        .iter()
        .map(|c| Sector {
            floor_h: read_i16(c, 0),
            ceil_h: read_i16(c, 2),
            floor_tex: read_name8(c, 4),
            ceil_tex: read_name8(c, 12),
            light: read_i16(c, 20),
        })
        .collect();

    if verts.is_empty() || sectors.is_empty() {
        return Err(format!("Map '{resolved_name}' has no vertices or sectors."));
    }

    // Open zero-height manual doors for static walkthrough exploration.
    open_manual_doors(&linedefs, &sidedefs, &mut sectors);

    let mut tris = Vec::with_capacity(linedefs.len() * 4 + ssectors.len() * 6);

    // 1) Walls from linedefs.
    for ld in &linedefs {
        let (Some(&p1), Some(&p2)) = (verts.get(ld.v1 as usize), verts.get(ld.v2 as usize)) else {
            continue;
        };
        if ld.front == NO_SIDEDEF {
            continue;
        }
        let Some(f_sd) = sidedefs.get(ld.front as usize) else {
            continue;
        };
        let Some(&f_sec) = sectors.get(f_sd.sector as usize) else {
            continue;
        };

        let dont_peg_top = (ld.flags & ML_DONTPEGTOP) != 0;
        let dont_peg_bot = (ld.flags & ML_DONTPEGBOTTOM) != 0;
        let wall_len = (p2.0 - p1.0).hypot(p2.1 - p1.1);

        if ld.back == NO_SIDEDEF {
            // One-sided solid wall facing the front (right) sector.
            if f_sec.ceil_h > f_sec.floor_h {
                let (tex_id, tex_h) = tex_loader.get_wall(&f_sd.mid_tex);
                let y_bot = f_sec.floor_h as f32;
                let y_top = f_sec.ceil_h as f32;
                let (v_top, v_bot) = if dont_peg_bot {
                    let vb = tex_h + f_sd.y_off as f32;
                    (vb - (y_top - y_bot), vb)
                } else {
                    let vt = f_sd.y_off as f32;
                    (vt, vt + (y_top - y_bot))
                };
                let col = shade_surface(SurfaceKind::Wall, &f_sd.mid_tex, f_sec.light);
                push_wall(
                    &mut tris,
                    p1,
                    p2,
                    y_bot,
                    y_top,
                    f_sd.x_off as f32,
                    f_sd.x_off as f32 + wall_len,
                    v_top,
                    v_bot,
                    col,
                    tex_id,
                    f_sec.light.clamp(0, 255) as u8,
                );
            }
        } else if let Some(b_sd) = sidedefs.get(ld.back as usize) {
            let Some(&b_sec) = sectors.get(b_sd.sector as usize) else {
                continue;
            };
            let both_sky = is_sky(&f_sec.ceil_tex) && is_sky(&b_sec.ceil_tex);
            let f_light = f_sec.light.clamp(0, 255) as u8;
            let b_light = b_sec.light.clamp(0, 255) as u8;

            // Front upper trim (or sky occluder when both sectors have F_SKY1).
            if f_sec.ceil_h > b_sec.ceil_h {
                let y_bot = b_sec.ceil_h as f32;
                let y_top = f_sec.ceil_h as f32;
                if both_sky {
                    push_wall(
                        &mut tris,
                        p1,
                        p2,
                        y_bot,
                        y_top,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        SKY_FALLBACK_COLOR,
                        SKY_TEX,
                        255,
                    );
                } else {
                    let (tex_id, tex_h) = tex_loader.get_wall(&f_sd.upper_tex);
                    let (v_top, v_bot) = if dont_peg_top {
                        let vt = f_sd.y_off as f32;
                        (vt, vt + (y_top - y_bot))
                    } else {
                        let vb = tex_h + f_sd.y_off as f32;
                        (vb - (y_top - y_bot), vb)
                    };
                    let col = shade_surface(SurfaceKind::Trim, &f_sd.upper_tex, f_sec.light);
                    push_wall(
                        &mut tris,
                        p1,
                        p2,
                        y_bot,
                        y_top,
                        f_sd.x_off as f32,
                        f_sd.x_off as f32 + wall_len,
                        v_top,
                        v_bot,
                        col,
                        tex_id,
                        f_light,
                    );
                }
            }
            if b_sec.floor_h > f_sec.floor_h {
                let (tex_id, _) = tex_loader.get_wall(&f_sd.lower_tex);
                let y_bot = f_sec.floor_h as f32;
                let y_top = b_sec.floor_h as f32;
                let (v_top, v_bot) = if dont_peg_bot {
                    let anchor = f_sec.ceil_h as f32;
                    (
                        (anchor - y_top) + f_sd.y_off as f32,
                        (anchor - y_bot) + f_sd.y_off as f32,
                    )
                } else {
                    let vt = f_sd.y_off as f32;
                    (vt, vt + (y_top - y_bot))
                };
                let col = shade_surface(SurfaceKind::Trim, &f_sd.lower_tex, f_sec.light);
                push_wall(
                    &mut tris,
                    p1,
                    p2,
                    y_bot,
                    y_top,
                    f_sd.x_off as f32,
                    f_sd.x_off as f32 + wall_len,
                    v_top,
                    v_bot,
                    col,
                    tex_id,
                    f_light,
                );
            }

            // Back upper trim (or sky occluder when both sectors have F_SKY1).
            if b_sec.ceil_h > f_sec.ceil_h {
                let y_bot = f_sec.ceil_h as f32;
                let y_top = b_sec.ceil_h as f32;
                if both_sky {
                    push_wall(
                        &mut tris,
                        p2,
                        p1,
                        y_bot,
                        y_top,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        SKY_FALLBACK_COLOR,
                        SKY_TEX,
                        255,
                    );
                } else {
                    let (tex_id, tex_h) = tex_loader.get_wall(&b_sd.upper_tex);
                    let (v_top, v_bot) = if dont_peg_top {
                        let vt = b_sd.y_off as f32;
                        (vt, vt + (y_top - y_bot))
                    } else {
                        let vb = tex_h + b_sd.y_off as f32;
                        (vb - (y_top - y_bot), vb)
                    };
                    let col = shade_surface(SurfaceKind::Trim, &b_sd.upper_tex, b_sec.light);
                    push_wall(
                        &mut tris,
                        p2,
                        p1,
                        y_bot,
                        y_top,
                        b_sd.x_off as f32,
                        b_sd.x_off as f32 + wall_len,
                        v_top,
                        v_bot,
                        col,
                        tex_id,
                        b_light,
                    );
                }
            }
            if f_sec.floor_h > b_sec.floor_h {
                let (tex_id, _) = tex_loader.get_wall(&b_sd.lower_tex);
                let y_bot = b_sec.floor_h as f32;
                let y_top = f_sec.floor_h as f32;
                let (v_top, v_bot) = if dont_peg_bot {
                    let anchor = b_sec.ceil_h as f32;
                    (
                        (anchor - y_top) + b_sd.y_off as f32,
                        (anchor - y_bot) + b_sd.y_off as f32,
                    )
                } else {
                    let vt = b_sd.y_off as f32;
                    (vt, vt + (y_top - y_bot))
                };
                let col = shade_surface(SurfaceKind::Trim, &b_sd.lower_tex, b_sec.light);
                push_wall(
                    &mut tris,
                    p2,
                    p1,
                    y_bot,
                    y_top,
                    b_sd.x_off as f32,
                    b_sd.x_off as f32 + wall_len,
                    v_top,
                    v_bot,
                    col,
                    tex_id,
                    b_light,
                );
            }
        }
    }

    // 2) Floors and ceilings by clipping the map bbox through the BSP tree and each subsector's segs.
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(vx, vy) in &verts {
        min_x = min_x.min(vx as f64);
        max_x = max_x.max(vx as f64);
        min_y = min_y.min(vy as f64);
        max_y = max_y.max(vy as f64);
    }
    let pad = 64.0;
    let root_poly = vec![
        (min_x - pad, min_y - pad),
        (max_x + pad, min_y - pad),
        (max_x + pad, max_y + pad),
        (min_x - pad, max_y + pad),
    ];

    let mut ctx = BspBuildCtx {
        verts: &verts,
        linedefs: &linedefs,
        sidedefs: &sidedefs,
        segs: &segs,
        ssectors: &ssectors,
        nodes: &nodes,
        sectors: &sectors,
        tex_loader: &mut tex_loader,
        tris: &mut tris,
    };
    if let Some(root_idx) = nodes.len().checked_sub(1) {
        emit_bsp_flats(&mut ctx, root_idx as u16, root_poly);
    } else if !ssectors.is_empty() {
        emit_subsector_flats(&mut ctx, 0, root_poly);
    }

    // 3) Player 1 spawn (thing type 1) with exact floor height from the BSP tree.
    let (spawn_x, spawn_y, spawn_angle) = things
        .iter()
        .find(|t| t.kind == 1)
        .map(|t| (t.x as f32, t.y as f32, t.angle as f32))
        .unwrap_or((
            (min_x + max_x) as f32 * 0.5,
            (min_y + max_y) as f32 * 0.5,
            90.0,
        ));
    let floor_z = find_floor_height(
        spawn_x as f64,
        spawn_y as f64,
        &nodes,
        &ssectors,
        &segs,
        &linedefs,
        &sidedefs,
        &sectors,
    );

    Ok(WadMap {
        name: resolved_name,
        tris,
        textures: tex_loader.textures,
        sky_tex_id,
        spawn_pos: doom_to_sliver(spawn_x, spawn_y, floor_z),
        spawn_yaw: (90.0 - spawn_angle).to_radians(),
    })
}

fn open_manual_doors(linedefs: &[Linedef], sidedefs: &[Sidedef], sectors: &mut [Sector]) {
    let mut door_target: HashMap<usize, i16> = HashMap::new();
    for ld in linedefs {
        if ld.front == NO_SIDEDEF || ld.back == NO_SIDEDEF {
            continue;
        }
        let (Some(fsd), Some(bsd)) = (
            sidedefs.get(ld.front as usize),
            sidedefs.get(ld.back as usize),
        ) else {
            continue;
        };
        let (f_idx, b_idx) = (fsd.sector as usize, bsd.sector as usize);
        if f_idx >= sectors.len() || b_idx >= sectors.len() || f_idx == b_idx {
            continue;
        }
        if is_manual_door_special(ld.special) {
            if sectors[b_idx].ceil_h <= sectors[b_idx].floor_h {
                door_target.entry(b_idx).or_insert(i16::MAX);
            }
            if sectors[f_idx].ceil_h <= sectors[f_idx].floor_h {
                door_target.entry(f_idx).or_insert(i16::MAX);
            }
        }
    }
    if door_target.is_empty() {
        return;
    }
    // Doom's EV_VerticalDoor opens a door to (lowest neighboring ceiling - 4).
    for ld in linedefs {
        if ld.front == NO_SIDEDEF || ld.back == NO_SIDEDEF {
            continue;
        }
        let (Some(fsd), Some(bsd)) = (
            sidedefs.get(ld.front as usize),
            sidedefs.get(ld.back as usize),
        ) else {
            continue;
        };
        let (f_idx, b_idx) = (fsd.sector as usize, bsd.sector as usize);
        if f_idx >= sectors.len() || b_idx >= sectors.len() || f_idx == b_idx {
            continue;
        }
        if let Some(best) = door_target.get_mut(&f_idx) {
            *best = (*best).min(sectors[b_idx].ceil_h);
        }
        if let Some(best) = door_target.get_mut(&b_idx) {
            *best = (*best).min(sectors[f_idx].ceil_h);
        }
    }
    for (sec_idx, min_adj_ceil) in door_target {
        let open_h = min_adj_ceil.saturating_sub(4);
        if open_h > sectors[sec_idx].floor_h {
            sectors[sec_idx].ceil_h = open_h;
        }
    }
}

struct BspBuildCtx<'a, 'b> {
    verts: &'a [(f32, f32)],
    linedefs: &'a [Linedef],
    sidedefs: &'a [Sidedef],
    segs: &'a [Seg],
    ssectors: &'a [Subsector],
    nodes: &'a [Node],
    sectors: &'a [Sector],
    tex_loader: &'a mut TexLoader<'b>,
    tris: &'a mut Vec<Tri>,
}

fn emit_bsp_flats(ctx: &mut BspBuildCtx<'_, '_>, child: u16, poly: Vec<(f64, f64)>) {
    if poly.len() < 3 {
        return;
    }
    if (child & SUBSECTOR_BIT) != 0 {
        emit_subsector_flats(ctx, (child & !SUBSECTOR_BIT) as usize, poly);
        return;
    }
    let Some(&node) = ctx.nodes.get(child as usize) else {
        return;
    };
    let (ax, ay) = (node.x as f64, node.y as f64);
    let (dx, dy) = (node.dx as f64, node.dy as f64);

    let right_poly = clip_poly_halfplane(&poly, ax, ay, dx, dy, true);
    let left_poly = clip_poly_halfplane(&poly, ax, ay, dx, dy, false);

    emit_bsp_flats(ctx, node.right_child, right_poly);
    emit_bsp_flats(ctx, node.left_child, left_poly);
}

fn subsector_sector(
    ss: Subsector,
    segs: &[Seg],
    linedefs: &[Linedef],
    sidedefs: &[Sidedef],
) -> Option<usize> {
    let start = ss.first_seg as usize;
    let end = start.saturating_add(ss.num_segs as usize).min(segs.len());
    for seg in &segs[start..end] {
        let ld = linedefs.get(seg.linedef as usize)?;
        let sd_idx = if seg.direction == 0 {
            ld.front
        } else {
            ld.back
        };
        if sd_idx != NO_SIDEDEF
            && let Some(sd) = sidedefs.get(sd_idx as usize)
        {
            return Some(sd.sector as usize);
        }
    }
    None
}

fn emit_subsector_flats(ctx: &mut BspBuildCtx<'_, '_>, ss_idx: usize, mut poly: Vec<(f64, f64)>) {
    let Some(&ss) = ctx.ssectors.get(ss_idx) else {
        return;
    };
    let Some(sec_idx) = subsector_sector(ss, ctx.segs, ctx.linedefs, ctx.sidedefs) else {
        return;
    };
    let Some(&sec) = ctx.sectors.get(sec_idx) else {
        return;
    };

    let start = ss.first_seg as usize;
    let end = start
        .saturating_add(ss.num_segs as usize)
        .min(ctx.segs.len());
    let ss_segs = &ctx.segs[start..end];

    // Clip the node polygon only by one-sided walls (void boundaries), using the parent
    // linedef's unrounded vertices instead of DoomBSP's i16-rounded split seg vertices.
    // Internal boundaries between subsectors are already partitioned watertightly by `NODES`.
    for seg in ss_segs {
        let Some(ld) = ctx.linedefs.get(seg.linedef as usize) else {
            continue;
        };
        if ld.back != NO_SIDEDEF && (ld.back as usize) < ctx.sidedefs.len() {
            continue;
        }
        let (v_start, v_end) = if seg.direction == 0 {
            (ld.v1, ld.v2)
        } else {
            (ld.v2, ld.v1)
        };
        let (Some(&a), Some(&b)) = (
            ctx.verts.get(v_start as usize),
            ctx.verts.get(v_end as usize),
        ) else {
            continue;
        };
        poly = clip_poly_halfplane(
            &poly,
            a.0 as f64,
            a.1 as f64,
            (b.0 - a.0) as f64,
            (b.1 - a.1) as f64,
            true,
        );
        if poly.len() < 3 {
            return;
        }
    }

    // Ensure CCW order in Doom 2D (+X right, +Y north) so fan triangles produce +Y floor normals.
    let area2 = poly_signed_area2(&poly);
    if area2.abs() < 1.0 {
        return;
    }
    if area2 < 0.0 {
        poly.reverse();
    }

    let sec_light = sec.light.clamp(0, 255) as u8;
    let (floor_tex_id, floor_col, floor_light) = if is_sky(&sec.floor_tex) {
        (SKY_TEX, SKY_FALLBACK_COLOR, 255)
    } else {
        (
            ctx.tex_loader.get_flat(&sec.floor_tex),
            shade_surface(SurfaceKind::Floor, &sec.floor_tex, sec.light),
            sec_light,
        )
    };
    let fy = sec.floor_h as f32;
    let p0_f = doom_to_sliver(poly[0].0 as f32, poly[0].1 as f32, fy);
    let uv0 = [poly[0].0 as f32, -(poly[0].1 as f32)];
    for i in 1..poly.len() - 1 {
        let p1 = doom_to_sliver(poly[i].0 as f32, poly[i].1 as f32, fy);
        let p2 = doom_to_sliver(poly[i + 1].0 as f32, poly[i + 1].1 as f32, fy);
        let uv1 = [poly[i].0 as f32, -(poly[i].1 as f32)];
        let uv2 = [poly[i + 1].0 as f32, -(poly[i + 1].1 as f32)];
        push_if_non_degenerate(
            ctx.tris,
            p0_f,
            p1,
            p2,
            uv0,
            uv1,
            uv2,
            floor_col,
            floor_tex_id,
            floor_light,
        );
    }

    if sec.ceil_h > sec.floor_h {
        let (ceil_tex_id, ceil_col, ceil_light) = if is_sky(&sec.ceil_tex) {
            (SKY_TEX, SKY_FALLBACK_COLOR, 255)
        } else {
            (
                ctx.tex_loader.get_flat(&sec.ceil_tex),
                shade_surface(SurfaceKind::Ceiling, &sec.ceil_tex, sec.light),
                sec_light,
            )
        };
        let cy = sec.ceil_h as f32;
        let p0_c = doom_to_sliver(poly[0].0 as f32, poly[0].1 as f32, cy);
        for i in 1..poly.len() - 1 {
            let p1 = doom_to_sliver(poly[i + 1].0 as f32, poly[i + 1].1 as f32, cy);
            let p2 = doom_to_sliver(poly[i].0 as f32, poly[i].1 as f32, cy);
            let uv1 = [poly[i + 1].0 as f32, -(poly[i + 1].1 as f32)];
            let uv2 = [poly[i].0 as f32, -(poly[i].1 as f32)];
            push_if_non_degenerate(
                ctx.tris,
                p0_c,
                p1,
                p2,
                uv0,
                uv1,
                uv2,
                ceil_col,
                ceil_tex_id,
                ceil_light,
            );
        }
    }
}

/// Sutherland-Hodgman 2D half-plane clip in Doom coordinates.
/// When `keep_right == true`, keeps points on the right side of `(ax, ay) + t*(dx, dy)`.
fn clip_poly_halfplane(
    poly: &[(f64, f64)],
    ax: f64,
    ay: f64,
    dx: f64,
    dy: f64,
    keep_right: bool,
) -> Vec<(f64, f64)> {
    let len = (dx * dx + dy * dy).sqrt();
    if poly.len() < 3 || len < 1e-9 {
        return poly.to_vec();
    }
    let inv_len = 1.0 / len;
    let sign = if keep_right { -1.0 } else { 1.0 };
    let dist = |p: (f64, f64)| -> f64 {
        let cross = dx * (p.1 - ay) - dy * (p.0 - ax);
        sign * cross * inv_len
    };

    const EPS: f64 = 1e-3;
    let mut out = Vec::with_capacity(poly.len() + 1);
    let mut prev = *poly.last().unwrap();
    let mut d_prev = dist(prev);

    for &curr in poly {
        let d_curr = dist(curr);
        let prev_in = d_prev >= -EPS;
        let curr_in = d_curr >= -EPS;

        if prev_in != curr_in {
            let denom = d_prev - d_curr;
            if denom.abs() > 1e-12 {
                let t = (d_prev / denom).clamp(0.0, 1.0);
                let inter = (
                    prev.0 + (curr.0 - prev.0) * t,
                    prev.1 + (curr.1 - prev.1) * t,
                );
                push_dedup(&mut out, inter);
            }
        }
        if curr_in {
            push_dedup(&mut out, curr);
        }
        prev = curr;
        d_prev = d_curr;
    }

    if out.len() >= 2 {
        let first = out[0];
        let last = *out.last().unwrap();
        if (first.0 - last.0).hypot(first.1 - last.1) < 1e-2 {
            out.pop();
        }
    }
    out
}

fn push_dedup(out: &mut Vec<(f64, f64)>, p: (f64, f64)) {
    if let Some(&last) = out.last()
        && (p.0 - last.0).hypot(p.1 - last.1) < 1e-2
    {
        return;
    }
    out.push(p);
}

fn poly_signed_area2(poly: &[(f64, f64)]) -> f64 {
    let mut sum = 0.0;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        sum += a.0 * b.1 - b.0 * a.1;
    }
    sum
}

#[allow(clippy::too_many_arguments)]
fn find_floor_height(
    px: f64,
    py: f64,
    nodes: &[Node],
    ssectors: &[Subsector],
    segs: &[Seg],
    linedefs: &[Linedef],
    sidedefs: &[Sidedef],
    sectors: &[Sector],
) -> f32 {
    let ss_idx = if let Some(root) = nodes.len().checked_sub(1) {
        let mut child = root as u16;
        let mut guard = 0usize;
        while (child & SUBSECTOR_BIT) == 0 && guard < nodes.len() + 1 {
            guard += 1;
            let Some(&n) = nodes.get(child as usize) else {
                return sectors[0].floor_h as f32;
            };
            let cross = (n.dx as f64) * (py - n.y as f64) - (n.dy as f64) * (px - n.x as f64);
            child = if cross <= 0.0 {
                n.right_child
            } else {
                n.left_child
            };
        }
        (child & !SUBSECTOR_BIT) as usize
    } else {
        0
    };

    ssectors
        .get(ss_idx)
        .and_then(|&ss| subsector_sector(ss, segs, linedefs, sidedefs))
        .and_then(|sec_idx| sectors.get(sec_idx))
        .map(|s| s.floor_h as f32)
        .unwrap_or(sectors[0].floor_h as f32)
}

#[allow(clippy::too_many_arguments)]
fn push_wall(
    out: &mut Vec<Tri>,
    p1: (f32, f32),
    p2: (f32, f32),
    y_bot: f32,
    y_top: f32,
    u0: f32,
    u1: f32,
    v_top: f32,
    v_bot: f32,
    color: [u8; 3],
    tex_id: u16,
    light: u8,
) {
    if y_top <= y_bot {
        return;
    }
    let a = doom_to_sliver(p1.0, p1.1, y_bot);
    let b = doom_to_sliver(p2.0, p2.1, y_bot);
    let c = doom_to_sliver(p2.0, p2.1, y_top);
    let d = doom_to_sliver(p1.0, p1.1, y_top);
    push_if_non_degenerate(
        out,
        a,
        b,
        c,
        [u0, v_bot],
        [u1, v_bot],
        [u1, v_top],
        color,
        tex_id,
        light,
    );
    push_if_non_degenerate(
        out,
        a,
        c,
        d,
        [u0, v_bot],
        [u1, v_top],
        [u0, v_top],
        color,
        tex_id,
        light,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_if_non_degenerate(
    out: &mut Vec<Tri>,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    uv0: [f32; 2],
    uv1: [f32; 2],
    uv2: [f32; 2],
    color: [u8; 3],
    tex_id: u16,
    light: u8,
) {
    let area2 = (p1 - p0).cross(p2 - p0).len();
    if area2 > 1e-6 {
        out.push(Tri::new_textured(
            p0, p1, p2, uv0, uv1, uv2, color, tex_id, light,
        ));
    }
}

#[derive(Clone, Copy)]
enum SurfaceKind {
    Floor,
    Ceiling,
    Wall,
    Trim,
}

fn shade_surface(kind: SurfaceKind, tex: &[u8; 8], light: i16) -> [u8; 3] {
    let name = name_str(tex);
    let base: [i32; 3] = if name.starts_with("NUKAGE") || name.starts_with("SLIME0") {
        [72, 148, 76]
    } else if name.starts_with("LAVA") || name.starts_with("BLOOD") {
        [168, 76, 56]
    } else if name.starts_with("FWATER") {
        [68, 108, 156]
    } else {
        match kind {
            SurfaceKind::Floor => [118, 114, 108],
            SurfaceKind::Ceiling => [88, 92, 98],
            SurfaceKind::Wall => [146, 142, 134],
            SurfaceKind::Trim => [130, 116, 102],
        }
    };

    // Small deterministic tint from the texture name so architectural materials stay distinct.
    let mut h: u32 = 2_166_136_261;
    for &b in tex {
        h ^= b as u32;
        h = h.wrapping_mul(16_777_619);
    }
    let dr = ((h & 0x1F) as i32) - 16;
    let dg = (((h >> 5) & 0x1F) as i32) - 16;
    let db = (((h >> 10) & 0x1F) as i32) - 16;

    let light_norm = (light.clamp(0, 255) as f32) / 255.0;
    let scale = 0.25 + 0.75 * light_norm;

    let ch = |v: i32| -> u8 {
        ((v.clamp(24, 240) as f32) * scale)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    [ch(base[0] + dr), ch(base[1] + dg), ch(base[2] + db)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_name(buf: &mut Vec<u8>, s: &str) {
        let mut raw = [0u8; 8];
        for (dst, &src) in raw.iter_mut().zip(s.as_bytes()) {
            *dst = src;
        }
        buf.extend_from_slice(&raw);
    }

    fn build_wad(lumps: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"IWAD");
        out.extend_from_slice(&(lumps.len() as i32).to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes()); // patched later

        let mut dir = Vec::new();
        for (name, data) in lumps {
            let pos = out.len() as i32;
            out.extend_from_slice(data);
            dir.extend_from_slice(&pos.to_le_bytes());
            dir.extend_from_slice(&(data.len() as i32).to_le_bytes());
            push_name(&mut dir, name);
        }
        let dir_off = out.len() as i32;
        out[8..12].copy_from_slice(&dir_off.to_le_bytes());
        out.extend_from_slice(&dir);
        out
    }

    fn synthetic_box_lumps(ceil_tex: &str) -> Vec<(&'static str, Vec<u8>)> {
        let mut things = Vec::new();
        for v in [0i16, 0, 90, 1, 7] {
            things.extend_from_slice(&v.to_le_bytes());
        }

        let mut verts = Vec::new();
        for (x, y) in [(-128i16, -128i16), (-128, 128), (128, 128), (128, -128)] {
            verts.extend_from_slice(&x.to_le_bytes());
            verts.extend_from_slice(&y.to_le_bytes());
        }

        let mut linedefs = Vec::new();
        for i in 0..4u16 {
            let next = (i + 1) % 4;
            for v in [i, next, 1u16, 0, 0, 0, NO_SIDEDEF] {
                linedefs.extend_from_slice(&v.to_le_bytes());
            }
        }

        let mut sidedefs = Vec::new();
        sidedefs.extend_from_slice(&0i16.to_le_bytes());
        sidedefs.extend_from_slice(&0i16.to_le_bytes());
        push_name(&mut sidedefs, "-");
        push_name(&mut sidedefs, "-");
        push_name(&mut sidedefs, "STARTAN3");
        sidedefs.extend_from_slice(&0u16.to_le_bytes());

        let mut segs = Vec::new();
        for i in 0..4u16 {
            let next = (i + 1) % 4;
            for v in [i, next, 0, i, 0, 0] {
                segs.extend_from_slice(&v.to_le_bytes());
            }
        }

        let mut ssectors = Vec::new();
        ssectors.extend_from_slice(&4u16.to_le_bytes());
        ssectors.extend_from_slice(&0u16.to_le_bytes());

        let mut sectors = Vec::new();
        sectors.extend_from_slice(&0i16.to_le_bytes());
        sectors.extend_from_slice(&128i16.to_le_bytes());
        push_name(&mut sectors, "FLOOR4_8");
        push_name(&mut sectors, ceil_tex);
        sectors.extend_from_slice(&192i16.to_le_bytes());
        sectors.extend_from_slice(&0i16.to_le_bytes());
        sectors.extend_from_slice(&0i16.to_le_bytes());

        vec![
            ("E1M1", Vec::new()),
            ("THINGS", things),
            ("LINEDEFS", linedefs),
            ("SIDEDEFS", sidedefs),
            ("VERTEXES", verts),
            ("SEGS", segs),
            ("SSECTORS", ssectors),
            ("NODES", Vec::new()),
            ("SECTORS", sectors),
        ]
    }

    #[test]
    fn loads_synthetic_room_with_inward_normals_and_spawn() {
        let wad = build_wad(&synthetic_box_lumps("CEIL3_5"));
        let map = load_wad_map_bytes(&wad, None).expect("synthetic WAD should parse");
        assert_eq!(map.name, "E1M1");
        assert_eq!(map.tris.len(), 12);
        assert!(map.spawn_pos.len() < 1e-4);
        assert!(map.spawn_yaw.abs() < 1e-4);

        let floor_tris = map.tris.iter().filter(|t| t.normal().y > 0.99).count();
        let ceil_tris = map.tris.iter().filter(|t| t.normal().y < -0.99).count();
        let wall_tris = map
            .tris
            .iter()
            .filter(|t| t.normal().y.abs() < 0.01)
            .count();
        assert_eq!(floor_tris, 2);
        assert_eq!(ceil_tris, 2);
        assert_eq!(wall_tris, 8);
    }

    #[test]
    fn sky_ceiling_emits_sky_tagged_triangles() {
        let wad = build_wad(&synthetic_box_lumps("F_SKY1"));
        let map = load_wad_map_bytes(&wad, Some("e1m1")).expect("sky WAD should parse");
        let ceil_tris: Vec<_> = map.tris.iter().filter(|t| t.normal().y < -0.99).collect();
        assert_eq!(ceil_tris.len(), 2);
        assert!(ceil_tris.iter().all(|t| t.tex_id == SKY_TEX));
        assert_eq!(map.tris.len(), 12);
    }

    #[test]
    fn sky_texture_name_matches_doom_episodes_and_doom2() {
        assert_eq!(name_str(&sky_texture_name("E1M1")), "SKY1");
        assert_eq!(name_str(&sky_texture_name("E2M2")), "SKY2");
        assert_eq!(name_str(&sky_texture_name("E3M6")), "SKY3");
        assert_eq!(name_str(&sky_texture_name("E4M1")), "SKY4");
        assert_eq!(name_str(&sky_texture_name("MAP01")), "SKY1");
        assert_eq!(name_str(&sky_texture_name("MAP15")), "SKY2");
        assert_eq!(name_str(&sky_texture_name("MAP30")), "SKY3");
    }

    #[test]
    fn decodes_palette_and_flat_texture() {
        let mut lumps = synthetic_box_lumps("F_SKY1");
        let mut playpal = vec![0u8; 768];
        // Palette index 42 -> (10, 200, 30)
        playpal[42 * 3] = 10;
        playpal[42 * 3 + 1] = 200;
        playpal[42 * 3 + 2] = 30;
        lumps.push(("PLAYPAL", playpal));
        lumps.push(("F_START", Vec::new()));
        lumps.push(("FLOOR4_8", vec![42u8; 4096]));
        lumps.push(("F_END", Vec::new()));

        let wad = build_wad(&lumps);
        let map = load_wad_map_bytes(&wad, None).expect("textured synthetic WAD should parse");
        assert_eq!(map.textures.len(), 1);
        assert_eq!(map.textures[0].width, 64);
        assert_eq!(map.textures[0].height, 64);
        assert_eq!(map.textures[0].pixels[0], 0x000A_C81E);
        assert!(
            map.tris
                .iter()
                .any(|t| t.normal().y > 0.99 && t.tex_id == 0)
        );
    }

    #[test]
    fn rejects_invalid_header() {
        let err = load_wad_map_bytes(b"NOPE00000000", None).unwrap_err();
        assert!(err.contains("Invalid WAD header"));
    }

    #[test]
    fn rounded_two_sided_seg_does_not_open_floor_wedge() {
        let mut things = Vec::new();
        for v in [0i16, 0, 90, 1, 7] {
            things.extend_from_slice(&v.to_le_bytes());
        }

        // 256x256 room plus vertex 4 rounded off the diagonal (-128,-128)->(128,128).
        let mut verts = Vec::new();
        for (x, y) in [
            (-128i16, -128i16),
            (-128, 128),
            (128, 128),
            (128, -128),
            (0, -2),
        ] {
            verts.extend_from_slice(&x.to_le_bytes());
            verts.extend_from_slice(&y.to_le_bytes());
        }

        let mut linedefs = Vec::new();
        for i in 0..4u16 {
            let next = (i + 1) % 4;
            for v in [i, next, 1u16, 0, 0, 0, NO_SIDEDEF] {
                linedefs.extend_from_slice(&v.to_le_bytes());
            }
        }
        // Linedef 4: internal two-sided diagonal 0 -> 2.
        for v in [0u16, 2, 4, 0, 0, 0, 0] {
            linedefs.extend_from_slice(&v.to_le_bytes());
        }

        let mut sidedefs = Vec::new();
        sidedefs.extend_from_slice(&0i16.to_le_bytes());
        sidedefs.extend_from_slice(&0i16.to_le_bytes());
        push_name(&mut sidedefs, "-");
        push_name(&mut sidedefs, "-");
        push_name(&mut sidedefs, "STARTAN3");
        sidedefs.extend_from_slice(&0u16.to_le_bytes());

        // Subsector 0 (right of diagonal): walls 2, 3 + rounded two-sided seg 0->4.
        // Subsector 1 (left of diagonal): walls 0, 1 + rounded two-sided seg 2->4.
        let mut segs = Vec::new();
        for &(v1, v2, ld, dir) in &[
            (2u16, 3u16, 2u16, 0u16),
            (3, 0, 3, 0),
            (0, 4, 4, 0),
            (0, 1, 0, 0),
            (1, 2, 1, 0),
            (2, 4, 4, 1),
        ] {
            for v in [v1, v2, 0, ld, dir, 0] {
                segs.extend_from_slice(&v.to_le_bytes());
            }
        }

        let mut ssectors = Vec::new();
        for &(count, first) in &[(3u16, 0u16), (3, 3)] {
            ssectors.extend_from_slice(&count.to_le_bytes());
            ssectors.extend_from_slice(&first.to_le_bytes());
        }

        let mut nodes = Vec::new();
        for v in [-128i16, -128, 256, 256] {
            nodes.extend_from_slice(&v.to_le_bytes());
        }
        nodes.extend_from_slice(&[0u8; 16]); // bboxes unused
        nodes.extend_from_slice(&(SUBSECTOR_BIT).to_le_bytes());
        nodes.extend_from_slice(&(SUBSECTOR_BIT | 1).to_le_bytes());

        let mut sectors = Vec::new();
        sectors.extend_from_slice(&0i16.to_le_bytes());
        sectors.extend_from_slice(&128i16.to_le_bytes());
        push_name(&mut sectors, "FLOOR4_8");
        push_name(&mut sectors, "F_SKY1");
        sectors.extend_from_slice(&192i16.to_le_bytes());
        sectors.extend_from_slice(&0i16.to_le_bytes());
        sectors.extend_from_slice(&0i16.to_le_bytes());

        let wad = build_wad(&[
            ("E1M1", Vec::new()),
            ("THINGS", things),
            ("LINEDEFS", linedefs),
            ("SIDEDEFS", sidedefs),
            ("VERTEXES", verts),
            ("SEGS", segs),
            ("SSECTORS", ssectors),
            ("NODES", nodes),
            ("SECTORS", sectors),
        ]);
        let map = load_wad_map_bytes(&wad, None).expect("split WAD should parse");
        let floor_area: f32 = map
            .tris
            .iter()
            .filter(|t| t.normal().y > 0.99)
            .map(|t| 0.5 * (t.p1 - t.p0).cross(t.p2 - t.p0).len())
            .sum();
        let expected = (256.0 * DOOM_SCALE) * (256.0 * DOOM_SCALE);
        assert!(
            (floor_area - expected).abs() < 1e-3,
            "floor area {floor_area} != expected {expected}"
        );
    }
}
