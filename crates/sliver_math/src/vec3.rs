use core::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    #[inline(always)]
    pub fn len_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn len(self) -> f32 {
        self.len_sq().sqrt()
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let l = self.len();
        if l > 1e-12 {
            self * (1.0 / l)
        } else {
            Self::ZERO
        }
    }

    #[inline(always)]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self + (other - self) * t
    }

    #[inline(always)]
    pub fn rotate_y(self, rad: f32) -> Self {
        let (s, c) = rad.sin_cos();
        Self {
            x: self.x * c + self.z * s,
            y: self.y,
            z: -self.x * s + self.z * c,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        Self::new(self.x + r.x, self.y + r.y, self.z + r.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        Self::new(self.x - r.x, self.y - r.y, self.z - r.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}
