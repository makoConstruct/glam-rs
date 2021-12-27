use crate::core::{
    storage::{XY, XYZ, XYZW},
    traits::{scalar::*, vector::*},
};
use std::simd::*;

#[inline(always)]
fn f32x4_isnan(v: f32x4) -> f32x4 {
    f32x4::ne(v, v)
}

/// Calculates the vector 3 dot product and returns answer in x lane of __m128.
#[inline(always)]
fn dot3_in_x(lhs: f32x4, rhs: f32x4) -> f32x4 {
    let x2_y2_z2_w2 = f32x4::mul(lhs, rhs);
    let y2_0_0_0 = simd_swizzle!(x2_y2_z2_w2, [1, 0, 0, 0]);
    let z2_0_0_0 = simd_swizzle!(x2_y2_z2_w2, [2, 0, 0, 0]);
    let x2y2_0_0_0 = f32x4::add(x2_y2_z2_w2, y2_0_0_0);
    f32x4::add(x2y2_0_0_0, z2_0_0_0)
}

/// Calculates the vector 4 dot product and returns answer in x lane of __m128.
#[inline(always)]
fn dot4_in_x(lhs: f32x4, rhs: f32x4) -> f32x4 {
    let x2_y2_z2_w2 = f32x4::mul(lhs, rhs);
    let z2_w2_0_0 = simd_swizzle!(x2_y2_z2_w2, [2, 3, 0, 0]);
    let x2z2_y2w2_0_0 = f32x4::add(x2_y2_z2_w2, z2_w2_0_0);
    let y2w2_0_0_0 = simd_swizzle!(x2z2_y2w2_0_0, [1, 0, 0, 0]);
    f32x4::add(x2z2_y2w2_0_0, y2w2_0_0_0)
}

impl MaskVectorConst for f32x4 {
    const FALSE: f32x4 = const_f32x4!([0.0; 4]);
}

impl MaskVector for f32x4 {
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        f32x4::bitand(self, other)
    }

    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        f32x4::bitor(self, other)
    }

    #[inline]
    fn not(self) -> Self {
        f32x4::bitnot(self)
    }
}

impl MaskVector3 for f32x4 {
    #[inline(always)]
    fn new(x: bool, y: bool, z: bool) -> Self {
        f32x4::from_bits(
            MaskConst::MASK[x as usize],
            MaskConst::MASK[y as usize],
            MaskConst::MASK[z as usize],
            0,
        )
    }

    #[inline(always)]
    fn bitmask(self) -> u32 {
        (self.to_bitmask() & 0x7) as u32
    }

    #[inline(always)]
    fn any(self) -> bool {
        (self.to_bitmask() & 0x7) != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        (self.to_bitmask() & 0x7) == 0x7
    }

    #[inline]
    fn into_bool_array(self) -> [bool; 3] {
        let bitmask = MaskVector3::bitmask(self);
        [(bitmask & 1) != 0, (bitmask & 2) != 0, (bitmask & 4) != 0]
    }

    #[inline]
    fn into_u32_array(self) -> [u32; 3] {
        let bitmask = MaskVector3::bitmask(self);
        [
            MaskConst::MASK[(bitmask & 1) as usize],
            MaskConst::MASK[((bitmask >> 1) & 1) as usize],
            MaskConst::MASK[((bitmask >> 2) & 1) as usize],
        ]
    }
}

impl MaskVector4 for f32x4 {
    #[inline(always)]
    fn new(x: bool, y: bool, z: bool, w: bool) -> Self {
        f32x4::from_bits(
            MaskConst::MASK[x as usize],
            MaskConst::MASK[y as usize],
            MaskConst::MASK[z as usize],
            MaskConst::MASK[w as usize],
        )
    }

    #[inline(always)]
    fn bitmask(self) -> u32 {
        self.to_bitmask() as u32
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.to_bitmask() != [0; 4]
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.to_bitmask() == [0xff; 4]
    }

    #[inline]
    fn into_bool_array(self) -> [bool; 4] {
        self.to_array()
    }

    #[inline]
    fn into_u32_array(self) -> [u32; 4] {
        self.to_int() as [u32; 4]
    }
}

impl VectorConst for f32x4 {
    const ZERO: f32x4 = const_f32x4!([0.0; 4]);
    const ONE: f32x4 = const_f32x4!([1.0; 4]);
}

impl NanConstEx for f32x4 {
    const NAN: f32x4 = const_f32x4!([f32::NAN; 4]);
}

impl Vector3Const for f32x4 {
    const X: f32x4 = const_f32x4!([1.0, 0.0, 0.0, 0.0]);
    const Y: f32x4 = const_f32x4!([0.0, 1.0, 0.0, 0.0]);
    const Z: f32x4 = const_f32x4!([0.0, 0.0, 1.0, 0.0]);
}

impl Vector4Const for f32x4 {
    const X: f32x4 = const_f32x4!([1.0, 0.0, 0.0, 0.0]);
    const Y: f32x4 = const_f32x4!([0.0, 1.0, 0.0, 0.0]);
    const Z: f32x4 = const_f32x4!([0.0, 0.0, 1.0, 0.0]);
    const W: f32x4 = const_f32x4!([0.0, 0.0, 0.0, 1.0]);
}

impl Vector<f32> for f32x4 {
    type Mask = f32x4;

    #[inline(always)]
    fn splat(s: f32) -> Self {
        f32x4::splat(s)
    }

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        mask.select(if_true, if_false)
    }

    #[inline(always)]
    fn cmpeq(self, other: Self) -> Self::Mask {
        f32x4::eq(self, other)
    }

    #[inline(always)]
    fn cmpne(self, other: Self) -> Self::Mask {
        f32x4::ne(self, other)
    }

    #[inline(always)]
    fn cmpge(self, other: Self) -> Self::Mask {
        f32x4::ge(self, other)
    }

    #[inline(always)]
    fn cmpgt(self, other: Self) -> Self::Mask {
        f32x4::gt(self, other)
    }

    #[inline(always)]
    fn cmple(self, other: Self) -> Self::Mask {
        f32x4::le(self, other)
    }

    #[inline(always)]
    fn cmplt(self, other: Self) -> Self::Mask {
        f32x4::lt(self, other)
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        f32x4::add(self, other)
    }

    #[inline(always)]
    fn div(self, other: Self) -> Self {
        f32x4::div(self, other)
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        f32x4::mul(self, other)
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        f32x4::add(f32x4::mul(self, b), c)
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        f32x4::sub(self, other)
    }

    #[inline(always)]
    fn add_scalar(self, other: f32) -> Self {
        f32x4::add(self, f32x4::splat(other))
    }

    #[inline(always)]
    fn sub_scalar(self, other: f32) -> Self {
        f32x4::sub(self, f32x4::splat(other))
    }

    #[inline(always)]
    fn mul_scalar(self, other: f32) -> Self {
        f32x4::mul(self, f32x4::splat(other))
    }

    #[inline(always)]
    fn div_scalar(self, other: f32) -> Self {
        f32x4::div(self, f32x4::splat(other))
    }

    #[inline(always)]
    fn rem(self, other: Self) -> Self {
        f32x4::rem(self, other)
    }

    #[inline(always)]
    fn rem_scalar(self, other: f32) -> Self {
        self.rem(f32x4::splat(other))
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f32x4::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f32x4::max(self, other)
    }
}

impl Vector3<f32> for f32x4 {
    #[inline(always)]
    fn new(x: f32, y: f32, z: f32) -> Self {
        f32x4::from_array([x, y, z, x])
    }

    #[inline(always)]
    fn x(self) -> f32 {
        self[0]
    }

    #[inline(always)]
    fn y(self) -> f32 {
        self[1]
    }

    #[inline(always)]
    fn z(self) -> f32 {
        self[2]
    }

    #[inline(always)]
    fn splat_x(self) -> Self {
        simd_swizzle!(self, [0, 0, 0, 0])
    }

    #[inline(always)]
    fn splat_y(self) -> Self {
        simd_swizzle!(self, [1, 1, 1, 1])
    }

    #[inline(always)]
    fn splat_z(self) -> Self {
        simd_swizzle!(self, [2, 2, 2, 2])
    }

    #[inline(always)]
    fn from_slice_unaligned(slice: &[f32]) -> Self {
        Vector3::new(slice[0], slice[1], slice[2])
    }

    #[inline(always)]
    fn write_to_slice_unaligned(self, slice: &mut [f32]) {
        let xyz = self.as_ref_xyz();
        slice[0] = xyz.x;
        slice[1] = xyz.y;
        slice[2] = xyz.z;
    }

    #[inline(always)]
    fn as_ref_xyz(&self) -> &XYZ<f32> {
        unsafe { &*(self as *const Self as *const XYZ<f32>) }
    }

    #[inline(always)]
    fn as_mut_xyz(&mut self) -> &mut XYZ<f32> {
        unsafe { &mut *(self as *mut Self as *mut XYZ<f32>) }
    }

    #[inline(always)]
    fn into_xy(self) -> XY<f32> {
        XY {
            x: self[0],
            y: self[1],
        }
    }

    #[inline]
    fn into_xyzw(self, w: f32) -> XYZW<f32> {
        let mut v = self;
        v[3] = w;
        unsafe { *(&v as *const f32x4 as *const XYZW<f32>) }
    }

    #[inline(always)]
    fn from_array(a: [f32; 3]) -> Self {
        Vector3::new(a[0], a[1], a[2])
    }

    #[inline(always)]
    fn into_array(self) -> [f32; 3] {
        [self[0], self[1], self[2]]
    }

    #[inline(always)]
    fn from_tuple(t: (f32, f32, f32)) -> Self {
        Vector3::new(t.0, t.1, t.2)
    }

    #[inline(always)]
    fn into_tuple(self) -> (f32, f32, f32) {
        (self[0], self[1], self[2])
    }

    #[inline]
    fn min_element(self) -> f32 {
        self.horizontal_min()
    }

    #[inline]
    fn max_element(self) -> f32 {
        self.horizontal_max()
    }

    #[inline]
    fn dot(self, other: Self) -> f32 {
        dot3_in_x(self, other)[0]
    }

    #[inline]
    fn dot_into_vec(self, other: Self) -> Self {
        let dot_in_x = dot3_in_x(self, other);
        simd_swizzle!(dot_in_x, [0, 0, 0, 0])
    }

    #[inline]
    fn cross(self, other: Self) -> Self {
        // x  <-  a.y*b.z - a.z*b.y
        // y  <-  a.z*b.x - a.x*b.z
        // z  <-  a.x*b.y - a.y*b.x
        // We can save a shuffle by grouping it in this wacky order:
        // (self.zxy() * other - self * other.zxy()).zxy()
        let lhszxy = simd_swizzle!(self, [2, 0, 1, 1]);
        let rhszxy = simd_swizzle!(other, [2, 0, 1, 1]);
        let lhszxy_rhs = f32x4::mul(lhszxy, other);
        let rhszxy_lhs = f32x4::mul(rhszxy, self);
        let sub = f32x4::sub(lhszxy_rhs, rhszxy_lhs);
        simd_swizzle!(sub, [2, 0, 1, 1])
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        glam_assert!(
            MaskVector3::all(min.cmple(max)),
            "clamp: expected min <= max"
        );
        self.clamp(min, max)
    }
}

impl Vector4<f32> for f32x4 {
    #[inline(always)]
    fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        f32x4::from_array([x, y, z, w])
    }

    #[inline(always)]
    fn x(self) -> f32 {
        self[0]
    }

    #[inline(always)]
    fn y(self) -> f32 {
        self[1]
    }

    #[inline(always)]
    fn z(self) -> f32 {
        self[2]
    }

    #[inline(always)]
    fn w(self) -> f32 {
        self[4]
    }

    #[inline(always)]
    fn splat_x(self) -> Self {
        simd_swizzle!(self, [0, 0, 0, 0])
    }

    #[inline(always)]
    fn splat_y(self) -> Self {
        simd_swizzle!(self, [1, 1, 1, 1])
    }

    #[inline(always)]
    fn splat_z(self) -> Self {
        simd_swizzle!(self, [2, 2, 2, 2])
    }

    #[inline(always)]
    fn splat_w(self) -> Self {
        simd_swizzle!(self, [3, 3, 3, 3])
    }

    #[inline(always)]
    fn from_slice_unaligned(slice: &[f32]) -> Self {
        f32x4::from_slice(slice)
    }

    #[inline(always)]
    fn write_to_slice_unaligned(self, slice: &mut [f32]) {
        slice[0] = self[0];
        slice[1] = self[1];
        slice[2] = self[2];
        slice[3] = self[3];
    }

    #[inline(always)]
    fn as_ref_xyzw(&self) -> &XYZW<f32> {
        unsafe { &*(self as *const Self as *const XYZW<f32>) }
    }

    #[inline(always)]
    fn as_mut_xyzw(&mut self) -> &mut XYZW<f32> {
        unsafe { &mut *(self as *mut Self as *mut XYZW<f32>) }
    }

    #[inline(always)]
    fn into_xy(self) -> XY<f32> {
        XY {
            x: self[0],
            y: self[1],
        }
    }

    #[inline(always)]
    fn into_xyz(self) -> XYZ<f32> {
        XYZ {
            x: self[0],
            y: self[1],
            z: self[2],
        }
    }

    #[inline(always)]
    fn from_array(a: [f32; 4]) -> Self {
        f32x4::from_array(a)
    }

    #[inline(always)]
    fn into_array(self) -> [f32; 4] {
        self.to_array()
    }

    #[inline(always)]
    fn from_tuple(t: (f32, f32, f32, f32)) -> Self {
        f32x4::from_array([t.0, t.1, t.2, t.3])
    }

    #[inline(always)]
    fn into_tuple(self) -> (f32, f32, f32, f32) {
        (self[0], self[1], self[2], self[3])
    }

    #[inline]
    fn min_element(self) -> f32 {
        self.horizontal_min()
    }

    #[inline]
    fn max_element(self) -> f32 {
        self.horizontal_max()
    }

    #[inline]
    fn dot(self, other: Self) -> f32 {
        dot4_in_x(self, other)[0]
    }

    #[inline]
    fn dot_into_vec(self, other: Self) -> Self {
        let dot_in_x = dot4_in_x(self, other);
        simd_swizzle!(dot_in_x, [0, 0, 0, 0])
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        glam_assert!(
            MaskVector4::all(min.cmple(max)),
            "clamp: expected min <= max"
        );
        self.clamp(min, max)
    }
}

impl SignedVector<f32> for f32x4 {
    #[inline(always)]
    fn neg(self) -> Self {
        f32x4::neg(self)
    }
}

impl SignedVector3<f32> for f32x4 {
    #[inline]
    fn abs(self) -> Self {
        f32x4::abs(self)
    }

    #[inline]
    fn signum(self) -> Self {
        f32x4::signum(self)
    }
}

impl SignedVector4<f32> for f32x4 {
    #[inline]
    fn abs(self) -> Self {
        f32x4::abs(self)
    }

    #[inline]
    fn signum(self) -> Self {
        f32x4::signum(self)
    }
}

impl FloatVector3<f32> for f32x4 {
    #[inline]
    fn is_finite(self) -> bool {
        let m = f32x4::is_finite(self);
        m[0] == -1 && m[1] == -1 && m[2] == -1
    }

    #[inline]
    fn is_nan(self) -> bool {
        MaskVector3::any(FloatVector3::is_nan_mask(self))
    }

    #[inline(always)]
    fn is_nan_mask(self) -> Self::Mask {
        f32x4::is_nan(self)
    }

    #[inline]
    fn floor(self) -> Self {
        f32x4::floor(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        f32x4::ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        f32x4::round(self)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        f32x4::recip(self)
    }

    #[inline]
    fn exp(self) -> Self {
        Vector3::new(self[0].exp(), self[1].exp(), self[2].exp())
    }

    #[inline]
    fn powf(self, n: f32) -> Self {
        Vector3::new(self[0].powf(n), self[1].powf(n), self[2].powf(n))
    }

    #[inline]
    fn length(self) -> f32 {
        let dot = dot3_in_x(self, self);
        f32x4::sqrt(dot)[0]
    }

    #[inline]
    fn length_recip(self) -> f32 {
        let dot = dot3_in_x(self, self);
        f32x4::recip(f32x4::sqrt(dot))[0]
    }

    #[inline]
    fn normalize(self) -> Self {
        let length = f32x4::sqrt(Vector3::dot_into_vec(self, self));
        #[allow(clippy::let_and_return)]
        let normalized = f32x4::div(self, length);
        glam_assert!(FloatVector3::is_finite(normalized));
        normalized
    }
}

impl FloatVector4<f32> for f32x4 {
    #[inline]
    fn is_finite(self) -> bool {
        let m = f32x4::is_finite(self);
        m == [-1, -1, -1, -1]
    }

    #[inline]
    fn is_nan(self) -> bool {
        MaskVector4::any(FloatVector4::is_nan_mask(self))
    }

    #[inline(always)]
    fn is_nan_mask(self) -> Self::Mask {
        f32x4::is_nan(self)
    }

    #[inline]
    fn floor(self) -> Self {
        f32x4::floor(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        f32x4::ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        f32x4::round(self)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        f32x4::div(Self::ONE, self)
    }

    #[inline]
    fn exp(self) -> Self {
        f32x4::from_array([self[0].exp(), self[1].exp(), self[2].exp(), self[3].exp()])
    }

    #[inline]
    fn powf(self, n: f32) -> Self {
        f32x4::from_array([self[0].powf(n), self[1].powf(n), self[2].powf(n), self[3].powf(n)])
    }

    #[inline]
    fn length(self) -> f32 {
        let dot = dot4_in_x(self, self);
        f32x4::sqrt(dot)[0]
    }

    #[inline]
    fn length_recip(self) -> f32 {
        let dot = dot4_in_x(self, self);
        f32x4::recip(f32x4::sqrt(dot))
    }

    #[inline]
    fn normalize(self) -> Self {
        let dot = Vector4::dot_into_vec(self, self);
        #[allow(clippy::let_and_return)]
        let normalized = f32x4::div(self, f32x4::sqrt(dot));
        glam_assert!(FloatVector4::is_finite(normalized));
        normalized
    }
}

impl From<XYZW<f32>> for f32x4 {
    #[inline(always)]
    fn from(v: XYZW<f32>) -> f32x4 {
        f32x4::from_array([v.x, v.y, v.z, v.w])
    }
}

impl From<XYZ<f32>> for f32x4 {
    #[inline(always)]
    fn from(v: XYZ<f32>) -> f32x4 {
        f32x4::from_array([v.x, v.y, v.z, v.z])
    }
}

impl From<XY<f32>> for f32x4 {
    #[inline(always)]
    fn from(v: XY<f32>) -> f32x4 {
        f32x4::from_array([v.x, v.y, v.y, v.y])
    }
}

impl From<f32x4> for XYZW<f32> {
    #[inline(always)]
    fn from(v: f32x4) -> XYZW<f32> {
        XYZW {
            x: v[0],
            y: v[1],
            z: v[2],
            w: v[3],
        }
    }
}

impl From<f32x4> for XYZ<f32> {
    #[inline(always)]
    fn from(v: f32x4) -> XYZ<f32> {
        XYZ {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }
}

impl From<f32x4> for XY<f32> {
    #[inline(always)]
    fn from(v: f32x4) -> XY<f32> {
        XY {
            x: v[0],
            y: v[1],
        }
    }
}
