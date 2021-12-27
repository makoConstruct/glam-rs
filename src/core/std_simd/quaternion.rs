use std::simd::*;
use core::ops::*;
use crate::core::{
    storage::XYZ,
    traits::{quaternion::Quaternion, scalar::*, vector::*},
};
use super::wrapper::*;

impl Quaternion<f32> for f32x4 {
    type SIMDVector3 = f32x4;

    #[inline(always)]
    fn conjugate(self) -> Self {
        const SIGN: f32x4 = const_f32x4!([-1.0, -1.0, -1.0, 1.0]);
        f32x4_mul(self, SIGN)
    }

    #[inline]
    fn lerp(self, end: Self, s: f32) -> Self {
        glam_assert!(FloatVector4::is_normalized(self));
        glam_assert!(FloatVector4::is_normalized(end));

        const NEG_ZERO: f32x4 = const_f32x4!([-0.0; 4]);
        let start = self;
        let end = end;
        let dot = Vector4::dot_into_vec(start, end);
        // Calculate the bias, if the dot product is positive or zero, there is no bias
        // but if it is negative, we want to flip the 'end' rotation XYZW components
        let bias = f32x4_bitand(dot, NEG_ZERO);
        let interpolated = f32x4_add(
            f32x4_mul(f32x4_sub(f32x4_bitxor(end, bias), start), f32x4_splat(s)),
            start,
        );
        FloatVector4::normalize(interpolated)
    }

    #[inline]
    fn slerp(self, end: Self, s: f32) -> Self {
        // http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
        glam_assert!(FloatVector4::is_normalized(self));
        glam_assert!(FloatVector4::is_normalized(end));

        const DOT_THRESHOLD: f32 = 0.9995;

        let dot = Vector4::dot(self, end);

        if dot > DOT_THRESHOLD {
            // assumes lerp returns a normalized quaternion
            self.lerp(end, s)
        } else {
            // assumes scalar_acos clamps the input to [-1.0, 1.0]
            let theta = dot.acos_approx();

            // TODO: v128_sin is broken
            // let x = 1.0 - s;
            // let y = s;
            // let z = 1.0;
            // let w = 0.0;
            // let tmp = f32x4::mul(f32x4_splat(theta), f32x4(x, y, z, w));
            // let tmp = v128_sin(tmp);
            let x = (theta * (1.0 - s)).sin();
            let y = (theta * s).sin();
            let z = theta.sin();
            let w = 0.0;
            let tmp = f32x4::from_array([x, y, z, w]);

            let scale1 = simd_swizzle!(tmp, [0, 0, 0, 0]);
            let scale2 = simd_swizzle!(tmp, [1, 1, 1, 1]);
            let theta_sin = simd_swizzle!(tmp, [2, 2, 2, 2]);

            self.mul(scale1).add(end.mul(scale2)).div(theta_sin)
        }
    }

    #[inline]
    fn mul_quaternion(self, other: Self) -> Self {
        glam_assert!(FloatVector4::is_normalized(self));
        glam_assert!(FloatVector4::is_normalized(other));
        // Based on https://github.com/nfrechette/rtm `rtm::quat_mul`
        let lhs = self;
        let rhs = other;

        const CONTROL_WZYX: f32x4 = const_f32x4!([1.0, -1.0, 1.0, -1.0]);
        const CONTROL_ZWXY: f32x4 = const_f32x4!([1.0, 1.0, -1.0, -1.0]);
        const CONTROL_YXWZ: f32x4 = const_f32x4!([-1.0, 1.0, 1.0, -1.0]);

        let r_xxxx = simd_swizzle!(lhs, [0, 0, 0, 0]);
        let r_yyyy = simd_swizzle!(lhs, [1, 1, 1, 1]);
        let r_zzzz = simd_swizzle!(lhs, [2, 2, 2, 2]);
        let r_wwww = simd_swizzle!(lhs, [3, 3, 3, 3]);

        let lxrw_lyrw_lzrw_lwrw = f32x4_mul(r_wwww, rhs);
        let l_wzyx = simd_swizzle!(rhs, [3, 2, 5, 4]);

        let lwrx_lzrx_lyrx_lxrx = f32x4_mul(r_xxxx, l_wzyx);
        let l_zwxy = simd_swizzle!(l_wzyx, [1, 0, 7, 6]);

        let lwrx_nlzrx_lyrx_nlxrx = f32x4_mul(lwrx_lzrx_lyrx_lxrx, CONTROL_WZYX);

        let lzry_lwry_lxry_lyry = f32x4_mul(r_yyyy, l_zwxy);
        let l_yxwz = simd_swizzle!(l_zwxy, [3, 2, 5, 4]);

        let lzry_lwry_nlxry_nlyry = f32x4_mul(lzry_lwry_lxry_lyry, CONTROL_ZWXY);

        let lyrz_lxrz_lwrz_lzrz = f32x4_mul(r_zzzz, l_yxwz);
        let result0 = f32x4_add(lxrw_lyrw_lzrw_lwrw, lwrx_nlzrx_lyrx_nlxrx);

        let nlyrz_lxrz_lwrz_wlzrz = f32x4_mul(lyrz_lxrz_lwrz_lzrz, CONTROL_YXWZ);
        let result1 = f32x4_add(lzry_lwry_nlxry_nlyry, nlyrz_lxrz_lwrz_wlzrz);
        f32x4_add(result0, result1)
    }

    #[inline]
    fn mul_vector3(self, other: XYZ<f32>) -> XYZ<f32> {
        self.mul_float4_as_vector3(other.into()).into()
    }

    #[inline]
    fn mul_float4_as_vector3(self, other: f32x4) -> f32x4 {
        glam_assert!(FloatVector4::is_normalized(self));
        const TWO: f32x4 = const_f32x4!([2.0; 4]);
        let w = simd_swizzle!(self, [3, 3, 3, 3]);
        let b = self;
        let b2 = Vector3::dot_into_vec(b, b);
        other
            .mul(w.mul(w).sub(b2))
            .add(b.mul(Vector3::dot_into_vec(other, b).mul(TWO)))
            .add(b.cross(other).mul(w.mul(TWO)))
    }
}
