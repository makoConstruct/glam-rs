use std::simd::{*, Which::*};

use crate::core::{
    storage::{Columns2, Columns3, Columns4, XY, XYZ},
    traits::{
        matrix::{
            FloatMatrix2x2, FloatMatrix3x3, FloatMatrix4x4, Matrix, Matrix2x2, Matrix3x3,
            Matrix4x4, MatrixConst,
        },
        projection::ProjectionMatrix,
        scalar::NanConstEx,
        vector::{FloatVector4, Vector, Vector4, Vector4Const, VectorConst},
    },
};

// f32x4 as a Matrix2x2
impl MatrixConst for f32x4 {
    const ZERO: f32x4 = const_f32x4!([0.0, 0.0, 0.0, 0.0]);
    const IDENTITY: f32x4 = const_f32x4!([1.0, 0.0, 0.0, 1.0]);
}

impl Matrix<f32> for f32x4 {}

impl Matrix2x2<f32, XY<f32>> for f32x4 {
    #[inline(always)]
    fn new(m00: f32, m01: f32, m10: f32, m11: f32) -> Self {
        f32x4::from_array([m00, m01, m10, m11])
    }

    #[inline(always)]
    fn from_cols(x_axis: XY<f32>, y_axis: XY<f32>) -> Self {
        Matrix2x2::new(x_axis.x, x_axis.y, y_axis.x, y_axis.y)
    }

    #[inline(always)]
    fn x_axis(&self) -> &XY<f32> {
        unsafe { &(*(self as *const Self as *const Columns2<XY<f32>>)).x_axis }
    }

    #[inline(always)]
    fn y_axis(&self) -> &XY<f32> {
        unsafe { &(*(self as *const Self as *const Columns2<XY<f32>>)).y_axis }
    }

    #[inline]
    fn determinant(&self) -> f32 {
        // self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
        let abcd = *self;
        let dcba = simd_swizzle!(abcd, [3, 2, 1, 0]);
        let prod = f32x4::add(abcd, dcba);
        let det = f32x4::sub(prod, simd_swizzle!(prod, [1, 1, 1, 1]));
        f32x4::index(det, 0)
    }

    #[inline(always)]
    fn transpose(&self) -> Self {
        simd_swizzle!(*self, [0, 2, 1, 3])
    }

    #[inline]
    fn mul_vector(&self, other: XY<f32>) -> XY<f32> {
        let abcd = *self;
        let xxyy = f32x4::from_array([other.x, other.x, other.y, other.y]);
        let axbxcydy = f32x4::mul(abcd, xxyy);
        let cydyaxbx = simd_swizzle!(axbxcydy, [2, 3, 0, 1]);
        let result = f32x4::add(axbxcydy, cydyaxbx);
        *(&result as *const f32x4 as *const XY<f32>)
    }

    #[inline]
    fn mul_matrix(&self, other: &Self) -> Self {
        let abcd = *self;
        let other = *other;
        let xxyy0 = simd_swizzle!(other, [0, 0, 1, 1]);
        let xxyy1 = simd_swizzle!(other, [2, 2, 3, 3]);
        let axbxcydy0 = f32x4::mul(abcd, xxyy0);
        let axbxcydy1 = f32x4::mul(abcd, xxyy1);
        let cydyaxbx0 = simd_swizzle!(axbxcydy0, [2, 3, 0, 1]);
        let cydyaxbx1 = simd_swizzle!(axbxcydy1, [2, 3, 0, 1]);
        let result0 = f32x4::add(axbxcydy0, cydyaxbx0);
        let result1 = f32x4::add(axbxcydy1, cydyaxbx1);
        simd_swizzle!(result0, result1, [First(0), First(1), Second(0), Second(1)])
    }

    #[inline]
    fn mul_scalar(&self, other: f32) -> Self {
        f32x4::mul(*self, f32x4::splat(other))
    }

    #[inline]
    fn add_matrix(&self, other: &Self) -> Self {
        f32x4::add(*self, *other)
    }

    #[inline]
    fn sub_matrix(&self, other: &Self) -> Self {
        f32x4::sub(*self, *other)
    }
}

impl FloatMatrix2x2<f32, XY<f32>> for f32x4 {
    #[inline]
    fn abs_diff_eq(&self, other: &Self, max_abs_diff: f32) -> bool {
        FloatVector4::abs_diff_eq(*self, *other, max_abs_diff)
    }

    #[inline]
    fn neg_matrix(&self) -> Self {
        f32x4::neg(*self)
    }

    #[inline]
    fn inverse(&self) -> Self {
        const SIGN: f32x4 = const_f32x4!([1.0, -1.0, -1.0, 1.0]);
        let abcd = *self;
        let dcba = simd_swizzle!(abcd, [3, 2, 1, 0]);
        let prod = f32x4::mul(abcd, dcba);
        let sub = f32x4::sub(prod, simd_swizzle!(prod, [1, 1, 1, 1]));
        let det = simd_swizzle!(sub, [0, 0, 0, 0]);
        let tmp = f32x4::div(SIGN, det);
        glam_assert!(tmp.is_finite());
        let dbca = simd_swizzle!(abcd, [3, 1, 2, 0]);
        f32x4::mul(dbca, tmp)
    }
}

impl MatrixConst for Columns3<f32x4> {
    const ZERO: Columns3<f32x4> = Columns3 {
        x_axis: VectorConst::ZERO,
        y_axis: VectorConst::ZERO,
        z_axis: VectorConst::ZERO,
    };
    const IDENTITY: Columns3<f32x4> = Columns3 {
        x_axis: f32x4::X,
        y_axis: f32x4::Y,
        z_axis: f32x4::Z,
    };
}

impl NanConstEx for Columns3<f32x4> {
    const NAN: Columns3<f32x4> = Columns3 {
        x_axis: f32x4::NAN,
        y_axis: f32x4::NAN,
        z_axis: f32x4::NAN,
    };
}

impl Matrix<f32> for Columns3<f32x4> {}

impl Matrix3x3<f32, f32x4> for Columns3<f32x4> {
    #[inline(always)]
    fn from_cols(x_axis: f32x4, y_axis: f32x4, z_axis: f32x4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    #[inline(always)]
    fn x_axis(&self) -> &f32x4 {
        &self.x_axis
    }

    #[inline(always)]
    fn y_axis(&self) -> &f32x4 {
        &self.y_axis
    }

    #[inline(always)]
    fn z_axis(&self) -> &f32x4 {
        &self.z_axis
    }

    #[inline]
    fn transpose(&self) -> Self {
        let tmp0 = simd_swizzle!(self.x_axis, self.y_axis, [First(0), First(1), Second(0), Second(1)]);
        let tmp1 = simd_swizzle!(self.x_axis, self.y_axis, [First(2), First(3), Second(2), Second(3)]);

        Self {
            x_axis: simd_swizzle!(tmp0, self.z_axis, [First(0), First(2), Second(0), Second(0)]),
            y_axis: simd_swizzle!(tmp0, self.z_axis, [First(1), First(3), Second(1), Second(1)]),
            z_axis: simd_swizzle!(tmp1, self.z_axis, [First(0), First(2), Second(2), Second(2)]),
        }
    }
}

impl FloatMatrix3x3<f32, f32x4> for Columns3<f32x4> {
    #[inline]
    fn transform_point2(&self, other: XY<f32>) -> XY<f32> {
        let mut res = self.x_axis.mul_scalar(other.x);
        res = self.y_axis.mul_scalar(other.y).add(res);
        res = self.z_axis.add(res);
        res.into()
    }

    #[inline]
    fn transform_vector2(&self, other: XY<f32>) -> XY<f32> {
        let mut res = self.x_axis.mul_scalar(other.x);
        res = self.y_axis.mul_scalar(other.y).add(res);
        res.into()
    }
}

impl MatrixConst for Columns4<f32x4> {
    const ZERO: Columns4<f32x4> = Columns4 {
        x_axis: VectorConst::ZERO,
        y_axis: VectorConst::ZERO,
        z_axis: VectorConst::ZERO,
        w_axis: VectorConst::ZERO,
    };
    const IDENTITY: Columns4<f32x4> = Columns4 {
        x_axis: f32x4::X,
        y_axis: f32x4::Y,
        z_axis: f32x4::Z,
        w_axis: f32x4::W,
    };
}

impl NanConstEx for Columns4<f32x4> {
    const NAN: Columns4<f32x4> = Columns4 {
        x_axis: f32x4::NAN,
        y_axis: f32x4::NAN,
        z_axis: f32x4::NAN,
        w_axis: f32x4::NAN,
    };
}

impl Matrix<f32> for Columns4<f32x4> {}

impl Matrix4x4<f32, f32x4> for Columns4<f32x4> {
    #[inline(always)]
    fn from_cols(x_axis: f32x4, y_axis: f32x4, z_axis: f32x4, w_axis: f32x4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    #[inline(always)]
    fn x_axis(&self) -> &f32x4 {
        &self.x_axis
    }

    #[inline(always)]
    fn y_axis(&self) -> &f32x4 {
        &self.y_axis
    }

    #[inline(always)]
    fn z_axis(&self) -> &f32x4 {
        &self.z_axis
    }

    #[inline(always)]
    fn w_axis(&self) -> &f32x4 {
        &self.w_axis
    }

    #[inline]
    fn determinant(&self) -> f32 {
        // Based on https://github.com/g-truc/glm `glm_mat4_determinant`
        let swp2a = i32x4_shuffle::<2, 1, 1, 0>(self.z_axis, self.z_axis);
        let swp3a = i32x4_shuffle::<3, 3, 2, 3>(self.w_axis, self.w_axis);
        let swp2b = i32x4_shuffle::<3, 3, 2, 3>(self.z_axis, self.z_axis);
        let swp3b = i32x4_shuffle::<2, 1, 2, 0>(self.w_axis, self.w_axis);
        let swp2c = i32x4_shuffle::<2, 1, 0, 0>(self.z_axis, self.z_axis);
        let swp3c = i32x4_shuffle::<0, 0, 2, 1>(self.w_axis, self.w_axis);

        let mula = f32x4_mul(swp2a, swp3a);
        let mulb = f32x4_mul(swp2b, swp3b);
        let mulc = f32x4_mul(swp2c, swp3c);
        let sube = f32x4_sub(mula, mulb);
        let subf = f32x4_sub(i32x4_shuffle::<6, 7, 2, 3>(mulc, mulc), mulc);

        let subfaca = i32x4_shuffle::<0, 0, 1, 2>(sube, sube);
        let swpfaca = i32x4_shuffle::<1, 0, 0, 0>(self.y_axis, self.y_axis);
        let mulfaca = f32x4_mul(swpfaca, subfaca);

        let subtmpb = i32x4_shuffle::<1, 3, 4, 4>(sube, subf);
        let subfacb = i32x4_shuffle::<0, 1, 1, 3>(subtmpb, subtmpb);
        let swpfacb = i32x4_shuffle::<2, 2, 1, 1>(self.y_axis, self.y_axis);
        let mulfacb = f32x4_mul(swpfacb, subfacb);

        let subres = f32x4_sub(mulfaca, mulfacb);
        let subtmpc = i32x4_shuffle::<2, 2, 4, 5>(sube, subf);
        let subfacc = i32x4_shuffle::<0, 2, 3, 3>(subtmpc, subtmpc);
        let swpfacc = i32x4_shuffle::<3, 3, 3, 2>(self.y_axis, self.y_axis);
        let mulfacc = f32x4_mul(swpfacc, subfacc);

        let addres = f32x4_add(subres, mulfacc);
        let detcof = f32x4_mul(addres, f32x4(1.0, -1.0, 1.0, -1.0));

        Vector4::dot(self.x_axis, detcof)
    }

    #[inline]
    fn transpose(&self) -> Self {
        // Based on https://github.com/microsoft/DirectXMath `XMMatrixTranspose`
        let tmp0 = i32x4_shuffle::<0, 1, 4, 5>(self.x_axis, self.y_axis);
        let tmp1 = i32x4_shuffle::<2, 3, 6, 7>(self.x_axis, self.y_axis);
        let tmp2 = i32x4_shuffle::<0, 1, 4, 5>(self.z_axis, self.w_axis);
        let tmp3 = i32x4_shuffle::<2, 3, 6, 7>(self.z_axis, self.w_axis);

        Self {
            x_axis: i32x4_shuffle::<0, 2, 4, 6>(tmp0, tmp2),
            y_axis: i32x4_shuffle::<1, 3, 5, 7>(tmp0, tmp2),
            z_axis: i32x4_shuffle::<0, 2, 4, 6>(tmp1, tmp3),
            w_axis: i32x4_shuffle::<1, 3, 5, 7>(tmp1, tmp3),
        }
    }
}

impl FloatMatrix4x4<f32, f32x4> for Columns4<f32x4> {
    type SIMDVector3 = f32x4;

    fn inverse(&self) -> Self {
        // Based on https://github.com/g-truc/glm `glm_mat4_inverse`
        let fac0 = {
            let swp0a = i32x4_shuffle::<3, 3, 7, 7>(self.w_axis, self.z_axis);
            let swp0b = i32x4_shuffle::<2, 2, 6, 6>(self.w_axis, self.z_axis);

            let swp00 = i32x4_shuffle::<2, 2, 6, 6>(self.z_axis, self.y_axis);
            let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
            let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
            let swp03 = i32x4_shuffle::<3, 3, 7, 7>(self.z_axis, self.y_axis);

            let mul00 = f32x4_mul(swp00, swp01);
            let mul01 = f32x4_mul(swp02, swp03);
            f32x4_sub(mul00, mul01)
        };
        let fac1 = {
            let swp0a = i32x4_shuffle::<3, 3, 7, 7>(self.w_axis, self.z_axis);
            let swp0b = i32x4_shuffle::<1, 1, 5, 5>(self.w_axis, self.z_axis);

            let swp00 = i32x4_shuffle::<1, 1, 5, 5>(self.z_axis, self.y_axis);
            let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
            let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
            let swp03 = i32x4_shuffle::<3, 3, 7, 7>(self.z_axis, self.y_axis);

            let mul00 = f32x4_mul(swp00, swp01);
            let mul01 = f32x4_mul(swp02, swp03);
            f32x4_sub(mul00, mul01)
        };
        let fac2 = {
            let swp0a = i32x4_shuffle::<2, 2, 6, 6>(self.w_axis, self.z_axis);
            let swp0b = i32x4_shuffle::<1, 1, 5, 5>(self.w_axis, self.z_axis);

            let swp00 = i32x4_shuffle::<1, 1, 5, 5>(self.z_axis, self.y_axis);
            let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
            let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
            let swp03 = i32x4_shuffle::<2, 2, 6, 6>(self.z_axis, self.y_axis);

            let mul00 = f32x4_mul(swp00, swp01);
            let mul01 = f32x4_mul(swp02, swp03);
            f32x4_sub(mul00, mul01)
        };
        let fac3 = {
            let swp0a = i32x4_shuffle::<3, 3, 7, 7>(self.w_axis, self.z_axis);
            let swp0b = i32x4_shuffle::<0, 0, 4, 4>(self.w_axis, self.z_axis);

            let swp00 = i32x4_shuffle::<0, 0, 4, 4>(self.z_axis, self.y_axis);
            let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
            let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
            let swp03 = i32x4_shuffle::<3, 3, 7, 7>(self.z_axis, self.y_axis);

            let mul00 = f32x4_mul(swp00, swp01);
            let mul01 = f32x4_mul(swp02, swp03);
            f32x4_sub(mul00, mul01)
        };
        let fac4 = {
            let swp0a = i32x4_shuffle::<2, 2, 6, 6>(self.w_axis, self.z_axis);
            let swp0b = i32x4_shuffle::<0, 0, 4, 4>(self.w_axis, self.z_axis);

            let swp00 = i32x4_shuffle::<0, 0, 4, 4>(self.z_axis, self.y_axis);
            let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
            let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
            let swp03 = i32x4_shuffle::<2, 2, 6, 6>(self.z_axis, self.y_axis);

            let mul00 = f32x4_mul(swp00, swp01);
            let mul01 = f32x4_mul(swp02, swp03);
            f32x4_sub(mul00, mul01)
        };
        let fac5 = {
            let swp0a = i32x4_shuffle::<1, 1, 5, 5>(self.w_axis, self.z_axis);
            let swp0b = i32x4_shuffle::<0, 0, 4, 4>(self.w_axis, self.z_axis);

            let swp00 = i32x4_shuffle::<0, 0, 4, 4>(self.z_axis, self.y_axis);
            let swp01 = i32x4_shuffle::<0, 0, 4, 6>(swp0a, swp0a);
            let swp02 = i32x4_shuffle::<0, 0, 4, 6>(swp0b, swp0b);
            let swp03 = i32x4_shuffle::<1, 1, 5, 5>(self.z_axis, self.y_axis);

            let mul00 = f32x4_mul(swp00, swp01);
            let mul01 = f32x4_mul(swp02, swp03);
            f32x4_sub(mul00, mul01)
        };
        let sign_a = f32x4(-1.0, 1.0, -1.0, 1.0);
        let sign_b = f32x4(1.0, -1.0, 1.0, -1.0);

        let temp0 = i32x4_shuffle::<0, 0, 4, 4>(self.y_axis, self.x_axis);
        let vec0 = i32x4_shuffle::<0, 2, 6, 6>(temp0, temp0);

        let temp1 = i32x4_shuffle::<1, 1, 5, 5>(self.y_axis, self.x_axis);
        let vec1 = i32x4_shuffle::<0, 2, 6, 6>(temp1, temp1);

        let temp2 = i32x4_shuffle::<2, 2, 6, 6>(self.y_axis, self.x_axis);
        let vec2 = i32x4_shuffle::<0, 2, 6, 6>(temp2, temp2);

        let temp3 = i32x4_shuffle::<3, 3, 7, 7>(self.y_axis, self.x_axis);
        let vec3 = i32x4_shuffle::<0, 2, 6, 6>(temp3, temp3);

        let mul00 = f32x4_mul(vec1, fac0);
        let mul01 = f32x4_mul(vec2, fac1);
        let mul02 = f32x4_mul(vec3, fac2);
        let sub00 = f32x4_sub(mul00, mul01);
        let add00 = f32x4_add(sub00, mul02);
        let inv0 = f32x4_mul(sign_b, add00);

        let mul03 = f32x4_mul(vec0, fac0);
        let mul04 = f32x4_mul(vec2, fac3);
        let mul05 = f32x4_mul(vec3, fac4);
        let sub01 = f32x4_sub(mul03, mul04);
        let add01 = f32x4_add(sub01, mul05);
        let inv1 = f32x4_mul(sign_a, add01);

        let mul06 = f32x4_mul(vec0, fac1);
        let mul07 = f32x4_mul(vec1, fac3);
        let mul08 = f32x4_mul(vec3, fac5);
        let sub02 = f32x4_sub(mul06, mul07);
        let add02 = f32x4_add(sub02, mul08);
        let inv2 = f32x4_mul(sign_b, add02);

        let mul09 = f32x4_mul(vec0, fac2);
        let mul10 = f32x4_mul(vec1, fac4);
        let mul11 = f32x4_mul(vec2, fac5);
        let sub03 = f32x4_sub(mul09, mul10);
        let add03 = f32x4_add(sub03, mul11);
        let inv3 = f32x4_mul(sign_a, add03);

        let row0 = i32x4_shuffle::<0, 0, 4, 4>(inv0, inv1);
        let row1 = i32x4_shuffle::<0, 0, 4, 4>(inv2, inv3);
        let row2 = i32x4_shuffle::<0, 2, 4, 6>(row0, row1);

        let dot0 = Vector4::dot(self.x_axis, row2);
        glam_assert!(dot0 != 0.0);

        let rcp0 = f32x4_splat(dot0.recip());

        Self {
            x_axis: f32x4_mul(inv0, rcp0),
            y_axis: f32x4_mul(inv1, rcp0),
            z_axis: f32x4_mul(inv2, rcp0),
            w_axis: f32x4_mul(inv3, rcp0),
        }
    }

    #[inline(always)]
    fn transform_point3(&self, other: XYZ<f32>) -> XYZ<f32> {
        self.x_axis
            .mul_scalar(other.x)
            .add(self.y_axis.mul_scalar(other.y))
            .add(self.z_axis.mul_scalar(other.z))
            .add(self.w_axis)
            .into()
    }

    #[inline(always)]
    fn transform_vector3(&self, other: XYZ<f32>) -> XYZ<f32> {
        self.x_axis
            .mul_scalar(other.x)
            .add(self.y_axis.mul_scalar(other.y))
            .add(self.z_axis.mul_scalar(other.z))
            .into()
    }

    #[inline]
    fn transform_float4_as_point3(&self, other: f32x4) -> f32x4 {
        let mut res = self.x_axis.mul(Vector4::splat_x(other));
        res = self.y_axis.mul_add(Vector4::splat_y(other), res);
        res = self.z_axis.mul_add(Vector4::splat_z(other), res);
        res = self.w_axis.add(res);
        res
    }

    #[inline]
    fn transform_float4_as_vector3(&self, other: f32x4) -> f32x4 {
        let mut res = self.x_axis.mul(Vector4::splat_x(other));
        res = self.y_axis.mul_add(Vector4::splat_y(other), res);
        res = self.z_axis.mul_add(Vector4::splat_z(other), res);
        res
    }

    #[inline]
    fn project_float4_as_point3(&self, other: f32x4) -> f32x4 {
        let mut res = self.x_axis.mul(Vector4::splat_x(other));
        res = self.y_axis.mul_add(Vector4::splat_y(other), res);
        res = self.z_axis.mul_add(Vector4::splat_z(other), res);
        res = self.w_axis.add(res);
        res = res.mul(res.splat_w().recip());
        res
    }
}

impl ProjectionMatrix<f32, f32x4> for Columns4<f32x4> {}

impl From<Columns3<XYZ<f32>>> for Columns3<f32x4> {
    #[inline(always)]
    fn from(v: Columns3<XYZ<f32>>) -> Columns3<f32x4> {
        Self {
            x_axis: v.x_axis.into(),
            y_axis: v.y_axis.into(),
            z_axis: v.z_axis.into(),
        }
    }
}

impl From<Columns3<f32x4>> for Columns3<XYZ<f32>> {
    #[inline(always)]
    fn from(v: Columns3<f32x4>) -> Columns3<XYZ<f32>> {
        Self {
            x_axis: v.x_axis.into(),
            y_axis: v.y_axis.into(),
            z_axis: v.z_axis.into(),
        }
    }
}
