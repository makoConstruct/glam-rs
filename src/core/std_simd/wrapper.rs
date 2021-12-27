use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Sub};
use std::simd::{f32x4, mask32x4};

#[inline(always)]
pub(crate) fn f32x4_add(a: f32x4, b: f32x4) -> f32x4 {
    a.add(b)
}

#[inline(always)]
pub(crate) fn f32x4_div(a: f32x4, b: f32x4) -> f32x4 {
    a.div(b)
}

#[inline(always)]
pub(crate) fn f32x4_mul(a: f32x4, b: f32x4) -> f32x4 {
    a.mul(b)
}

#[inline(always)]
pub(crate) fn f32x4_rem(a: f32x4, b: f32x4) -> f32x4 {
    a.rem(b)
}

#[inline(always)]
pub(crate) fn f32x4_sub(a: f32x4, b: f32x4) -> f32x4 {
    a.sub(b)
}

#[inline(always)]
pub(crate) fn f32x4_neg(a: f32x4) -> f32x4 {
    a.neg()
}

#[inline(always)]
pub(crate) fn f32x4_mul_add(a: f32x4, b: f32x4, c: f32x4) -> f32x4 {
    f32x4::mul_add(a, b, c)
}

#[inline(always)]
pub(crate) fn f32x4_min(a: f32x4, b: f32x4) -> f32x4 {
    a.min(b)
}

#[inline(always)]
pub(crate) fn f32x4_max(a: f32x4, b: f32x4) -> f32x4 {
    a.max(b)
}

#[inline(always)]
pub(crate) fn f32x4_abs(a: f32x4) -> f32x4 {
    a.abs()
}

#[inline(always)]
pub(crate) fn f32x4_signum(a: f32x4) -> f32x4 {
    a.signum()
}

#[inline(always)]
pub(crate) fn f32x4_floor(a: f32x4) -> f32x4 {
    f32x4::floor(a)
}

#[inline(always)]
pub(crate) fn f32x4_ceil(a: f32x4) -> f32x4 {
    a.ceil()
}

#[inline(always)]
pub(crate) fn f32x4_round(a: f32x4) -> f32x4 {
    a.round()
}

#[inline(always)]
pub(crate) fn f32x4_recip(a: f32x4) -> f32x4 {
    a.recip()
}

#[inline(always)]
pub(crate) fn f32x4_sqrt(a: f32x4) -> f32x4 {
    a.sqrt()
}

#[inline(always)]
pub(crate) fn f32x4_splat(a: f32) -> f32x4 {
    f32x4::splat(a)
}

#[inline(always)]
pub(crate) fn f32x4_bitand(a: f32x4, b: f32x4) -> f32x4 {
    let a = a.to_bits();
    let b = b.to_bits();
    f32x4::from_bits(a.bitand(b))
}

#[inline(always)]
pub(crate) fn f32x4_bitor(a: f32x4, b: f32x4) -> f32x4 {
    let a = a.to_bits();
    let b = b.to_bits();
    f32x4::from_bits(a.bitor(b))
}

#[inline(always)]
pub(crate) fn f32x4_bitxor(a: f32x4, b: f32x4) -> f32x4 {
    let a = a.to_bits();
    let b = b.to_bits();
    f32x4::from_bits(a.bitxor(b))
}

#[inline(always)]
pub(crate) fn mask32x4_to_bitmask(a: mask32x4) -> u32 {
    a.to_bitmask()[0] as u32
}

#[inline(always)]
pub(crate) fn mask32x4_bitand(a: mask32x4, b: mask32x4) -> mask32x4 {
    a.bitand(b)
}

#[inline(always)]
pub(crate) fn mask32x4_bitor(a: mask32x4, b: mask32x4) -> mask32x4 {
    a.bitor(b)
}

#[inline(always)]
pub(crate) fn mask32x4_bitxor(a: mask32x4, b: mask32x4) -> mask32x4 {
    a.bitxor(b)
}

#[inline(always)]
pub(crate) fn mask32x4_not(a: mask32x4) -> mask32x4 {
    a.not()
}

#[inline(always)]
pub(crate) fn mask32x4_any(a: mask32x4) -> bool {
    a.any()
}

#[inline(always)]
pub(crate) fn mask32x4_all(a: mask32x4) -> bool {
    a.all()
}
