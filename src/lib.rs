use std::arch::aarch64::{float32x4_t, float32x4x2_t, float64x2_t};
use pulp::{aarch64::Neon, aarch64::NeonFcma, f32x4, f64x2, Simd};
use rlst::{c32, c64, RlstScalar};


pub fn matvec4x4_row_major<U>(matrix: &[U], vector: &[U], save_buffer: &mut [U], alpha: U)
where
    U: RlstScalar,
{
    let s1 = vector[0];
    let s2 = vector[1];
    let s3 = vector[2];
    let s4 = vector[3];

    for i in 0..4 {
        let mut sum = U::zero();
        sum += matrix[i * 4] * s1;
        sum += matrix[i * 4 + 1] * s2;
        sum += matrix[i * 4 + 2] * s3;
        sum += matrix[i * 4 + 3] * s4;
        save_buffer[i] += sum * alpha
    }
}

pub fn matvec8x8_row_major<U>(matrix: &[U], vector: &[U], save_buffer: &mut [U], alpha: U)
where
    U: RlstScalar,
{
    let s1 = vector[0];
    let s2 = vector[1];
    let s3 = vector[2];
    let s4 = vector[3];
    let s5 = vector[4];
    let s6 = vector[5];
    let s7 = vector[6];
    let s8 = vector[7];

    for i in 0..8 {
        let mut sum = U::zero();
        sum += matrix[i * 4] * s1;
        sum += matrix[i * 4 + 1] * s2;
        sum += matrix[i * 4 + 2] * s3;
        sum += matrix[i * 4 + 3] * s4;
        sum += matrix[i * 4 + 4] * s5;
        sum += matrix[i * 4 + 5] * s6;
        sum += matrix[i * 4 + 6] * s7;
        sum += matrix[i * 4 + 7] * s8;
        save_buffer[i] += sum * alpha
    }
}

pub fn matvec4x4_col_major<U>(matrix: &[U], vector: &[U], save_buffer: &mut [U], alpha: U)
where
    U: RlstScalar,
{
    for i in 0..4 {
        // cols
        for j in 0..4 {
            // rows
            save_buffer[j] += matrix[i * 4 + j] * vector[j]
        }
    }

    for i in 0..4 {
        save_buffer[i] *= alpha
    }
}

pub fn matvec8x8_col_major<U>(matrix: &[U], vector: &[U], save_buffer: &mut [U], alpha: U)
where
    U: RlstScalar,
{
    for i in 0..8 {
        // cols
        for j in 0..8 {
            // rows
            save_buffer[j] += matrix[i * 8 + j] * vector[j]
        }
    }

    for i in 0..8 {
        save_buffer[i] *= alpha
    }
}

#[inline]
fn fma<T: 'static>(x: T, y: T, z: T) -> T {
    use coe::coerce_static as to;
    if coe::is_same::<T, f32>() {
        to(f32::mul_add(to(x), to(y), to(z)))
    } else if coe::is_same::<T, f64>() {
        to(f64::mul_add(to(x), to(y), to(z)))
    } else {
        panic!()
    }
}

pub struct ComplexMul4x4Neon32<'a> {
    pub simd: Neon,
    pub alpha: f32,
    pub matrix: &'a [[c32; 4]; 4],
    pub vector: &'a [c32; 4],
    pub result: &'a mut [c32; 4],
}

pub struct ComplexMul4x4NeonFcma32<'a> {
    pub simd: NeonFcma,
    pub alpha: f32,
    pub matrix: &'a [[c32; 4]; 4],
    pub vector: &'a [c32; 4],
    pub result: &'a mut [c32; 4],
}

pub struct ComplexMul8x8NeonFcma32<'a> {
    pub simd: NeonFcma,
    pub alpha: f32,
    pub matrix: &'a [c32; 64],
    pub vector: &'a [c32; 8],
    pub result: &'a mut [c32; 8],
}

pub struct ComplexMul8x8NeonFcma64<'a> {
    pub simd: NeonFcma,
    pub alpha: f64,
    pub matrix: &'a [c64; 64],
    pub vector: &'a [c64; 8],
    pub result: &'a mut [c64; 8],
}

// pub struct ComplexMul8x8Avx232<'a> {
//     pub simd: NeonFcma,
//     pub alpha: f32,
//     pub matrix: &'a [c32; 64],
//     pub vector: &'a [c32; 8],
//     pub result: &'a mut [c32; 8],
// }

// pub struct ComplexMul8x8Avx264<'a> {
//     pub simd: NeonFcma,
//     pub alpha: f32,
//     pub matrix: &'a [c32; 64],
//     pub vector: &'a [c32; 8],
//     pub result: &'a mut [c32; 8],
// }

impl pulp::NullaryFnOnce for ComplexMul4x4Neon32<'_> {
    type Output = ();

    #[inline(always)]
    fn call(self) -> Self::Output {
        let Self {
            simd,
            alpha,
            matrix,
            vector,
            result,
        } = self;

        let mut acc_im = f32x4(0., 0., 0., 0.);
        let mut acc_re = f32x4(0., 0., 0., 0.);

        let ptr = vector.as_ptr() as *const f32;
        let float32x4x2_t(vec_re, vec_im) = unsafe { simd.neon.vld2q_f32(ptr) };
        let c: f32x4 = unsafe { std::mem::transmute(vec_re) };
        let d: f32x4 = unsafe { std::mem::transmute(vec_im) };

        for column in matrix.iter() {
            let ptr = column.as_ptr() as *const f32;
            let float32x4x2_t(row_re, row_im) = unsafe { simd.neon.vld2q_f32(ptr) };
            let a: f32x4 = unsafe { std::mem::transmute(row_re) };
            let b: f32x4 = unsafe { std::mem::transmute(row_im) };

            let ac = simd.f32s_neg(simd.mul_f32x4(a, c));
            let re = simd.f32s_neg(simd.mul_add_f32x4(b, d, ac));
            acc_re = simd.add_f32x4(acc_re, re);

            let ad = simd.mul_f32x4(a, d);
            let im = simd.mul_add_f32x4(b, c, ad);
            acc_im = simd.add_f32x4(acc_im, im);
        }

        let acc_re: float32x4_t = unsafe { std::mem::transmute(acc_re) };
        let acc_im: float32x4_t = unsafe { std::mem::transmute(acc_im) };

        let acc_re = simd.neon.vmulq_n_f32(acc_re, alpha);
        let acc_im = simd.neon.vmulq_n_f32(acc_im, alpha);

        let res = float32x4x2_t(acc_re, acc_im);
        let ptr = result.as_ptr() as *mut f32;
        unsafe { simd.neon.vst2q_f32(ptr, res) };
    }
}

impl pulp::NullaryFnOnce for ComplexMul4x4NeonFcma32<'_> {
    type Output = ();

    #[inline(always)]
    fn call(self) -> Self::Output {
        let Self {
            simd,
            alpha,
            matrix,
            vector,
            result,
        } = self;

        let mut a1 = f32x4(0., 0., 0., 0.);
        let mut a2 = f32x4(0., 0., 0., 0.);

        let [v1, v2]: [f32x4; 2] = pulp::cast(*vector);

        // Unroll loop
        let [m1, m2]: [f32x4; 2] = pulp::cast(*&matrix[0]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);

        let [m1, m2]: [f32x4; 2] = pulp::cast(*&matrix[1]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);

        let [m1, m2]: [f32x4; 2] = pulp::cast(*&matrix[2]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);

        let [m1, m2]: [f32x4; 2] = pulp::cast(*&matrix[3]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);

        let a1: float32x4_t = unsafe { std::mem::transmute(a1) };
        let a2: float32x4_t = unsafe { std::mem::transmute(a2) };

        let a1 = simd.neon.vmulq_n_f32(a1, alpha);
        let a2 = simd.neon.vmulq_n_f32(a2, alpha);

        let ptr = result.as_ptr() as *mut f32;
        unsafe { simd.neon.vst1q_f32(ptr, a1) };
        unsafe { simd.neon.vst1q_f32(ptr.add(4), a2) };
    }
}

impl pulp::NullaryFnOnce for ComplexMul8x8NeonFcma32<'_> {
    type Output = ();

    #[inline(always)]
    fn call(self) -> Self::Output {
        let Self {
            simd,
            alpha,
            matrix,
            vector,
            result,
        } = self;

        let mut a1 = f32x4(0., 0., 0., 0.);
        let mut a2 = f32x4(0., 0., 0., 0.);
        let mut a3 = f32x4(0., 0., 0., 0.);
        let mut a4 = f32x4(0., 0., 0., 0.);

        let (matrix, _) = pulp::as_arrays::<8, _>(matrix);
        let [v1, v2, v3, v4]: [f32x4; 4] = pulp::cast(*vector);

        // Unroll loop
        let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[0]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);
        a3 = simd.c32s_mul_add_e(m3, v3, a3);
        a4 = simd.c32s_mul_add_e(m4, v4, a4);

        let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[1]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);
        a3 = simd.c32s_mul_add_e(m3, v3, a3);
        a4 = simd.c32s_mul_add_e(m4, v4, a4);

        let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[2]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);
        a3 = simd.c32s_mul_add_e(m3, v3, a3);
        a4 = simd.c32s_mul_add_e(m4, v4, a4);

        let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[3]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);
        a3 = simd.c32s_mul_add_e(m3, v3, a3);
        a4 = simd.c32s_mul_add_e(m4, v4, a4);

        let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[4]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);
        a3 = simd.c32s_mul_add_e(m3, v3, a3);
        a4 = simd.c32s_mul_add_e(m4, v4, a4);

        let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[5]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);
        a3 = simd.c32s_mul_add_e(m3, v3, a3);
        a4 = simd.c32s_mul_add_e(m4, v4, a4);

        let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[6]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);
        a3 = simd.c32s_mul_add_e(m3, v3, a3);
        a4 = simd.c32s_mul_add_e(m4, v4, a4);

        let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[7]);
        a1 = simd.c32s_mul_add_e(m1, v1, a1);
        a2 = simd.c32s_mul_add_e(m2, v2, a2);
        a3 = simd.c32s_mul_add_e(m3, v3, a3);
        a4 = simd.c32s_mul_add_e(m4, v4, a4);

        let a1: float32x4_t = unsafe { std::mem::transmute(a1) };
        let a2: float32x4_t = unsafe { std::mem::transmute(a2) };
        let a3: float32x4_t = unsafe { std::mem::transmute(a3) };
        let a4: float32x4_t = unsafe { std::mem::transmute(a4) };

        let a1 = simd.neon.vmulq_n_f32(a1, alpha);
        let a2 = simd.neon.vmulq_n_f32(a2, alpha);
        let a3 = simd.neon.vmulq_n_f32(a3, alpha);
        let a4 = simd.neon.vmulq_n_f32(a4, alpha);

        let ptr = result.as_ptr() as *mut f32;
        unsafe { simd.neon.vst1q_f32(ptr, a1) };
        unsafe { simd.neon.vst1q_f32(ptr.add(4), a2) };
        unsafe { simd.neon.vst1q_f32(ptr.add(8), a3) };
        unsafe { simd.neon.vst1q_f32(ptr.add(12), a4) };
    }
}

impl pulp::NullaryFnOnce for ComplexMul8x8NeonFcma64<'_> {
    type Output = ();

    #[inline(always)]
    fn call(self) -> Self::Output {
        let Self {
            simd,
            alpha,
            matrix,
            vector,
            result,
        } = self;

        let mut a1 = f64x2(0., 0.);
        let mut a2 = f64x2(0., 0.);
        let mut a3 = f64x2(0., 0.);
        let mut a4 = f64x2(0., 0.);
        let mut a5 = f64x2(0., 0.);
        let mut a6 = f64x2(0., 0.);
        let mut a7 = f64x2(0., 0.);
        let mut a8 = f64x2(0., 0.);

        let (matrix, _) = pulp::as_arrays::<8, _>(matrix);
        let [v1, v2, v3, v4, v5, v6, v7, v8]: [f64x2; 8] = pulp::cast(*vector);

        // Unroll loop
        let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[0]);
        a1 = simd.c64s_mul_add_e(m1, v1, a1);
        a2 = simd.c64s_mul_add_e(m2, v2, a2);
        a3 = simd.c64s_mul_add_e(m3, v3, a3);
        a4 = simd.c64s_mul_add_e(m4, v4, a4);
        a5 = simd.c64s_mul_add_e(m5, v5, a5);
        a6 = simd.c64s_mul_add_e(m6, v6, a6);
        a7 = simd.c64s_mul_add_e(m7, v7, a7);
        a8 = simd.c64s_mul_add_e(m8, v8, a8);

        let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[1]);
        a1 = simd.c64s_mul_add_e(m1, v1, a1);
        a2 = simd.c64s_mul_add_e(m2, v2, a2);
        a3 = simd.c64s_mul_add_e(m3, v3, a3);
        a4 = simd.c64s_mul_add_e(m4, v4, a4);
        a5 = simd.c64s_mul_add_e(m5, v5, a5);
        a6 = simd.c64s_mul_add_e(m6, v6, a6);
        a7 = simd.c64s_mul_add_e(m7, v7, a7);
        a8 = simd.c64s_mul_add_e(m8, v8, a8);

        let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[2]);
        a1 = simd.c64s_mul_add_e(m1, v1, a1);
        a2 = simd.c64s_mul_add_e(m2, v2, a2);
        a3 = simd.c64s_mul_add_e(m3, v3, a3);
        a4 = simd.c64s_mul_add_e(m4, v4, a4);
        a5 = simd.c64s_mul_add_e(m5, v5, a5);
        a6 = simd.c64s_mul_add_e(m6, v6, a6);
        a7 = simd.c64s_mul_add_e(m7, v7, a7);
        a8 = simd.c64s_mul_add_e(m8, v8, a8);

        let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[3]);
        a1 = simd.c64s_mul_add_e(m1, v1, a1);
        a2 = simd.c64s_mul_add_e(m2, v2, a2);
        a3 = simd.c64s_mul_add_e(m3, v3, a3);
        a4 = simd.c64s_mul_add_e(m4, v4, a4);
        a5 = simd.c64s_mul_add_e(m5, v5, a5);
        a6 = simd.c64s_mul_add_e(m6, v6, a6);
        a7 = simd.c64s_mul_add_e(m7, v7, a7);
        a8 = simd.c64s_mul_add_e(m8, v8, a8);

        let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[4]);
        a1 = simd.c64s_mul_add_e(m1, v1, a1);
        a2 = simd.c64s_mul_add_e(m2, v2, a2);
        a3 = simd.c64s_mul_add_e(m3, v3, a3);
        a4 = simd.c64s_mul_add_e(m4, v4, a4);
        a5 = simd.c64s_mul_add_e(m5, v5, a5);
        a6 = simd.c64s_mul_add_e(m6, v6, a6);
        a7 = simd.c64s_mul_add_e(m7, v7, a7);
        a8 = simd.c64s_mul_add_e(m8, v8, a8);

        let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[5]);
        a1 = simd.c64s_mul_add_e(m1, v1, a1);
        a2 = simd.c64s_mul_add_e(m2, v2, a2);
        a3 = simd.c64s_mul_add_e(m3, v3, a3);
        a4 = simd.c64s_mul_add_e(m4, v4, a4);
        a5 = simd.c64s_mul_add_e(m5, v5, a5);
        a6 = simd.c64s_mul_add_e(m6, v6, a6);
        a7 = simd.c64s_mul_add_e(m7, v7, a7);
        a8 = simd.c64s_mul_add_e(m8, v8, a8);

        let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[6]);
        a1 = simd.c64s_mul_add_e(m1, v1, a1);
        a2 = simd.c64s_mul_add_e(m2, v2, a2);
        a3 = simd.c64s_mul_add_e(m3, v3, a3);
        a4 = simd.c64s_mul_add_e(m4, v4, a4);
        a5 = simd.c64s_mul_add_e(m5, v5, a5);
        a6 = simd.c64s_mul_add_e(m6, v6, a6);
        a7 = simd.c64s_mul_add_e(m7, v7, a7);
        a8 = simd.c64s_mul_add_e(m8, v8, a8);

        let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[7]);
        a1 = simd.c64s_mul_add_e(m1, v1, a1);
        a2 = simd.c64s_mul_add_e(m2, v2, a2);
        a3 = simd.c64s_mul_add_e(m3, v3, a3);
        a4 = simd.c64s_mul_add_e(m4, v4, a4);
        a5 = simd.c64s_mul_add_e(m5, v5, a5);
        a6 = simd.c64s_mul_add_e(m6, v6, a6);
        a7 = simd.c64s_mul_add_e(m7, v7, a7);
        a8 = simd.c64s_mul_add_e(m8, v8, a8);

        let a1: float64x2_t = unsafe { std::mem::transmute(a1) };
        let a2: float64x2_t = unsafe { std::mem::transmute(a2) };
        let a3: float64x2_t = unsafe { std::mem::transmute(a3) };
        let a4: float64x2_t = unsafe { std::mem::transmute(a4) };
        let a5: float64x2_t = unsafe { std::mem::transmute(a5) };
        let a6: float64x2_t = unsafe { std::mem::transmute(a6) };
        let a7: float64x2_t = unsafe { std::mem::transmute(a7) };
        let a8: float64x2_t = unsafe { std::mem::transmute(a8) };

        let a1 = simd.neon.vmulq_n_f64(a1, alpha);
        let a2 = simd.neon.vmulq_n_f64(a2, alpha);
        let a3 = simd.neon.vmulq_n_f64(a3, alpha);
        let a4 = simd.neon.vmulq_n_f64(a4, alpha);
        let a5 = simd.neon.vmulq_n_f64(a5, alpha);
        let a6 = simd.neon.vmulq_n_f64(a6, alpha);
        let a7 = simd.neon.vmulq_n_f64(a7, alpha);
        let a8 = simd.neon.vmulq_n_f64(a8, alpha);

        let ptr = result.as_ptr() as *mut f64;
        unsafe { simd.neon.vst1q_f64(ptr, a1) };
        unsafe { simd.neon.vst1q_f64(ptr.add(2), a2) };
        unsafe { simd.neon.vst1q_f64(ptr.add(4), a3) };
        unsafe { simd.neon.vst1q_f64(ptr.add(6), a4) };
        unsafe { simd.neon.vst1q_f64(ptr.add(8), a5) };
        unsafe { simd.neon.vst1q_f64(ptr.add(10), a6) };
        unsafe { simd.neon.vst1q_f64(ptr.add(12), a7) };
        unsafe { simd.neon.vst1q_f64(ptr.add(14), a8) };
    }
}


#[cfg(test)]
mod test {

    use super::*;
    use num_traits::*;

    #[test]
    fn test_8x8_f64() {
        let mut expected = [c64::zero(); 8];

        let mut matrix = [c64::zero(); 64];
        let mut vector = [c64::zero(); 8];
        let alpha = c64::one() * 12.;

        for i in 0..8 {
            let num = (i + 1) as f64;
            vector[i] = c64::new(num, num);
            for j in 0..8 {
                matrix[i * 8 + j] = c64::new((i + 1) as f64, (j + 1) as f64);
            }
        }

        matvec8x8_col_major(&matrix, &vector, &mut expected, alpha);

        let mut matrix = [c64::zero(); 64];
        let mut vector = [c64::zero(); 8];
        let mut result = [c64::zero(); 8];

        for i in 0..8 {
            let num = (i + 1) as f64;
            vector[i] = c64::new(num, num);
            for j in 0..8 {
                matrix[i*8 + j] = c64::new((i + 1) as f64, (j + 1) as f64);
            }
        }

        let simd = NeonFcma::try_new().unwrap();
        simd.vectorize(ComplexMul8x8NeonFcma64 {
            simd,
            alpha: alpha.re(),
            matrix: &matrix,
            vector: &vector,
            result: &mut result,
        });

        expected.iter().zip(result).for_each(|(e, r)| {
            println!("e {:?} r {:?}", e, r);
            assert!((e - r).abs() < 1e-10)
        });
    }

    #[test]
    fn test_8x8_f32() {
        let mut expected = [c32::zero(); 8];

        let mut matrix = [c32::zero(); 64];
        let mut vector = [c32::zero(); 8];
        let alpha = c32::one() * 10.;

        for i in 0..8 {
            let num = (i + 1) as f32;
            vector[i] = c32::new(num, num);
            for j in 0..8 {
                matrix[i * 8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
            }
        }

        matvec8x8_col_major(&matrix, &vector, &mut expected, alpha);

        let mut matrix = [c32::zero(); 64];
        let mut vector = [c32::zero(); 8];
        let mut result = [c32::zero(); 8];

        for i in 0..8 {
            let num = (i + 1) as f32;
            vector[i] = c32::new(num, num);
            for j in 0..8 {
                matrix[i*8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
            }
        }

        let simd = NeonFcma::try_new().unwrap();
        simd.vectorize(ComplexMul8x8NeonFcma32 {
            simd,
            alpha: alpha.re(),
            matrix: &matrix,
            vector: &vector,
            result: &mut result,
        });

        expected.iter().zip(result).for_each(|(e, r)| {
            println!("e {:?} r {:?}", e, r);
            assert!((e - r).abs() < 1e-10)
        });
    }
}
