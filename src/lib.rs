use rlst::{c32, c64, RlstScalar};

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    use super::*;
    use pulp::{aarch64::Neon, aarch64::NeonFcma, f32x4, f64x2, Simd};
    use std::arch::aarch64::{float32x4_t, float32x4x2_t, float64x2_t};

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
            let [r1, r2, r3, r4]: [f32x4; 4] = pulp::cast(*result);
            let alpha = simd.f32s_splat(alpha);

            let (matrix, _) = pulp::as_arrays::<8, _>(matrix);
            let [v1, v2, v3, v4]: [f32x4; 4] = pulp::cast(*vector);

            let v01 = f32x4(v1.0, v1.1, v1.0, v1.1);
            let v02 = f32x4(v1.2, v1.3, v1.2, v1.3);
            let v03 = f32x4(v2.0, v2.1, v2.0, v2.1);
            let v04 = f32x4(v2.2, v2.3, v2.2, v2.3);
            let v05 = f32x4(v3.0, v3.1, v3.0, v3.1);
            let v06 = f32x4(v3.2, v3.3, v3.2, v3.3);
            let v07 = f32x4(v4.0, v4.1, v4.0, v4.1);
            let v08 = f32x4(v4.2, v4.3, v4.2, v4.3);

            // Unroll loop
            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[0]);
            a1 = simd.c32s_mul_add_e(m1, v01, a1);
            a2 = simd.c32s_mul_add_e(m2, v01, a2);
            a3 = simd.c32s_mul_add_e(m3, v01, a3);
            a4 = simd.c32s_mul_add_e(m4, v01, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[1]);
            a1 = simd.c32s_mul_add_e(m1, v02, a1);
            a2 = simd.c32s_mul_add_e(m2, v02, a2);
            a3 = simd.c32s_mul_add_e(m3, v02, a3);
            a4 = simd.c32s_mul_add_e(m4, v02, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[2]);
            a1 = simd.c32s_mul_add_e(m1, v03, a1);
            a2 = simd.c32s_mul_add_e(m2, v03, a2);
            a3 = simd.c32s_mul_add_e(m3, v03, a3);
            a4 = simd.c32s_mul_add_e(m4, v03, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[3]);
            a1 = simd.c32s_mul_add_e(m1, v04, a1);
            a2 = simd.c32s_mul_add_e(m2, v04, a2);
            a3 = simd.c32s_mul_add_e(m3, v04, a3);
            a4 = simd.c32s_mul_add_e(m4, v04, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[4]);
            a1 = simd.c32s_mul_add_e(m1, v05, a1);
            a2 = simd.c32s_mul_add_e(m2, v05, a2);
            a3 = simd.c32s_mul_add_e(m3, v05, a3);
            a4 = simd.c32s_mul_add_e(m4, v05, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[5]);
            a1 = simd.c32s_mul_add_e(m1, v06, a1);
            a2 = simd.c32s_mul_add_e(m2, v06, a2);
            a3 = simd.c32s_mul_add_e(m3, v06, a3);
            a4 = simd.c32s_mul_add_e(m4, v06, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[6]);
            a1 = simd.c32s_mul_add_e(m1, v07, a1);
            a2 = simd.c32s_mul_add_e(m2, v07, a2);
            a3 = simd.c32s_mul_add_e(m3, v07, a3);
            a4 = simd.c32s_mul_add_e(m4, v07, a4);

            let [m1, m2, m3, m4]: [f32x4; 4] = pulp::cast(*&matrix[7]);
            a1 = simd.c32s_mul_add_e(m1, v08, a1);
            a2 = simd.c32s_mul_add_e(m2, v08, a2);
            a3 = simd.c32s_mul_add_e(m3, v08, a3);
            a4 = simd.c32s_mul_add_e(m4, v08, a4);

            a1 = simd.mul_add_f32x4(a1, alpha, r1);
            a2 = simd.mul_add_f32x4(a2, alpha, r2);
            a3 = simd.mul_add_f32x4(a3, alpha, r3);
            a4 = simd.mul_add_f32x4(a4, alpha, r4);

            let a1: float32x4_t = unsafe { std::mem::transmute(a1) };
            let a2: float32x4_t = unsafe { std::mem::transmute(a2) };
            let a3: float32x4_t = unsafe { std::mem::transmute(a3) };
            let a4: float32x4_t = unsafe { std::mem::transmute(a4) };

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

            let [r1, r2, r3, r4, r5, r6, r7, r8]: [f64x2; 8] = pulp::cast(*result);
            let alpha = simd.f64s_splat(alpha);

            let (matrix, _) = pulp::as_arrays::<8, _>(matrix);
            let [v1, v2, v3, v4, v5, v6, v7, v8]: [f64x2; 8] = pulp::cast(*vector);

            // Unroll loop
            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[0]);
            a1 = simd.c64s_mul_add_e(m1, v1, a1);
            a2 = simd.c64s_mul_add_e(m2, v1, a2);
            a3 = simd.c64s_mul_add_e(m3, v1, a3);
            a4 = simd.c64s_mul_add_e(m4, v1, a4);
            a5 = simd.c64s_mul_add_e(m5, v1, a5);
            a6 = simd.c64s_mul_add_e(m6, v1, a6);
            a7 = simd.c64s_mul_add_e(m7, v1, a7);
            a8 = simd.c64s_mul_add_e(m8, v1, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[1]);
            a1 = simd.c64s_mul_add_e(m1, v2, a1);
            a2 = simd.c64s_mul_add_e(m2, v2, a2);
            a3 = simd.c64s_mul_add_e(m3, v2, a3);
            a4 = simd.c64s_mul_add_e(m4, v2, a4);
            a5 = simd.c64s_mul_add_e(m5, v2, a5);
            a6 = simd.c64s_mul_add_e(m6, v2, a6);
            a7 = simd.c64s_mul_add_e(m7, v2, a7);
            a8 = simd.c64s_mul_add_e(m8, v2, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[2]);
            a1 = simd.c64s_mul_add_e(m1, v3, a1);
            a2 = simd.c64s_mul_add_e(m2, v3, a2);
            a3 = simd.c64s_mul_add_e(m3, v3, a3);
            a4 = simd.c64s_mul_add_e(m4, v3, a4);
            a5 = simd.c64s_mul_add_e(m5, v3, a5);
            a6 = simd.c64s_mul_add_e(m6, v3, a6);
            a7 = simd.c64s_mul_add_e(m7, v3, a7);
            a8 = simd.c64s_mul_add_e(m8, v3, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[3]);
            a1 = simd.c64s_mul_add_e(m1, v4, a1);
            a2 = simd.c64s_mul_add_e(m2, v4, a2);
            a3 = simd.c64s_mul_add_e(m3, v4, a3);
            a4 = simd.c64s_mul_add_e(m4, v4, a4);
            a5 = simd.c64s_mul_add_e(m5, v4, a5);
            a6 = simd.c64s_mul_add_e(m6, v4, a6);
            a7 = simd.c64s_mul_add_e(m7, v4, a7);
            a8 = simd.c64s_mul_add_e(m8, v4, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[4]);
            a1 = simd.c64s_mul_add_e(m1, v5, a1);
            a2 = simd.c64s_mul_add_e(m2, v5, a2);
            a3 = simd.c64s_mul_add_e(m3, v5, a3);
            a4 = simd.c64s_mul_add_e(m4, v5, a4);
            a5 = simd.c64s_mul_add_e(m5, v5, a5);
            a6 = simd.c64s_mul_add_e(m6, v5, a6);
            a7 = simd.c64s_mul_add_e(m7, v5, a7);
            a8 = simd.c64s_mul_add_e(m8, v5, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[5]);
            a1 = simd.c64s_mul_add_e(m1, v6, a1);
            a2 = simd.c64s_mul_add_e(m2, v6, a2);
            a3 = simd.c64s_mul_add_e(m3, v6, a3);
            a4 = simd.c64s_mul_add_e(m4, v6, a4);
            a5 = simd.c64s_mul_add_e(m5, v6, a5);
            a6 = simd.c64s_mul_add_e(m6, v6, a6);
            a7 = simd.c64s_mul_add_e(m7, v6, a7);
            a8 = simd.c64s_mul_add_e(m8, v6, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[6]);
            a1 = simd.c64s_mul_add_e(m1, v7, a1);
            a2 = simd.c64s_mul_add_e(m2, v7, a2);
            a3 = simd.c64s_mul_add_e(m3, v7, a3);
            a4 = simd.c64s_mul_add_e(m4, v7, a4);
            a5 = simd.c64s_mul_add_e(m5, v7, a5);
            a6 = simd.c64s_mul_add_e(m6, v7, a6);
            a7 = simd.c64s_mul_add_e(m7, v7, a7);
            a8 = simd.c64s_mul_add_e(m8, v7, a8);

            let [m1, m2, m3, m4, m5, m6, m7, m8]: [f64x2; 8] = pulp::cast(*&matrix[7]);
            a1 = simd.c64s_mul_add_e(m1, v8, a1);
            a2 = simd.c64s_mul_add_e(m2, v8, a2);
            a3 = simd.c64s_mul_add_e(m3, v8, a3);
            a4 = simd.c64s_mul_add_e(m4, v8, a4);
            a5 = simd.c64s_mul_add_e(m5, v8, a5);
            a6 = simd.c64s_mul_add_e(m6, v8, a6);
            a7 = simd.c64s_mul_add_e(m7, v8, a7);
            a8 = simd.c64s_mul_add_e(m8, v8, a8);

            a1 = simd.mul_add_f64x2(a1, alpha, r1);
            a2 = simd.mul_add_f64x2(a2, alpha, r2);
            a3 = simd.mul_add_f64x2(a3, alpha, r3);
            a4 = simd.mul_add_f64x2(a4, alpha, r4);
            a5 = simd.mul_add_f64x2(a5, alpha, r5);
            a6 = simd.mul_add_f64x2(a6, alpha, r6);
            a7 = simd.mul_add_f64x2(a7, alpha, r7);
            a8 = simd.mul_add_f64x2(a8, alpha, r8);

            let a1: float64x2_t = unsafe { std::mem::transmute(a1) };
            let a2: float64x2_t = unsafe { std::mem::transmute(a2) };
            let a3: float64x2_t = unsafe { std::mem::transmute(a3) };
            let a4: float64x2_t = unsafe { std::mem::transmute(a4) };
            let a5: float64x2_t = unsafe { std::mem::transmute(a5) };
            let a6: float64x2_t = unsafe { std::mem::transmute(a6) };
            let a7: float64x2_t = unsafe { std::mem::transmute(a7) };
            let a8: float64x2_t = unsafe { std::mem::transmute(a8) };

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
        fn test_8x8_f64_col_major() {
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
                    matrix[i * 8 + j] = c64::new((i + 1) as f64, (j + 1) as f64);
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
        fn test_8x8_f64_row_major() {
            let mut expected = [c64::zero(); 8];
            let mut expected_t = [c64::zero(); 8];

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

            let mut matrix_t = [c64::zero(); 64];
            for i in 0..8 {
                // rows
                for j in 0..8 {
                    // cols
                    matrix_t[j * 8 + i] = matrix[i * 8 + j];
                }
            }

            for i in 0..8 {
                for j in 0..8 {
                    print!("{:4} ", matrix[i + j * 8]);
                }
                println!();
            }

            println!("");

            for i in 0..8 {
                for j in 0..8 {
                    print!("{:4} ", matrix_t[i + j * 8]);
                }
                println!();
            }

            matvec8x8_col_major(&matrix, &vector, &mut expected, alpha);
            matvec8x8_row_major(&matrix_t, &vector, &mut expected_t, alpha);

            expected
                .iter()
                .zip(expected_t.iter())
                .for_each(|(e, e_t)| assert!((e - e_t).abs() < 1e-5));

            let mut matrix = [c64::zero(); 64];
            let mut vector = [c64::zero(); 8];
            let mut result = [c64::zero(); 8];

            for i in 0..8 {
                let num = (i + 1) as f64;
                vector[i] = c64::new(num, num);
                for j in 0..8 {
                    matrix[i * 8 + j] = c64::new((i + 1) as f64, (j + 1) as f64);
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
                    matrix[i * 8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
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

        #[test]
        fn test_8x8_f32_row_major() {
            let mut expected = [c32::zero(); 8];
            let mut expected_t = [c32::zero(); 8];

            let mut matrix = [c32::zero(); 64];
            let mut vector = [c32::zero(); 8];
            let alpha = c32::one() * 12.;

            for i in 0..8 {
                let num = (i + 1) as f32;
                vector[i] = c32::new(num, num);
                for j in 0..8 {
                    matrix[i * 8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
                }
            }

            let mut matrix_t = [c32::zero(); 64];
            for i in 0..8 {
                // rows
                for j in 0..8 {
                    // cols
                    matrix_t[j * 8 + i] = matrix[i * 8 + j];
                }
            }

            for i in 0..8 {
                for j in 0..8 {
                    print!("{:4} ", matrix[i + j * 8]);
                }
                println!();
            }

            println!("");

            for i in 0..8 {
                for j in 0..8 {
                    print!("{:4} ", matrix_t[i + j * 8]);
                }
                println!();
            }

            matvec8x8_col_major(&matrix, &vector, &mut expected, alpha);
            matvec8x8_row_major(&matrix_t, &vector, &mut expected_t, alpha);

            expected
                .iter()
                .zip(expected_t.iter())
                .for_each(|(e, e_t)| assert!((e - e_t).abs() < 1e-5));

            let mut matrix = [c32::zero(); 64];
            let mut vector = [c32::zero(); 8];
            let mut result = [c32::zero(); 8];

            for i in 0..8 {
                let num = (i + 1) as f32;
                vector[i] = c32::new(num, num);
                for j in 0..8 {
                    matrix[i * 8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
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
}

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    use std::arch::x86_64::{__m128, __m256};

    use super::*;

    use num_traits::Zero;
    use pulp::{f32x16, f32x4, f32x8, x86::V3, Simd};

    pub struct ComplexMul8x8Avx32<'a> {
        pub simd: V3,
        pub alpha: f32,
        pub matrix: &'a [c32; 64],
        pub vector: &'a [c32; 8],
        pub result: &'a mut [c32; 8],
    }

    pub struct ComplexMul8x8Avx64<'a> {
        pub simd: V3,
        pub alpha: f32,
        pub matrix: &'a [c32; 64],
        pub vector: &'a [c32; 8],
        pub result: &'a mut [c32; 8],
    }

    pub const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
        ((z << 6) | (y << 4) | (x << 2) | w) as i32
    }

    const MASK: i32 = _MM_SHUFFLE(1, 0, 1, 0);
    const MASK2: i32 = _MM_SHUFFLE(3, 2, 3, 2);

    impl pulp::NullaryFnOnce for ComplexMul8x8Avx32<'_> {
        type Output = ();

        fn call(self) -> Self::Output {
            let Self {
                simd,
                alpha,
                matrix,
                vector,
                result,
            } = self;

            // Accumulators
            let mut a1 = simd.avx._mm256_set1_ps(0.);
            let mut a2 = simd.avx._mm256_set1_ps(0.);

            // Read matrix into slices
            let (matrix, _) = pulp::as_arrays::<8, _>(matrix);

            // Negator
            let neg = simd.avx._mm256_set_ps(-1., 1., -1., 1., -1., 1., -1., 1.);

            // Load vector
            let [v1, v2, v3, v4]: [__m128; 4] = pulp::cast(*vector);

            // Consider vector[0] matrix[0]
            {
                let v01 = simd.sse._mm_shuffle_ps::<0b01000100>(v1, v1);
                let v01 = simd.avx._mm256_broadcast_ps(&v01); // fill with first imag number repeated
                let v01_s = simd
                    .avx
                    ._mm256_mul_ps(simd.avx._mm256_permute_ps::<0b10110001>(v01), neg); // switch real/imag parts, negate imaginary part

                let [m1, m2]: [__m256; 2] = pulp::cast(*&matrix[0]);

                // 1. Find the real part of mult
                let m1_re = simd.avx._mm256_mul_ps(m1, v01);
                let m2_re = simd.avx._mm256_mul_ps(m2, v01);

                // 2. Find imaginary part
                let m1_im = simd.avx._mm256_mul_ps(m1, v01_s);
                let m2_im = simd.avx._mm256_mul_ps(m2, v01_s);

                let r1 = simd.avx._mm256_hsub_ps(m1_re, m1_im);
                let r1 = simd.avx._mm256_permute_ps::<0b11011000>(r1);

                let r2 = simd.avx._mm256_hsub_ps(m2_re, m2_im);
                let r2 = simd.avx._mm256_permute_ps::<0b11011000>(r2);

                a1 = simd.avx._mm256_add_ps(a1, r1);
                a2 = simd.avx._mm256_add_ps(a2, r2);
            }

            // Consider vector[1] matrix[1]
            {
                let v02 = simd.sse._mm_shuffle_ps::<0b11101110>(v1, v1);
                let v02 = simd.avx._mm256_broadcast_ps(&v02); // fill with first imag number repeated
                let v02_s = simd
                    .avx
                    ._mm256_mul_ps(simd.avx._mm256_permute_ps::<0b10110001>(v02), neg); // switch real/imag parts, negate imaginary part

                let [m1, m2]: [__m256; 2] = pulp::cast(*&matrix[1]);

                // println!("v02 {:?} m1 {:?} m2 {:?}", v02, m1, m2);

                // 1. Find the real part of mult
                let m1_re = simd.avx._mm256_mul_ps(m1, v02);
                let m2_re = simd.avx._mm256_mul_ps(m2, v02);

                // 2. Find imaginary part
                let m1_im = simd.avx._mm256_mul_ps(m1, v02_s);
                let m2_im = simd.avx._mm256_mul_ps(m2, v02_s);

                let r1 = simd.avx._mm256_hsub_ps(m1_re, m1_im);
                let r1 = simd.avx._mm256_permute_ps::<0b11011000>(r1);

                let r2 = simd.avx._mm256_hsub_ps(m2_re, m2_im);
                let r2 = simd.avx._mm256_permute_ps::<0b11011000>(r2);

                a1 = simd.avx._mm256_add_ps(a1, r1);
                a2 = simd.avx._mm256_add_ps(a2, r2);
                // println!("r1 {:?} r2 {:?}", r1, r2);
            }

            // Consider vector[2] matrix[2]
            {
                let v03 = simd.sse._mm_shuffle_ps::<0b01000100>(v2, v2);
                let v03 = simd.avx._mm256_broadcast_ps(&v03); // fill with first imag number repeated
                let v03_s = simd
                    .avx
                    ._mm256_mul_ps(simd.avx._mm256_permute_ps::<0b10110001>(v03), neg); // switch real/imag parts, negate imaginary part

                let [m1, m2]: [__m256; 2] = pulp::cast(*&matrix[2]);

                // 1. Find the real part of mult
                let m1_re = simd.avx._mm256_mul_ps(m1, v03);
                let m2_re = simd.avx._mm256_mul_ps(m2, v03);

                // 2. Find imaginary part
                let m1_im = simd.avx._mm256_mul_ps(m1, v03_s);
                let m2_im = simd.avx._mm256_mul_ps(m2, v03_s);

                let r1 = simd.avx._mm256_hsub_ps(m1_re, m1_im);
                let r1 = simd.avx._mm256_permute_ps::<0b11011000>(r1);

                let r2 = simd.avx._mm256_hsub_ps(m2_re, m2_im);
                let r2 = simd.avx._mm256_permute_ps::<0b11011000>(r2);

                a1 = simd.avx._mm256_add_ps(a1, r1);
                a2 = simd.avx._mm256_add_ps(a2, r2);
            }

            // Consider vector[3] matrix[3]
            {
                let v04 = simd.sse._mm_shuffle_ps::<0b11101110>(v2, v2);
                let v04 = simd.avx._mm256_broadcast_ps(&v04); // fill with first imag number repeated
                let v04_s = simd
                    .avx
                    ._mm256_mul_ps(simd.avx._mm256_permute_ps::<0b10110001>(v04), neg); // switch real/imag parts, negate imaginary part

                let [m1, m2]: [__m256; 2] = pulp::cast(*&matrix[1]);

                // 1. Find the real part of mult
                let m1_re = simd.avx._mm256_mul_ps(m1, v04);
                let m2_re = simd.avx._mm256_mul_ps(m2, v04);

                // 2. Find imaginary part
                let m1_im = simd.avx._mm256_mul_ps(m1, v04_s);
                let m2_im = simd.avx._mm256_mul_ps(m2, v04_s);

                let r1 = simd.avx._mm256_hsub_ps(m1_re, m1_im);
                let r1 = simd.avx._mm256_permute_ps::<0b11011000>(r1);

                let r2 = simd.avx._mm256_hsub_ps(m2_re, m2_im);
                let r2 = simd.avx._mm256_permute_ps::<0b11011000>(r2);

                a1 = simd.avx._mm256_add_ps(a1, r1);
                a2 = simd.avx._mm256_add_ps(a2, r2);
            }

            // Consider vector[4] matrix[4]
            {
                let v05 = simd.sse._mm_shuffle_ps::<0b01000100>(v3, v3);
                let v05: __m256 = simd.avx._mm256_broadcast_ps(&v05); // fill with first imag number repeated
                let v05_s = simd
                    .avx
                    ._mm256_mul_ps(simd.avx._mm256_permute_ps::<0b10110001>(v05), neg); // switch real/imag parts, negate imaginary part

                let [m1, m2]: [__m256; 2] = pulp::cast(*&matrix[2]);

                // 1. Find the real part of mult
                let m1_re = simd.avx._mm256_mul_ps(m1, v05);
                let m2_re = simd.avx._mm256_mul_ps(m2, v05);

                // 2. Find imaginary part
                let m1_im = simd.avx._mm256_mul_ps(m1, v05_s);
                let m2_im = simd.avx._mm256_mul_ps(m2, v05_s);

                let r1 = simd.avx._mm256_hsub_ps(m1_re, m1_im);
                let r1 = simd.avx._mm256_permute_ps::<0b11011000>(r1);

                let r2 = simd.avx._mm256_hsub_ps(m2_re, m2_im);
                let r2 = simd.avx._mm256_permute_ps::<0b11011000>(r2);

                a1 = simd.avx._mm256_add_ps(a1, r1);
                a2 = simd.avx._mm256_add_ps(a2, r2);
            }

            // Consider vector[5] matrix[5]
            {
                let v06 = simd.sse._mm_shuffle_ps::<0b11101110>(v3, v3);
                let v06 = simd.avx._mm256_broadcast_ps(&v06); // fill with first imag number repeated
                let v06_s = simd
                    .avx
                    ._mm256_mul_ps(simd.avx._mm256_permute_ps::<0b10110001>(v06), neg); // switch real/imag parts, negate imaginary part

                let [m1, m2]: [__m256; 2] = pulp::cast(*&matrix[1]);

                // 1. Find the real part of mult
                let m1_re = simd.avx._mm256_mul_ps(m1, v06);
                let m2_re = simd.avx._mm256_mul_ps(m2, v06);

                // 2. Find imaginary part
                let m1_im = simd.avx._mm256_mul_ps(m1, v06_s);
                let m2_im = simd.avx._mm256_mul_ps(m2, v06_s);

                let r1 = simd.avx._mm256_hsub_ps(m1_re, m1_im);
                let r1 = simd.avx._mm256_permute_ps::<0b11011000>(r1);

                let r2 = simd.avx._mm256_hsub_ps(m2_re, m2_im);
                let r2 = simd.avx._mm256_permute_ps::<0b11011000>(r2);

                a1 = simd.avx._mm256_add_ps(a1, r1);
                a2 = simd.avx._mm256_add_ps(a2, r2);
            }

            // Consider vector[6] matrix[6]
            {
                let v07 = simd.sse._mm_shuffle_ps::<0b01000100>(v4, v4);
                let v07: __m256 = simd.avx._mm256_broadcast_ps(&v07); // fill with first imag number repeated
                let v07_s = simd
                    .avx
                    ._mm256_mul_ps(simd.avx._mm256_permute_ps::<0b10110001>(v07), neg); // switch real/imag parts, negate imaginary part

                let [m1, m2]: [__m256; 2] = pulp::cast(*&matrix[2]);

                // 1. Find the real part of mult
                let m1_re = simd.avx._mm256_mul_ps(m1, v07);
                let m2_re = simd.avx._mm256_mul_ps(m2, v07);

                // 2. Find imaginary part
                let m1_im = simd.avx._mm256_mul_ps(m1, v07_s);
                let m2_im = simd.avx._mm256_mul_ps(m2, v07_s);

                let r1 = simd.avx._mm256_hsub_ps(m1_re, m1_im);
                let r1 = simd.avx._mm256_permute_ps::<0b11011000>(r1);

                let r2 = simd.avx._mm256_hsub_ps(m2_re, m2_im);
                let r2 = simd.avx._mm256_permute_ps::<0b11011000>(r2);

                a1 = simd.avx._mm256_add_ps(a1, r1);
                a2 = simd.avx._mm256_add_ps(a2, r2);
            }

            // Consider vector[7] matrix[7]
            {
                let v08 = simd.sse._mm_shuffle_ps::<0b11101110>(v4, v4);
                let v08 = simd.avx._mm256_broadcast_ps(&v08); // fill with first imag number repeated
                let v08_s = simd
                    .avx
                    ._mm256_mul_ps(simd.avx._mm256_permute_ps::<0b10110001>(v08), neg); // switch real/imag parts, negate imaginary part

                let [m1, m2]: [__m256; 2] = pulp::cast(*&matrix[1]);

                // 1. Find the real part of mult
                let m1_re = simd.avx._mm256_mul_ps(m1, v08);
                let m2_re = simd.avx._mm256_mul_ps(m2, v08);

                // 2. Find imaginary part
                let m1_im = simd.avx._mm256_mul_ps(m1, v08_s);
                let m2_im = simd.avx._mm256_mul_ps(m2, v08_s);

                let r1 = simd.avx._mm256_hsub_ps(m1_re, m1_im);
                let r1 = simd.avx._mm256_permute_ps::<0b11011000>(r1);

                let r2 = simd.avx._mm256_hsub_ps(m2_re, m2_im);
                let r2 = simd.avx._mm256_permute_ps::<0b11011000>(r2);

                a1 = simd.avx._mm256_add_ps(a1, r1);
                a2 = simd.avx._mm256_add_ps(a2, r2);
            }

            let ptr = result.as_ptr() as *mut f32;
            unsafe { simd.avx._mm256_store_ps(ptr, a1) }
            unsafe { simd.avx._mm256_store_ps(ptr.add(8), a2) }
        }
    }

    #[cfg(test)]
    mod test {

        use super::*;
        use num_traits::{One, Zero};

        #[test]
        fn test_8x8_f32_row_major() {
            let mut expected = [c32::zero(); 8];
            let mut expected_t = [c32::zero(); 8];

            let mut matrix = [c32::zero(); 64];
            let mut vector = [c32::zero(); 8];
            let alpha = c32::one() * 12.;

            for i in 0..8 {
                let num = (i + 1) as f32;
                vector[i] = c32::new(num, -num);
                for j in 0..8 {
                    matrix[i * 8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
                }
            }

            let mut matrix_t = [c32::zero(); 64];
            for i in 0..8 {
                // rows
                for j in 0..8 {
                    // cols
                    matrix_t[j * 8 + i] = matrix[i * 8 + j];
                }
            }

            // for i in 0..8 {
            //     for j in 0..8 {
            //         print!("{:4} ", matrix[i + j * 8]);
            //     }
            //     println!();
            // }

            // println!("");

            // for i in 0..8 {
            //     for j in 0..8 {
            //         print!("{:4} ", matrix_t[i + j * 8]);
            //     }
            //     println!();
            // }

            matvec8x8_col_major(&matrix, &vector, &mut expected, alpha);
            matvec8x8_row_major(&matrix_t, &vector, &mut expected_t, alpha);

            expected
                .iter()
                .zip(expected_t.iter())
                .for_each(|(e, e_t)| assert!((e - e_t).abs() < 1e-5));

            let mut matrix = [c32::zero(); 64];
            let mut result = [c32::zero(); 8];

            for i in 0..8 {
                for j in 0..8 {
                    matrix[i * 8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
                }
            }

            let simd = V3::try_new().unwrap();
            simd.vectorize(ComplexMul8x8Avx32 {
                simd,
                alpha: alpha.re(),
                matrix: &matrix,
                vector: &vector,
                result: &mut result,
            });

            assert!(false);
            // expected.iter().zip(result).for_each(|(e, r)| {
            //     println!("e {:?} r {:?}", e, r);
            //     assert!((e - r).abs() < 1e-10)
            // });
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

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
    // let s1 = vector[0];
    // let s2 = vector[1];
    // let s3 = vector[2];
    // let s4 = vector[3];
    // let s5 = vector[4];
    // let s6 = vector[5];
    // let s7 = vector[6];
    // let s8 = vector[7];

    // for i in 0..8 {
    //     // Row index

    //     let mut sum = U::zero();
    //     sum += matrix[i * 8] * s1;
    //     sum += matrix[i * 8 + 1] * s2;
    //     sum += matrix[i * 8 + 2] * s3;
    //     sum += matrix[i * 8 + 3] * s4;
    //     sum += matrix[i * 8 + 4] * s5;
    //     sum += matrix[i * 8 + 5] * s6;
    //     sum += matrix[i * 8 + 6] * s7;
    //     sum += matrix[i * 8 + 7] * s8;
    //     save_buffer[i] += sum * alpha
    // }

    for i in 0..8 {
        let mut sum = U::zero();
        let row = &matrix[i * 8..(i + 1) * 8];
        for j in 0..8 {
            sum += row[j] * vector[j]
        }

        save_buffer[i] += sum * alpha;
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
            save_buffer[j] += matrix[i * 4 + j] * vector[i]
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
            save_buffer[j] += matrix[i * 8 + j] * vector[i]
            // save_buffer[j] += matrix[i + 8*j] * vector[j]
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
