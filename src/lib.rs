use std::arch::aarch64::{float32x4_t, float32x4x2_t};

use rlst::{c32, c64};

use pulp::{aarch64::Neon, f32x4};
use num_traits::Zero;

pub struct ComplexMulNeon32<'a> {
    pub simd: Neon,
    pub alpha: f32,
    pub matrix: &'a [c32; 4],
    pub vector: &'a [c32; 4],
    pub result: &'a mut [c32; 4]
}

impl pulp::NullaryFnOnce for ComplexMulNeon32<'_> {

    type Output = ();

    #[inline(always)]
    fn call(self) -> Self::Output {
        let Self {
            simd,
            alpha,
            matrix: left,
            vector: right,
            result
        } = self;

        unsafe {
            let ptr = left.as_ptr() as *const f32;
            let left_d: float32x4x2_t = simd.neon.vld2q_f32(ptr);

            let ptr = right.as_ptr() as *const f32;
            let right_d = simd.neon.vld2q_f32(ptr);

            let ac = simd.neon.vmulq_f32(left_d.0, right_d.0);
            let bd = simd.neon.vmulq_f32(left_d.1, right_d.1);
            let re = simd.neon.vsubq_f32(ac, bd);

            let ad = simd.neon.vmulq_f32(left_d.0, right_d.1);
            let bc = simd.neon.vmulq_f32(left_d.1, right_d.0);
            let im = simd.neon.vaddq_f32(ad, bc);

            let res = float32x4x2_t(re, im);
            let ptr = result.as_ptr() as *mut f32;
            simd.neon.vst2q_f32(ptr, res);
        }

    }
}
