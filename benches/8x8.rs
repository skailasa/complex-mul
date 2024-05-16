use criterion::{criterion_group, criterion_main, Criterion};

use complex_mul::matvec8x8_row_major;
use num_traits::{One, Zero};
use rlst::{c32, c64};

pub fn naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("naive autovectoriser");

    let mut vector = [c32::zero(); 8];
    let mut save_buffer = [c32::zero(); 8];

    let mut matrix = [c32::zero(); 64];
    let alpha = c32::one();

    for i in 0..8 {
        let num = (i + 1) as f32;
        vector[i] = c32::new(num, num);
        for j in 0..8 {
            matrix[i * 8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
        }
    }

    group.bench_function("matvec8x8f32_row_major", |b| {
        b.iter(|| matvec8x8_row_major(&matrix, &vector, &mut save_buffer, alpha))
    });

    let mut vector = [c64::zero(); 8];
    let mut save_buffer = [c64::zero(); 8];

    let mut matrix = [c64::zero(); 64];
    let alpha = c64::one();

    for i in 0..8 {
        let num = (i + 1) as f64;
        vector[i] = c64::new(num, num);
        for j in 0..8 {
            matrix[i * 8 + j] = c64::new((i + 1) as f64, (j + 1) as f64);
        }
    }
    group.bench_function("matvec8x8f64_row_major", |b| {
        b.iter(|| matvec8x8_row_major(&matrix, &vector, &mut save_buffer, alpha))
    });
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    use super::*;
    use complex_mul::{ComplexMul8x8NeonFcma32, ComplexMul8x8NeonFcma64};
    use pulp::aarch64::NeonFcma;
    pub fn explicit_simd(c: &mut Criterion) {
        let mut group = c.benchmark_group("explicit simd");

        let mut matrix = [c32::zero(); 64];
        let mut vector = [c32::zero(); 8];
        let mut result = [c32::zero(); 8];

        let alpha = 1f32;

        for i in 0..8 {
            let num = (i + 1) as f32;
            vector[i] = c32::new(num, num);
            // expected[i] = matrix[i] * vector[i];
            for j in 0..8 {
                matrix[i * 8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
            }
        }

        let simd = NeonFcma::try_new().unwrap();

        group.bench_function("ComplexMul8x8NeonFcma32", |b| {
            b.iter(|| {
                simd.vectorize(ComplexMul8x8NeonFcma32 {
                    simd,
                    alpha,
                    matrix: &matrix,
                    vector: &vector,
                    result: &mut result,
                })
            })
        });

        let mut matrix = [c64::zero(); 64];
        let mut vector = [c64::zero(); 8];
        let mut result = [c64::zero(); 8];

        let alpha = 1f64;

        for i in 0..8 {
            let num = (i + 1) as f64;
            vector[i] = c64::new(num, num);
            // expected[i] = matrix[i] * vector[i];
            for j in 0..8 {
                matrix[i * 8 + j] = c64::new((i + 1) as f64, (j + 1) as f64);
            }
        }

        let simd = NeonFcma::try_new().unwrap();

        group.bench_function("ComplexMul8x8NeonFcma64", |b| {
            b.iter(|| {
                simd.vectorize(ComplexMul8x8NeonFcma64 {
                    simd,
                    alpha,
                    matrix: &matrix,
                    vector: &vector,
                    result: &mut result,
                })
            })
        });
    }
}

#[cfg(target_arch = "x86_64")]
criterion_group!(benches, naive);

#[cfg(target_arch = "aarch64")]
criterion_group!(benches, naive, explicit_simd);

criterion_main!(benches);
