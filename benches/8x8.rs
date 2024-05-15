use criterion::{criterion_group, criterion_main, Criterion};

use complex_mul::{matvec8x8_row_major, ComplexMul8x8NeonFcma32};
use num_traits::{One, Zero};
use pulp::aarch64::NeonFcma;
use rlst::c32;

pub fn naive(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("naive autovectoriser");

    group.bench_function("matvec8x8_row_major", |b| {
        b.iter(|| matvec8x8_row_major(&matrix, &vector, &mut save_buffer, alpha))
    });
}

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
            matrix[i*8 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
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
}

criterion_group!(benches, naive, explicit_simd);
criterion_main!(benches);
