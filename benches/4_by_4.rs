use criterion::{black_box, criterion_group, criterion_main, Criterion};

use complex_mul::{matvec4x4_col_major, matvec4x4_row_major, ComplexMul4x4NeonFcma32, ComplexMul4x4Neon32};
use pulp::aarch64::{Neon, NeonFcma};
use rlst::c32;
use num_traits::{One, Zero};


pub fn naive(c: &mut Criterion) {
    let mut vector = [c32::zero(); 4];
    let mut save_buffer = [c32::zero(); 4];

    let mut matrix = [c32::zero(); 16];
    let alpha = c32::one();

    for i in 0..4 {
        let num = (i+1) as f32;
        vector[i] = c32::new(num, num);
        for j in 0..4 {
            matrix[i*4+j] = c32::new((i + 1) as f32,( j + 1) as f32);
        }
    }

    let mut group = c.benchmark_group("naive autovectoriser");

    group.bench_function("matvec4x4_col_major", |b| {
        b.iter(|| matvec4x4_row_major(&matrix, &vector, &mut save_buffer, alpha))
    });
}

pub fn explicit_simd(c: &mut Criterion) {


    let mut matrix = [[c32::zero(); 4]; 4];
    let mut vector = [c32::zero(); 4];
    let mut result = [c32::zero(); 4];

    let alpha = 1f32;

    for i in 0..matrix.len() {
        let num = (i+1) as f32;
        vector[i] = c32::new(num, num);
        // expected[i] = matrix[i] * vector[i];
        for j in 0..matrix.len() {
            matrix[i][j] = c32::new((i + 1) as f32,( j + 1) as f32);
        }
    }

    let mut group = c.benchmark_group("explicit simd");

    let simd = Neon::try_new().unwrap();

    group.bench_function("ComplexMulNeon32", |b| {
        b.iter(||
            simd.vectorize(ComplexMul4x4Neon32 {
                simd,
                alpha,
                matrix: &matrix,
                vector: &vector,
                result: &mut result
            })
        )
    });

    let mut matrix = [[c32::zero(); 4]; 4];
    let mut vector = [c32::zero(); 4];
    let mut result = [c32::zero(); 4];

    let alpha = 1f32;

    for i in 0..matrix.len() {
        let num = (i+1) as f32;
        vector[i] = c32::new(num, num);
        // expected[i] = matrix[i] * vector[i];
        for j in 0..matrix.len() {
            matrix[i][j] = c32::new((i + 1) as f32,( j + 1) as f32);
        }
    }

    let simd = NeonFcma::try_new().unwrap();

    group.bench_function("ComplexMulNeonFcma32", |b| {
        b.iter(||
            simd.vectorize(ComplexMul4x4NeonFcma32 {
                simd,
                alpha,
                matrix: &matrix,
                vector: &vector,
                result: &mut result
            })
        )
    });

}


criterion_group!(benches, naive, explicit_simd);
criterion_main!(benches);