use complex_mul::{matvec4x4_col_major, ComplexMul4x4NeonFcma32};
use num_traits::{One, Zero};
use pulp::aarch64::NeonFcma;
use rlst::c32;

fn main() {
    let mut vector = [c32::zero(); 4];
    let mut save_buffer = [c32::zero(); 4];

    let mut matrix = [c32::zero(); 16];
    let alpha = c32::one();

    for i in 0..4 {
        let num = (i + 1) as f32;
        vector[i] = c32::new(num, num);
        for j in 0..4 {
            matrix[i * 4 + j] = c32::new((i + 1) as f32, (j + 1) as f32);
        }
    }

    matvec4x4_col_major(&matrix, &vector, &mut save_buffer, alpha);

    println!("FOO {:?}", save_buffer);

    let mut matrix = [[c32::zero(); 4]; 4];
    let mut vector = [c32::zero(); 4];
    let mut result = [c32::zero(); 4];
    let mut expected = [c32::zero(); 4];

    let alpha = 1f32;

    for i in 0..8 {
        let num = (i + 1) as f32;
        vector[i] = c32::new(num, num);
        // expected[i] = matrix[i] * vector[i];
        for j in 0..8 {
            matrix[i][j] = c32::new((i + 1) as f32, (j + 1) as f32);
        }
    }

    // let simd = Neon::try_new().unwrap();
    // let s = Instant::now();
    // simd.vectorize(ComplexMulNeon32 {
    //     simd,
    //     alpha,
    //     matrix: &matrix,
    //     vector: &vector,
    //     result: &mut result
    // });
    // println!("SIMD {:?}", s.elapsed());

    // let s = Instant::now();
    let simd = NeonFcma::try_new().unwrap();
    simd.vectorize(ComplexMul4x4NeonFcma32 {
        simd,
        alpha,
        matrix: &matrix,
        vector: &vector,
        result: &mut result,
    });

    println!("BAR {:?}", result);
    // println!("SIMD {:?}", s.elapsed());
}
