use std::time::Instant;

use complex_mul::ComplexMulNeon32;

use itertools::izip;
use num_complex::ComplexFloat;
use rlst::c32;
use num_traits::Zero;
use pulp::aarch64::Neon;

fn main() {

    let mut matrix = [c32::zero(); 4];
    let mut vector = [c32::zero(); 4];
    let mut result = [c32::zero(); 4];
    // let mut expected = [c32::zero(); 4];

    let alpha = 0f32;

    for i in 0..matrix.len() {
        let num = (i+1) as f32;
        matrix[i] = c32::new(-1. * num, -2. * num);
        vector[i] = c32::new(num, 2. * num);
        // expected[i] = matrix[i] * vector[i];
    }

    let simd = Neon::try_new().unwrap();

    simd.vectorize(ComplexMulNeon32 {
        simd,
        alpha,
        matrix: &matrix,
        vector: &vector,
        result: &mut result
    });

    // println!("expected {:?}", expected);
    // println!("found {:?}", result);

    // let n = 1000000*4;
    // let mut left = vec![c32::zero(); n];
    // let mut right = vec![c32::zero(); n];
    // let mut result = vec![c32::zero(); n];
    // let mut expected = vec![c32::zero(); n];

    // for i in 0..left.len() {
    //     let num = (i+1) as f32;
    //     left[i] = c32::new(-1. * num, -2. * num);
    //     right[i] = c32::new(num, 2. * num);
    // }

    // let (left_head, left_tail) = pulp::as_arrays::<4, _>(left.as_slice());
    // let (right_head, right_tail) = pulp::as_arrays::<4, _>(right.as_slice());
    // let (result_head, result_tail) = pulp::as_arrays_mut::<4, _>(result.as_mut_slice());

    // let simd = Neon::try_new().unwrap();
    // let alpha = 0.0;

    // let s = Instant::now();
    // for (left, right, res) in izip!(left_head, right_head, result_head) {
    //     simd.vectorize(ComplexMulNeon32 {
    //         simd,
    //         alpha,
    //         left: &left,
    //         right: &right,
    //         result:  res
    //     })
    // };
    // println!("SIMD {:?}", s.elapsed());

    // let s = Instant::now();
    // for i in 0..left.len() {
    //     expected[i] = left[i] * right[i];
    // }
    // println!("AUTO {:?}", s.elapsed());

    // expected.iter().zip(result.iter()).for_each(|(a, b)| {
    //     // println!("{:?} {:?} {:?}",a, b, (a-b).abs());
    //     assert!((a-b).abs() < 1e-5)
    // })
}
