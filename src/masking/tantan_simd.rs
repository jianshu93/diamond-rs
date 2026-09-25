//! SIMD TANTAN forward/backward steps for x86-64 and AArch64.
//!
//! Uses AVX2 or SSE2 on x86-64 and NEON on AArch64, selected at runtime.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Check whether the AVX2 implementation is available at runtime.
#[cfg(target_arch = "x86_64")]
pub fn has_avx2_fma() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2_fma() -> bool {
    false
}

pub fn has_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return has_avx2_fma() || is_x86_feature_detected!("sse2");
    }
    #[cfg(target_arch = "aarch64")]
    {
        return true;
    }
    #[allow(unreachable_code)]
    false
}

/// AVX2 horizontal sum: matches C++ hsum(__m256 a) exactly.
///   1. Split into two 128-bit halves, add them
///   2. Two horizontal adds
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(a: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(a);
    let vhigh = _mm256_extractf128_ps(a, 1);
    let vsum = _mm_add_ps(vlow, vhigh);
    let vsum = _mm_hadd_ps(vsum, vsum);
    let vsum = _mm_hadd_ps(vsum, vsum);
    _mm_cvtss_f32(vsum)
}

/// AVX2 forward step: matches C++ forward_step() exactly.
///
/// Processes f[0..48] with AVX2 (6 chunks of 8), then f[48..50] scalar.
///
/// The C++ build compiles arch_avx2 with `-mavx2` but **not** `-mfma`, so its
/// `fmadd(a, b, c)` macro falls back to `add(mul(a, b), c)` (vector8_avx2.h:135).
/// We must do the same here: fused multiply-add would differ by 1 ULP and flip
/// mask decisions at the p_mask boundary.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn forward_step_avx2(
    f: &mut [f32; 50],
    d: &[f32; 50],
    e_seg: &[f32],
    b: &mut f32,
    f2f: f32,
    p_repeat_end: f32,
    b2b: f32,
    f_sum_prev: f32,
) -> f32 {
    let b_old = *b;
    let vf2f = _mm256_set1_ps(f2f);
    let vb_old = _mm256_set1_ps(b_old);
    let mut f_sum_new = 0.0f32;

    // Process 48 elements in 6 SIMD chunks
    for off in (0..48).step_by(8) {
        let vf = _mm256_loadu_ps(f.as_ptr().add(off));
        let vd = _mm256_loadu_ps(d.as_ptr().add(off));
        let ve = _mm256_loadu_ps(e_seg.as_ptr().add(off));
        // tmp = (vf * vf2f) + (vb_old * vd)  — separate mul+add, NOT FMA.
        let tmp = _mm256_add_ps(_mm256_mul_ps(vf, vf2f), _mm256_mul_ps(vb_old, vd));
        let vf_new = _mm256_mul_ps(tmp, ve);
        _mm256_storeu_ps(f.as_mut_ptr().add(off), vf_new);
        f_sum_new += hsum_avx2(vf_new);
    }

    // Scalar tail for elements 48, 49
    for off in 48..50 {
        let vf = (f[off] * f2f + b_old * d[off]) * e_seg[off];
        f[off] = vf;
        f_sum_new += vf;
    }

    *b = b_old * b2b + f_sum_prev * p_repeat_end;
    f_sum_new
}

/// AVX2 backward step: matches C++ backward_step() exactly.
/// See forward_step_avx2 for the reason FMA is avoided.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn backward_step_avx2(
    f: &mut [f32; 50],
    d: &[f32; 50],
    e_seg: &[f32],
    b: &mut f32,
    f2f: f32,
    p_repeat_end: f32,
    b2b: f32,
) {
    let vf2f = _mm256_set1_ps(f2f);
    let vc = _mm256_set1_ps(p_repeat_end * *b);
    let mut tsum = 0.0f32;

    for off in (0..48).step_by(8) {
        let vf = _mm256_loadu_ps(f.as_ptr().add(off));
        let ve = _mm256_loadu_ps(e_seg.as_ptr().add(off));
        let vd = _mm256_loadu_ps(d.as_ptr().add(off));
        let vf_e = _mm256_mul_ps(vf, ve);
        let vt = _mm256_mul_ps(vf_e, vd);
        tsum += hsum_avx2(vt);
        // f[off] = (vf_e * vf2f) + vc  — separate mul+add, NOT FMA.
        let vf_new = _mm256_add_ps(_mm256_mul_ps(vf_e, vf2f), vc);
        _mm256_storeu_ps(f.as_mut_ptr().add(off), vf_new);
    }

    for off in 48..50 {
        let vf = f[off] * e_seg[off];
        tsum += vf * d[off];
        f[off] = vf * f2f + p_repeat_end * *b;
    }

    *b = b2b * *b + tsum;
}

/// AVX2 scale: multiply all 50 elements by s (matches C++ SIMD::scale)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn scale_avx2(f: &mut [f32; 50], s: f32) {
    let vs = _mm256_set1_ps(s);
    for off in (0..48).step_by(8) {
        let v = _mm256_loadu_ps(f.as_ptr().add(off));
        _mm256_storeu_ps(f.as_mut_ptr().add(off), _mm256_mul_ps(v, vs));
    }
    f[48] *= s;
    f[49] *= s;
}

/// AVX2 sum of all 50 elements (for terminal z computation)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sum_avx2(f: &[f32; 50]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for off in (0..48).step_by(8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(f.as_ptr().add(off)));
    }
    hsum_avx2(acc) + f[48] + f[49]
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn forward_step_simd(
    f: &mut [f32; 50],
    d: &[f32; 50],
    e_seg: &[f32],
    b: &mut f32,
    f2f: f32,
    p_repeat_end: f32,
    b2b: f32,
    f_sum_prev: f32,
) -> f32 {
    if has_avx2_fma() {
        forward_step_avx2(f, d, e_seg, b, f2f, p_repeat_end, b2b, f_sum_prev)
    } else {
        forward_step_sse2(f, d, e_seg, b, f2f, p_repeat_end, b2b, f_sum_prev)
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn backward_step_simd(
    f: &mut [f32; 50],
    d: &[f32; 50],
    e_seg: &[f32],
    b: &mut f32,
    f2f: f32,
    p_repeat_end: f32,
    b2b: f32,
) {
    if has_avx2_fma() {
        backward_step_avx2(f, d, e_seg, b, f2f, p_repeat_end, b2b)
    } else {
        backward_step_sse2(f, d, e_seg, b, f2f, p_repeat_end, b2b)
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn scale_simd(f: &mut [f32; 50], s: f32) {
    if has_avx2_fma() {
        scale_avx2(f, s)
    } else {
        scale_sse2(f, s)
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn sum_simd(f: &[f32; 50]) -> f32 {
    if has_avx2_fma() {
        sum_avx2(f)
    } else {
        sum_sse2(f)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hsum_sse2(value: __m128) -> f32 {
    let high = _mm_movehl_ps(value, value);
    let pair = _mm_add_ps(value, high);
    let shuffled = _mm_shuffle_ps(pair, pair, 0x55);
    _mm_cvtss_f32(_mm_add_ss(pair, shuffled))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn forward_step_sse2(
    f: &mut [f32; 50],
    d: &[f32; 50],
    e_seg: &[f32],
    b: &mut f32,
    f2f: f32,
    p_repeat_end: f32,
    b2b: f32,
    f_sum_prev: f32,
) -> f32 {
    let old_b = *b;
    let vf2f = _mm_set1_ps(f2f);
    let vb = _mm_set1_ps(old_b);
    let mut total = 0.0;
    for offset in (0..48).step_by(4) {
        let value = _mm_loadu_ps(f.as_ptr().add(offset));
        let repeat = _mm_loadu_ps(d.as_ptr().add(offset));
        let emission = _mm_loadu_ps(e_seg.as_ptr().add(offset));
        let next = _mm_mul_ps(
            _mm_add_ps(_mm_mul_ps(value, vf2f), _mm_mul_ps(vb, repeat)),
            emission,
        );
        _mm_storeu_ps(f.as_mut_ptr().add(offset), next);
        total += hsum_sse2(next);
    }
    for offset in 48..50 {
        f[offset] = (f[offset] * f2f + old_b * d[offset]) * e_seg[offset];
        total += f[offset];
    }
    *b = old_b * b2b + f_sum_prev * p_repeat_end;
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn backward_step_sse2(
    f: &mut [f32; 50],
    d: &[f32; 50],
    e_seg: &[f32],
    b: &mut f32,
    f2f: f32,
    p_repeat_end: f32,
    b2b: f32,
) {
    let vf2f = _mm_set1_ps(f2f);
    let constant = _mm_set1_ps(p_repeat_end * *b);
    let mut total = 0.0;
    for offset in (0..48).step_by(4) {
        let value = _mm_mul_ps(
            _mm_loadu_ps(f.as_ptr().add(offset)),
            _mm_loadu_ps(e_seg.as_ptr().add(offset)),
        );
        total += hsum_sse2(_mm_mul_ps(value, _mm_loadu_ps(d.as_ptr().add(offset))));
        _mm_storeu_ps(
            f.as_mut_ptr().add(offset),
            _mm_add_ps(_mm_mul_ps(value, vf2f), constant),
        );
    }
    for offset in 48..50 {
        let value = f[offset] * e_seg[offset];
        total += value * d[offset];
        f[offset] = value * f2f + p_repeat_end * *b;
    }
    *b = b2b * *b + total;
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn scale_sse2(f: &mut [f32; 50], scale: f32) {
    let scale = _mm_set1_ps(scale);
    for offset in (0..48).step_by(4) {
        _mm_storeu_ps(
            f.as_mut_ptr().add(offset),
            _mm_mul_ps(_mm_loadu_ps(f.as_ptr().add(offset)), scale),
        );
    }
    f[48] *= _mm_cvtss_f32(scale);
    f[49] *= _mm_cvtss_f32(scale);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sum_sse2(f: &[f32; 50]) -> f32 {
    let mut total = _mm_setzero_ps();
    for offset in (0..48).step_by(4) {
        total = _mm_add_ps(total, _mm_loadu_ps(f.as_ptr().add(offset)));
    }
    hsum_sse2(total) + f[48] + f[49]
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn forward_step_simd(
    f: &mut [f32; 50],
    d: &[f32; 50],
    e_seg: &[f32],
    b: &mut f32,
    f2f: f32,
    p_repeat_end: f32,
    b2b: f32,
    f_sum_prev: f32,
) -> f32 {
    use std::arch::aarch64::*;

    let b_old = *b;
    let vf2f = vdupq_n_f32(f2f);
    let vb_old = vdupq_n_f32(b_old);
    let mut total = 0.0f32;
    for off in (0..48).step_by(4) {
        let vf = vld1q_f32(f.as_ptr().add(off));
        let vd = vld1q_f32(d.as_ptr().add(off));
        let ve = vld1q_f32(e_seg.as_ptr().add(off));
        let tmp = vfmaq_f32(vmulq_f32(vb_old, vd), vf, vf2f);
        let next = vmulq_f32(tmp, ve);
        vst1q_f32(f.as_mut_ptr().add(off), next);
        total += vaddvq_f32(next);
    }
    for off in 48..50 {
        f[off] = (f[off] * f2f + b_old * d[off]) * e_seg[off];
        total += f[off];
    }
    *b = b_old * b2b + f_sum_prev * p_repeat_end;
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn backward_step_simd(
    f: &mut [f32; 50],
    d: &[f32; 50],
    e_seg: &[f32],
    b: &mut f32,
    f2f: f32,
    p_repeat_end: f32,
    b2b: f32,
) {
    use std::arch::aarch64::*;

    let vf2f = vdupq_n_f32(f2f);
    let vc = vdupq_n_f32(p_repeat_end * *b);
    let mut total = 0.0f32;
    for off in (0..48).step_by(4) {
        let vf = vld1q_f32(f.as_ptr().add(off));
        let ve = vld1q_f32(e_seg.as_ptr().add(off));
        let vd = vld1q_f32(d.as_ptr().add(off));
        let vf_e = vmulq_f32(vf, ve);
        let weighted = vmulq_f32(vf_e, vd);
        total += vaddvq_f32(weighted);
        let next = vfmaq_f32(vc, vf_e, vf2f);
        vst1q_f32(f.as_mut_ptr().add(off), next);
    }
    for off in 48..50 {
        let vf = f[off] * e_seg[off];
        total += vf * d[off];
        f[off] = vf * f2f + p_repeat_end * *b;
    }
    *b = b2b * *b + total;
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn scale_simd(f: &mut [f32; 50], s: f32) {
    use std::arch::aarch64::*;

    let scale = vdupq_n_f32(s);
    for off in (0..48).step_by(4) {
        let values = vld1q_f32(f.as_ptr().add(off));
        vst1q_f32(f.as_mut_ptr().add(off), vmulq_f32(values, scale));
    }
    f[48] *= s;
    f[49] *= s;
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sum_simd(f: &[f32; 50]) -> f32 {
    use std::arch::aarch64::*;

    let mut acc = vdupq_n_f32(0.0);
    for off in (0..48).step_by(4) {
        let values = vld1q_f32(f.as_ptr().add(off));
        acc = vaddq_f32(acc, values);
    }
    vaddvq_f32(acc) + f[48] + f[49]
}

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests {
    use super::*;

    fn inputs() -> ([f32; 50], [f32; 50], [f32; 50]) {
        let mut f = [0.0; 50];
        let mut d = [0.0; 50];
        let mut e = [0.0; 50];
        for i in 0..50 {
            f[i] = (i as f32 + 1.0) / 97.0;
            d[i] = (50 - i) as f32 / 193.0;
            e[i] = (i % 11 + 3) as f32 / 17.0;
        }
        (f, d, e)
    }

    #[test]
    fn neon_forward_backward_scale_and_sum_match_scalar() {
        let close = |left: f32, right: f32| {
            let tolerance = 8.0 * f32::EPSILON * left.abs().max(right.abs()).max(1.0);
            assert!((left - right).abs() <= tolerance, "{left} != {right}");
        };
        let arrays_close = |left: &[f32], right: &[f32]| {
            for (&left, &right) in left.iter().zip(right) {
                close(left, right);
            }
        };
        let (mut f, d, e) = inputs();
        let mut expected = f;
        let mut b = 0.73f32;
        let mut expected_b = b;
        let f2f = 0.91f32;
        let repeat_end = 0.08f32;
        let b2b = 0.87f32;
        let previous_sum = 1.13f32;
        let old_b = expected_b;
        let mut expected_sum = 0.0;
        for i in 0..50 {
            expected[i] = (expected[i] * f2f + old_b * d[i]) * e[i];
            expected_sum += expected[i];
        }
        expected_b = old_b * b2b + previous_sum * repeat_end;
        let sum = unsafe {
            forward_step_simd(&mut f, &d, &e, &mut b, f2f, repeat_end, b2b, previous_sum)
        };
        arrays_close(&f, &expected);
        close(b, expected_b);
        close(sum, expected_sum);

        let old_b = b;
        let mut expected_total = 0.0;
        for i in 0..50 {
            let value = expected[i] * e[i];
            expected_total += value * d[i];
            expected[i] = value * f2f + repeat_end * old_b;
        }
        expected_b = b2b * old_b + expected_total;
        unsafe { backward_step_simd(&mut f, &d, &e, &mut b, f2f, repeat_end, b2b) };
        arrays_close(&f, &expected);
        close(b, expected_b);

        let factor = 1.37f32;
        expected.iter_mut().for_each(|value| *value *= factor);
        unsafe { scale_simd(&mut f, factor) };
        arrays_close(&f, &expected);
        close(unsafe { sum_simd(&f) }, f.iter().sum::<f32>());
    }
}
