use crate::basic::reduction::Reduction;
use crate::basic::value::{is_amino_acid, letter_mask, Letter, SEED_MASK};

/// Matches C++ `reduce_seq_generic(const Letter*, const Letter*)`.
pub fn reduce_seq_generic(seq: &[Letter], map: &[Letter]) -> [Letter; 16] {
    let mut d = [0; 16];
    for i in 0..16 {
        d[i] = map[letter_mask(seq[i]) as usize];
    }
    d
}

/// Matches C++ `reduce_seq(const Letter*, const Letter*)`.
pub fn reduce_seq(seq: &[Letter], map: &[Letter]) -> [Letter; 16] {
    reduce_seq_generic(seq, map)
}

/// Matches C++ `match_block_reduced(const Letter*, const Letter*, const Reduction&)`.
pub fn match_block_reduced(x: &[Letter], y: &[Letter], reduction: &Reduction) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if x.len() >= 16 && y.len() >= 16 {
        return unsafe { match_block_reduced_neon(x, y, reduction) };
    }
    match_block_reduced_partial(x, y, 16, reduction)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn match_block_reduced_neon(x: &[Letter], y: &[Letter], reduction: &Reduction) -> u32 {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(31);
    let vx = vandq_u8(vld1q_u8(x.as_ptr().cast()), mask);
    let vy = vandq_u8(vld1q_u8(y.as_ptr().cast()), mask);
    let map = reduction.map8();
    let table_lo = vld1q_u8(map.as_ptr().cast());
    let table_hi = vld1q_u8(map.as_ptr().add(16).cast());
    let sixteen = vdupq_n_u8(16);
    let reduce = |letters: uint8x16_t| {
        let low = vqtbl1q_u8(table_lo, letters);
        let high = vqtbl1q_u8(table_hi, vsubq_u8(letters, sixteen));
        vbslq_u8(vcltq_u8(letters, sixteen), low, high)
    };
    let valid = |letters: uint8x16_t| {
        vandq_u8(
            vandq_u8(
                vmvnq_u8(vceqq_u8(letters, vdupq_n_u8(23))),
                vmvnq_u8(vceqq_u8(letters, vdupq_n_u8(24))),
            ),
            vmvnq_u8(vceqq_u8(letters, vdupq_n_u8(31))),
        )
    };
    let matches = vandq_u8(
        vceqq_u8(reduce(vx), reduce(vy)),
        vandq_u8(valid(vx), valid(vy)),
    );
    let mut lanes = [0u8; 16];
    vst1q_u8(lanes.as_mut_ptr(), matches);
    lanes.iter().enumerate().fold(0u32, |bits, (lane, &value)| {
        bits | (((value >> 7) as u32) << lane)
    })
}

fn match_block_reduced_partial(x: &[Letter], y: &[Letter], n: usize, reduction: &Reduction) -> u32 {
    let mut r = 0u32;
    for i in (0..n.min(x.len()).min(y.len())).rev() {
        r <<= 1;
        let lx = letter_mask(x[i]);
        let ly = letter_mask(y[i]);
        if !is_amino_acid(lx) || !is_amino_acid(ly) {
            continue;
        }
        if reduction.reduce(lx) == reduction.reduce(ly) {
            r |= 1;
        }
    }
    r
}

/// Matches C++ `reduced_match32(const Letter*, const Letter*, unsigned, const Reduction&)`.
pub fn reduced_match32(q: &[Letter], s: &[Letter], len: u32, reduction: &Reduction) -> u64 {
    let len = (len as usize).min(q.len()).min(s.len());
    let mut x = match_block_reduced_partial(q, s, len.min(16), reduction) as u64;
    if len > 16 {
        x |= (match_block_reduced_partial(&q[16..], &s[16..], (len - 16).min(16), reduction)
            as u64)
            << 16;
    }
    if len < 32 {
        x &= (1u64 << len) - 1;
    }
    x
}

/// Matches C++ `reduced_match(const Letter*, const Letter*, int, const Reduction&)`.
pub fn reduced_match(q: &[Letter], s: &[Letter], len: i32, reduction: &Reduction) -> u64 {
    assert!(len <= 64);
    let len = (len as usize).min(q.len()).min(s.len());
    if len < 64 {
        let mask = (1u64 << len) - 1;
        let mut m = match_block_reduced_partial(q, s, len.min(16), reduction) as u64;
        if len <= 16 {
            return m & mask;
        }
        m |= (match_block_reduced_partial(&q[16..], &s[16..], (len - 16).min(16), reduction)
            as u64)
            << 16;
        if len <= 32 {
            return m & mask;
        }
        m |= (match_block_reduced_partial(&q[32..], &s[32..], (len - 32).min(16), reduction)
            as u64)
            << 32;
        if len <= 48 {
            return m & mask;
        }
        m |= (match_block_reduced_partial(&q[48..], &s[48..], len - 48, reduction) as u64) << 48;
        m & mask
    } else {
        match_block_reduced(q, s, reduction) as u64
            | ((match_block_reduced(&q[16..], &s[16..], reduction) as u64) << 16)
            | ((match_block_reduced(&q[32..], &s[32..], reduction) as u64) << 32)
            | ((match_block_reduced(&q[48..], &s[48..], reduction) as u64) << 48)
    }
}

/// Matches C++ `seed_mask(const Letter*, int)`.
pub fn seed_mask(s: &[Letter], len: i32) -> u64 {
    assert!(len <= 64);
    let mut mask = 0u64;
    for i in 0..(len as usize).min(s.len()) {
        if (s[i] & SEED_MASK) != 0 {
            mask |= 1u64 << i;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::value::{MASK_LETTER, STOP_LETTER};

    #[test]
    fn test_match_block_reduced() {
        let reduction = Reduction::default_reduction();
        let x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut y = x;
        let reduced = reduce_seq(&x, reduction.map8());
        for i in 0..16 {
            assert_eq!(reduced[i], reduction.map8()[x[i] as usize]);
        }
        assert_eq!(match_block_reduced(&x, &y, &reduction), 0xffff);
        y[0] = 1;
        y[1] = 11;
        y[2] = STOP_LETTER;
        let mask = match_block_reduced(&x, &y, &reduction);
        assert_eq!(mask & 1, 0);
        assert_eq!(mask & 2, 2);
        assert_eq!(mask & 4, 0);
    }

    #[test]
    fn test_match_block_reduced_accepts_short_tail_blocks() {
        let reduction = Reduction::default_reduction();
        let x = [0, 1, 2, 3, 4, 5, 6];

        assert_eq!(match_block_reduced(&x, &x, &reduction), 0x7f);
        assert_eq!(reduced_match(&x, &x, 64, &reduction), 0x7f);
        assert_eq!(reduced_match32(&x, &x, 32, &reduction), 0x7f);
        assert_eq!(seed_mask(&x, 64), 0);
    }

    #[test]
    fn test_reduced_match_and_seed_mask() {
        let reduction = Reduction::default_reduction();
        let q: Vec<Letter> = (0..64).map(|i| (i % 20) as Letter).collect();
        let mut s = q.clone();
        s[3] = MASK_LETTER;
        let m = reduced_match(&q, &s, 20, &reduction);
        assert_eq!(m & (1 << 0), 1);
        assert_eq!(m & (1 << 3), 0);
        assert_eq!(m >> 20, 0);
        assert_eq!(reduced_match32(&q, &s, 20, &reduction), m);

        let mut masked = vec![0i8; 64];
        masked[2] = SEED_MASK;
        masked[9] = SEED_MASK | 1;
        assert_eq!(seed_mask(&masked, 12), (1 << 2) | (1 << 9));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_reduced_match_matches_scalar() {
        let reduction = Reduction::default_reduction();
        for offset in 0..32u8 {
            let x = (0..16)
                .map(|i| ((i * 7 + offset as usize) % 32) as Letter)
                .collect::<Vec<_>>();
            let mut y = x.clone();
            y[offset as usize % 16] = ((offset as usize + 11) % 32) as Letter;
            assert_eq!(
                unsafe { match_block_reduced_neon(&x, &y, &reduction) },
                match_block_reduced_partial(&x, &y, 16, &reduction)
            );
        }
    }
}
