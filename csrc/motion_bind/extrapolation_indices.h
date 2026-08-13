#ifndef PYTOAST_EXTRAPOLATION_INDICES_H
#define PYTOAST_EXTRAPOLATION_INDICES_H

#include <toast/utils/vector_math.h>

HOSTDEVICE auto ctz(unsigned mask) {
#ifdef __CUDA_ARCH__
    return __ffs(mask) - 1;
#else
    return mask ? __builtin_ctz(mask) : -1;
#endif
}

HOSTDEVICE auto clz(unsigned mask) {
#ifdef __CUDA_ARCH__
    return __clz(mask);
#else
    return __builtin_clz(mask);
#endif
}

HOSTDEVICE auto popc(unsigned mask) {
#ifdef __CUDA_ARCH__
    return __popc(mask);
#else
    return __builtin_popcount(mask);
#endif
}


HOSTDEVICE int kth_set_bit(uint32_t mask, int k) {
    while (k--) {mask &= (mask - 1);}
    return ctz(mask);
}

HOSTDEVICE int inv_kth_set_bit(uint32_t mask, int k) {
    int count = 0;
    while (mask) {
        int bit = 31 - clz(mask);
        if (count == k) return bit;
        mask &= ~(1u << bit);
        ++count;
    }
    return -1;
}



template <typename F>
HOSTDEVICE int4 find_positions_bitpacked(uint num_keyframes, uint K, const F& mask_packed_accessor) {
    int left_i = -1; // first valid index
    int left_j = -1; // k-th valid index (or last valid index if less than k valid elements exist)
    int right_i = -1; // k-th from
    int right_j = -1; // last valid index

    uint fwd_batch_offset = 0;
    uint fwd_count = 0;

    while ((fwd_count <= K) && (fwd_batch_offset < num_keyframes)) {
        uint cur_mask = mask_packed_accessor(fwd_batch_offset);
        uint cur_num_valid = popc(cur_mask);
        uint new_fwd_count = fwd_count + cur_num_valid;

        if (cur_num_valid) {
            left_i = (left_i == -1) ? fwd_batch_offset + ctz(cur_mask) : left_i;
            left_j = fwd_batch_offset + 31 - clz(cur_mask);
        }
        if (new_fwd_count > K) {
            left_j = fwd_batch_offset + kth_set_bit(cur_mask, K - fwd_count);
        }

        fwd_count = new_fwd_count;
        fwd_batch_offset += 32;
    }

    if (left_i == -1 && left_j == -1) {
        // no valid elements exist
        return {-1, -1, -1, -1};
    }

    if (fwd_count <= K) {
        // k or less valid elements.
        // no point in running backward loop
        return {left_i, left_j, left_i, left_j};
    }

    int bwd_batch_offset = static_cast<int>(num_keyframes / 32) * 32;
    uint bwd_count = 0;

    while ((bwd_count <= K) && (bwd_batch_offset >= 0)) {
        uint cur_mask = mask_packed_accessor(bwd_batch_offset);
        uint cur_num_valid = popc(cur_mask);
        uint new_bwd_count = bwd_count + cur_num_valid;

        if (cur_num_valid) {
            right_j = (right_j == -1) ? bwd_batch_offset + 31 - clz(cur_mask) : right_j;
            right_i = bwd_batch_offset + ctz(cur_mask);
        }
        if (new_bwd_count > K) {
            right_i = bwd_batch_offset + inv_kth_set_bit(cur_mask, (K-bwd_count));
        }

        bwd_count = new_bwd_count;
        bwd_batch_offset -= 32;
    }

    return {left_i, left_j, right_i, right_j};
}

#endif //PYTOAST_EXTRAPOLATION_INDICES_H
