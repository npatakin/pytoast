#include "interpolation.cuh"
#include "extrapolation_indices.h"

#include <toast/utils/vector_math.h>
#include <toast/utils/bwd_math_utils.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>
#include <toast/math/motion.h>


#ifdef TOAST_WITH_CUDA

__global__ void extrapolate_indices_cuda(
    const uint batch_size,
    const uint num_keyframes,
    const uint extrapolate_window,
    const bool* masks,
    int4* extrapolate_indices
) {
    // block_size: [32, batch_elements_per_block]
    // grid_size: [(batch_size + batch_elements_per_block - 1) / batch_elements_per_block,]

    const uint batch_element_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (batch_element_idx >= batch_size)
        return;
    const uint laneIdx = threadIdx.x;

    masks += batch_element_idx * num_keyframes;

    extrapolate_indices[batch_element_idx] = find_positions_bitpacked(num_keyframes, extrapolate_window, [&](uint pack_offset) {
        const uint element_idx = pack_offset + laneIdx;
        const bool cur_mask_value = (element_idx < num_keyframes) ? masks[element_idx] : false;
        return __ballot_sync(0xFFFFFFFF, cur_mask_value);
    });
}

#endif // TOAST_WITH_CUDA


torch::Tensor extrapolate_indices(const torch::Tensor& masks, uint extrapolate_window) {
    const uint batch_size = masks.size(0);
    const uint num_keyframes = masks.size(1);

    auto out = torch::empty({batch_size, 4}, at::TensorOptions().device(masks.device()).dtype(torch::kInt32));

    auto masks_ptr = (const bool*)masks.data_ptr();
    auto out_ptr = (int4*)out.data_ptr();

    if (masks.device().is_cuda()) {
#ifdef TOAST_WITH_CUDA
        const at::cuda::OptionalCUDAGuard device_guard(masks.device());
        auto stream = at::cuda::getCurrentCUDAStream();
        const uint num_threads = 1024;

        dim3 threads(32, num_threads / 32, 1);
        const uint num_blocks = (batch_size + threads.y - 1) / threads.y;

        extrapolate_indices_cuda<<<num_blocks, threads, 0, stream>>>(
            batch_size, num_keyframes, extrapolate_window, masks_ptr, out_ptr
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
        TORCH_CHECK(false, TOAST_NO_CUDA_SUPPORT_MSG);
#endif
    } else {

        at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; i++) {
                const bool* cur_masks_ptr = masks_ptr + num_keyframes * i;
                out_ptr[i] = find_positions_bitpacked(num_keyframes, extrapolate_window, [&](uint pack_offset) {
                    uint result = 0;
                    #pragma unroll
                    for (int bit_idx = 0; bit_idx < min(32, num_keyframes - pack_offset); ++bit_idx) {
                        result |= cur_masks_ptr[pack_offset + bit_idx] << bit_idx;
                    }
                    return result;
                });
            }
        });
    }

    return out;
}





void bind_interpolation(py::module_ &m) {
    m.def("extrapolate_indices", &extrapolate_indices);


}
