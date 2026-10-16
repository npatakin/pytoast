#include "trajectory.cuh"
#include "interpolation.cuh"

#include <toast/utils/vector_math.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>
#include <toast/math/motion.h>

#include <ATen/ExpandUtils.h>


struct TrajParams {
    uint num_keyframes;
    uint extrapolation_window;
    bool extrapolate;
    bool use_only_valid;
    bool has_extr;
    bool mask_op_and;
    bool has_quats;
    bool has_t;
    bool has_mask;
    bool want_mask;
    bool want_v;
    bool want_w;
    bool want_indices;
};


// Largest index with time[idx] <= query, or -1.  time[0..n) must be sorted.
template <typename T>
HOSTDEVICE int last_le_index(const T* time, uint n, const T& query) {
    uint lo = 0, hi = n;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2;
        if (time[mid] <= query) lo = mid + 1;
        else hi = mid;
    }
    return static_cast<int>(lo) - 1;
}


// Bracketing keyframes for one query among keyframes [first, last]: binary
// search, then walk outward to the nearest valid keyframe on each side.
// mask == nullptr means "all valid".
//
// The search is bounded to [first, last] -- the first and the last valid
// keyframe when masking -- rather than the whole row.  Rows are commonly padded
// past their last keyframe with masked entries holding no real timestamp
// (zeros); searching through those would break the sorted order the binary
// search relies on and send every query to the end of the row.  Inside
// [first, last] the timestamps must be sorted, masked ones included.
//
// Returns left < first for a query before the range and right > last for one
// at or after its end.  Both ends being valid, the walks stay inside the range
// otherwise.
template <typename T>
HOSTDEVICE int2 bracket_query(
    const T* time, const bool* mask, int first, int last, const T& query_time
) {
    int left = first + last_le_index(
        time + first, static_cast<uint>(last - first + 1), query_time);
    int right = left + 1;

    if (mask != nullptr) {
        while (left >= first && !mask[left]) --left;
        while (right <= last && !mask[right]) ++right;
    }
    return int2{left, right};
}


DEFINE_KERNEL_CAPTURE(launch_trajectory_fwd, {
    const auto [out_quat, out_t, out_mask, out_v, out_w, out_idx,
                query_times, seq_times, seq_quats, seq_t, seq_mask, extr_pairs] = ptrs;
    using T = std::remove_const_t<std::remove_pointer_t<decltype(query_times)>>;
    using dtype = dtype_t<T>;
    using vec3_t = typename dtype::vec3;
    using quat_type = quat<dtype>;

    const auto& params = std::get<0>(captures);
    const uint K = params.num_keyframes;

    const T* time_base = seq_times + K * offsets[7];
    const bool* mask_base = params.has_mask ? (seq_mask + K * offsets[10]) : nullptr;
    // Masked keyframes only take part in bracketing when asked for.
    const bool* bracket_mask = params.use_only_valid ? mask_base : nullptr;

    const T query_time = query_times[offsets[6]];

    // The boundary keyframe pairs (first valid, k-th valid from the start,
    // k-th valid from the end, last valid) depend only on the mask, so they are
    // shared by every query in this batch element: either precomputed once by
    // extrapolate_indices(), or a closed form when nothing is masked out.  All
    // four are -1 when the element has no valid keyframe.
    int4 pairs;
    if (params.has_extr) {
        pairs = extr_pairs[offsets[11]];
    } else {
        const int last = static_cast<int>(K) - 1;
        const int k = min(static_cast<int>(params.extrapolation_window), last);
        pairs = int4{0, k, last - k, last};
    }
    const int first_keyframe = pairs.x;
    const int last_keyframe = pairs.w;

    int2 ref = int2{-1, -1};
    if (first_keyframe >= 0) {
        ref = bracket_query(
            time_base, bracket_mask, first_keyframe, last_keyframe, query_time);

        if (ref.x < first_keyframe || ref.y > last_keyframe) {
            // Query falls outside the sequence.
            const bool before = ref.x < first_keyframe;
            if (!params.extrapolate) {
                // clamp to the boundary pose rather than extrapolating through it
                pairs = int4{pairs.x, pairs.x, pairs.w, pairs.w};
            }
            ref = before ? int2{pairs.x, pairs.y} : int2{pairs.z, pairs.w};
        }
    }

    if (params.want_indices) out_idx[offsets[5]] = ref;

    if (ref.x < 0 || ref.y < 0) {
        // no valid keyframes at all -- identity rotation, zero everything else
        if (params.want_mask) out_mask[offsets[2]] = false;
        if (params.has_quats) out_quat[offsets[0]] = quat_type();
        if (params.has_t) out_t[offsets[1]] = vec3_t{T(0), T(0), T(0)};
        if (params.want_v) out_v[offsets[3]] = vec3_t{T(0), T(0), T(0)};
        if (params.want_w) out_w[offsets[4]] = vec3_t{T(0), T(0), T(0)};
    } else {
        const T left_time = time_base[ref.x];
        const T right_time = time_base[ref.y];
        const T delta_time = right_time - left_time;
        const bool valid_delta = delta_time > dtype::eps;
        const T rel_time = valid_delta ? (query_time - left_time) / delta_time : T(0.5);

        if (params.want_mask) {
            if (!params.has_mask) {
                out_mask[offsets[2]] = true;
            } else {
                const bool l = mask_base[ref.x];
                const bool r = mask_base[ref.y];
                out_mask[offsets[2]] = params.mask_op_and ? (l && r) : (l || r);
            }
        }

        if (params.has_quats) {
            const quat_type* q_base = (const quat_type*)(seq_quats) + K * offsets[8];
            const quat_type q_left = q_base[ref.x];
            const quat_type q_right = q_base[ref.y];
            out_quat[offsets[0]] = q_left.slerp(q_right, rel_time);
            if (params.want_w) {
                out_w[offsets[4]] = valid_delta
                    ? angular_velocity_short_arc<dtype>(q_left, q_right, delta_time)
                    : vec3_t{T(0), T(0), T(0)};
            }
        }

        if (params.has_t) {
            const vec3_t* t_base = (const vec3_t*)(seq_t) + K * offsets[9];
            const vec3_t t_left = t_base[ref.x];
            const vec3_t t_right = t_base[ref.y];
            const vec3_t diff = t_right - t_left;
            out_t[offsets[1]] = t_left + rel_time * diff;
            if (params.want_v) {
                out_v[offsets[3]] = valid_delta ? diff / delta_time
                                                : vec3_t{T(0), T(0), T(0)};
            }
        }
    }
})


// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

namespace {

// (..., K, C) -> (..., Q, K*C) with stride 0 along Q.
at::Tensor as_block(const at::Tensor& seq, int64_t num_queries, int64_t num_keyframes) {
    auto sizes = seq.sizes().vec();
    const int64_t channels = (seq.dim() == 0) ? 1 : sizes.back();
    // collapse the trailing (K, C) into one dimension
    std::vector<int64_t> flat(sizes.begin(), sizes.end() - 2);
    flat.push_back(1);
    flat.push_back(num_keyframes * channels);
    auto blocked = seq.reshape(flat);

    auto expanded = flat;
    expanded[expanded.size() - 2] = num_queries;
    return blocked.expand(expanded);
}

// (..., K) -> (..., Q, K) with stride 0 along Q.  For 1-channel sequences.
at::Tensor as_block_1d(const at::Tensor& seq, int64_t num_queries) {
    auto sizes = seq.sizes().vec();
    std::vector<int64_t> flat(sizes.begin(), sizes.end());
    flat.insert(flat.end() - 1, 1);
    auto blocked = seq.reshape(flat);
    auto expanded = flat;
    expanded[expanded.size() - 2] = num_queries;
    return blocked.expand(expanded);
}

// A stand-in operand for something the caller did not supply.  One element,
// expanded to the iteration shape, never dereferenced by the kernel.
at::Tensor dummy_like(
    const std::vector<int64_t>& batch_query_shape,
    int64_t last_dim,
    const at::TensorOptions& options
) {
    std::vector<int64_t> ones(batch_query_shape.size(), 1);
    ones.push_back(last_dim);
    auto sizes = batch_query_shape;
    sizes.push_back(last_dim);
    // empty, not zeros: the kernel never dereferences these, and a device-side
    // fill would cost more than the whole kernel at small problem sizes.
    return at::empty(ones, options).expand(sizes);
}

}  // namespace


// The kernels find a row of an operand as its base pointer plus the row index
// times the row size, and load whole vectors from it. That works when the row
// (the last `row_dims` dimensions) is stored contiguously, every batch stride is
// a multiple of the row size (or 0, for a broadcast operand) and the data is
// aligned for the vector type. Any other layout is copied into one that is.
at::Tensor kernel_layout(const at::Tensor& t, int64_t row_dims, int64_t alignment) {
    bool fits = t.stride(-1) == 1;
    int64_t row_size = 1;
    for (int64_t d = t.dim() - 1; d >= t.dim() - row_dims; --d) {
        fits &= t.size(d) == 1 || t.stride(d) == row_size;
        row_size *= t.size(d);
    }
    for (int64_t d = 0; d < t.dim() - row_dims; ++d) {
        fits &= t.size(d) == 1 || t.stride(d) == 0 || (row_size > 0 && t.stride(d) % row_size == 0);
    }
    fits &= reinterpret_cast<uintptr_t>(t.data_ptr()) % alignment == 0;
    return fits ? t : t.clone(at::MemoryFormat::Contiguous);
}

// Alignment of a quaternion in memory: the kernels load it as one vec4.
int64_t quat_alignment(const at::Tensor& t) {
    return t.scalar_type() == at::kDouble ? alignof(Float64::vec4) : alignof(Float32::vec4);
}

std::tuple<
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>,
    std::optional<at::Tensor>, std::optional<at::Tensor>
>
trajectory_fwd(
    const at::Tensor& query_times,          // (..., Q)
    const at::Tensor& seq_times_in,         // (..., K)
    const std::optional<at::Tensor>& seq_quats_in,   // (..., K, 4)
    const std::optional<at::Tensor>& seq_t_in,       // (..., K, 3)
    const std::optional<at::Tensor>& seq_mask_in,    // (..., K)
    bool use_only_valid_keyframes,
    bool extrapolate,
    int64_t extrapolation_window,
    bool mask_op_and,
    bool return_velocities,
    bool return_indices,
    bool return_mask
) {
    TORCH_CHECK(query_times.dim() >= 1, "query_times must have at least 1 dimension");
    TORCH_CHECK(seq_times_in.dim() >= 1, "seq_times must have at least 1 dimension");
    TORCH_CHECK(seq_times_in.size(-1) > 0, "seq_times must have at least one keyframe");
    TORCH_CHECK(extrapolation_window >= 1, "extrapolation_window must be at least 1, got ", extrapolation_window);
    // The kernels read every operand with the dtype of seq_times.
    const auto dtype = seq_times_in.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kDouble, "seq_times must be float32 or float64");
    TORCH_CHECK(query_times.scalar_type() == dtype, "query_times must have the dtype of seq_times");
    TORCH_CHECK(!seq_quats_in || seq_quats_in->scalar_type() == dtype, "seq_quats must have the dtype of seq_times");
    TORCH_CHECK(!seq_t_in || seq_t_in->scalar_type() == dtype, "seq_t must have the dtype of seq_times");
    TORCH_CHECK(!seq_mask_in || seq_mask_in->scalar_type() == at::kBool, "seq_mask must be a bool tensor");

    const auto seq_times = kernel_layout(seq_times_in, 1, seq_times_in.element_size());
    const std::optional<at::Tensor> seq_quats = seq_quats_in
        ? std::optional(kernel_layout(*seq_quats_in, 2, quat_alignment(*seq_quats_in))) : std::nullopt;
    const std::optional<at::Tensor> seq_t = seq_t_in
        ? std::optional(kernel_layout(*seq_t_in, 2, seq_t_in->element_size())) : std::nullopt;
    const std::optional<at::Tensor> seq_mask = seq_mask_in
        ? std::optional(kernel_layout(*seq_mask_in, 1, 1)) : std::nullopt;

    const int64_t num_queries = query_times.size(-1);
    const int64_t num_keyframes = seq_times.size(-1);

    // Broadcast the batch dimensions of every operand against each other.
    auto batch_shape = query_times.sizes().vec();
    batch_shape.pop_back();
    auto seq_batch = seq_times.sizes().vec();
    seq_batch.pop_back();
    batch_shape = at::infer_size(batch_shape, seq_batch);

    auto iter_shape = batch_shape;              // (..., Q)
    iter_shape.push_back(num_queries);

    auto float_opts = seq_times.options();
    auto q_times = query_times.expand(iter_shape).unsqueeze(-1);   // (..., Q, 1)

    auto seq_times_b = as_block_1d(
        seq_times.expand([&]{ auto s = batch_shape; s.push_back(num_keyframes); return s; }()),
        num_queries);

    const bool has_quats = seq_quats.has_value();
    const bool has_t = seq_t.has_value();
    const bool has_mask = seq_mask.has_value();

    // Outputs.  Anything not asked for gets a one-element dummy.
    std::optional<at::Tensor> out_quat, out_t, out_mask, out_v, out_w, out_indices;

    // An unrequested output is a one-element tensor expanded to the iteration
    // shape, exactly like an absent input.  TensorIterator normally rejects a
    // stride-0 output, so the iterator below is built with the memory-overlap
    // check disabled; that is sound here only because the matching want_* flag
    // keeps the kernel from ever writing to it.  This avoids allocating (and
    // touching) full-size buffers nobody asked for.
    auto make_out = [&](bool wanted, int64_t channels, const at::TensorOptions& opts) {
        auto sizes = iter_shape;
        sizes.push_back(channels);
        return wanted ? at::empty(sizes, opts)
                      : dummy_like(iter_shape, channels, opts);
    };

    auto out_quat_t = make_out(has_quats, 4, float_opts);
    auto out_t_t = make_out(has_t, 3, float_opts);
    auto out_mask_t = make_out(return_mask, 1, float_opts.dtype(at::kBool));
    auto out_v_t = make_out(return_velocities && has_t, 3, float_opts);
    auto out_w_t = make_out(return_velocities && has_quats, 3, float_opts);
    auto out_idx_t = make_out(return_indices, 2, float_opts.dtype(at::kInt));

    auto seq_quats_b = has_quats
        ? as_block(seq_quats->expand([&]{ auto s = batch_shape; s.push_back(num_keyframes); s.push_back(4); return s; }()),
                   num_queries, num_keyframes)
        : dummy_like(iter_shape, 4, float_opts);
    auto seq_t_b = has_t
        ? as_block(seq_t->expand([&]{ auto s = batch_shape; s.push_back(num_keyframes); s.push_back(3); return s; }()),
                   num_queries, num_keyframes)
        : dummy_like(iter_shape, 3, float_opts);
    auto seq_mask_b = has_mask
        ? as_block_1d(seq_mask->expand([&]{ auto s = batch_shape; s.push_back(num_keyframes); return s; }()),
                      num_queries)
        : dummy_like(iter_shape, 1, float_opts.dtype(at::kBool));


    const bool masking_active = use_only_valid_keyframes && has_mask;
    at::Tensor extr_b;
    if (masking_active) {
        // One contiguous row per batch element: extrapolate_indices walks it
        // with a plain pointer, so a broadcast mask must be materialized.
        auto mask_flat = seq_mask->expand(
            [&]{ auto s = batch_shape; s.push_back(num_keyframes); return s; }()
        ).reshape({-1, num_keyframes}).contiguous();
        auto extr = extrapolate_indices(mask_flat, static_cast<uint>(extrapolation_window));
        auto extr_shape = batch_shape;
        extr_shape.push_back(1);
        extr_shape.push_back(4);
        extr_b = extr.reshape(extr_shape).expand(
            [&]{ auto s = batch_shape; s.push_back(num_queries); s.push_back(4); return s; }());
    } else {
        extr_b = dummy_like(iter_shape, 4, float_opts.dtype(at::kInt));
    }

    TrajParams params{
        static_cast<uint>(num_keyframes),
        static_cast<uint>(extrapolation_window),
        extrapolate,
        use_only_valid_keyframes && has_mask,
        masking_active,
        mask_op_and,
        has_quats, has_t, has_mask,
        return_mask,
        return_velocities && has_t,
        return_velocities && has_quats,
        return_indices
    };

    auto iter = make_tensor_iterator(
        {out_quat_t, out_t_t, out_mask_t, out_v_t, out_w_t, out_idx_t},
        {q_times, seq_times_b, seq_quats_b, seq_t_b, seq_mask_b, extr_b},
        {-1}, /*check_same_dtype=*/false, /*check_mem_overlap=*/false);

    AT_DISPATCH_FLOATING_TYPES(seq_times.scalar_type(), "trajectory_fwd", [&]() {
        using dtype = dtype_t<scalar_t>;
        using vec3 = typename dtype::vec3;
        // Each query does a binary search over K keyframes, far more work than a
    // typical elementwise op, so the CPU grain must be small enough to actually
    // spread across cores (ATen's 32768 default caps it at a couple of threads).
    constexpr int CPU_GRAIN = 512;
    // Register-heavy kernel (binary search + slerp + velocity math): the
        // default 1024 threads/block overflows the register file, especially in
        // double precision ("too many resources requested for launch").
        constexpr int threads = std::is_same_v<scalar_t, float> ? 256 : 128;
        launch_trajectory_fwd<threads, CPU_GRAIN, std::tuple<TrajParams>,
            quat<dtype>*, vec3*, bool*, vec3*, vec3*, int2*,
            const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const bool*,
            const int4*
        >(iter, std::make_tuple(params));
    });

    if (has_quats) out_quat = out_quat_t;
    if (has_t) out_t = out_t_t;
    if (return_mask) out_mask = out_mask_t.squeeze(-1);
    if (params.want_v) out_v = out_v_t;
    if (params.want_w) out_w = out_w_t;
    if (return_indices) out_indices = out_idx_t;

    return {out_quat, out_t, out_mask, out_v, out_w, out_indices};
}




struct TrajBwdParams {
    uint num_keyframes;
    bool has_quats;
    bool has_t;
    bool has_grad_v;
    bool has_grad_w;
};


DEFINE_KERNEL_CAPTURE(launch_trajectory_bwd, {
    const auto [d_query_times, d_seq_times, d_seq_quats, d_seq_t,
                query_times, seq_times, seq_quats, seq_t, ref_indices,
                grad_quat, grad_t, grad_v, grad_w] = ptrs;
    using T = std::remove_const_t<std::remove_pointer_t<decltype(query_times)>>;
    using dtype = dtype_t<T>;
    using vec3_t = typename dtype::vec3;
    using vec4_t = typename dtype::vec4;

    const auto& params = std::get<0>(captures);
    const uint K = params.num_keyframes;

    const int2 ref = ref_indices[offsets[8]];

    if (ref.x < 0 || ref.y < 0) {
        d_query_times[offsets[0]] = T(0);
    } else {
        const T* time_base = seq_times + K * offsets[5];
        T* d_time_base = d_seq_times + K * offsets[1];

        const T query_time = query_times[offsets[4]];
        const T left_time = time_base[ref.x];
        const T right_time = time_base[ref.y];
        const T delta_time = right_time - left_time;
        const bool valid_delta = delta_time > dtype::eps;
        const T rel_time = valid_delta ? (query_time - left_time) / delta_time : T(0.5);
        // Only used when valid_delta, so the gap is above eps and needs no clamp.
        // Clamping the square to eps distorted gradients for gaps below sqrt(eps).
        const T delta_time2 = delta_time * delta_time;

        T d_rel_time = T(0);
        T d_delta_time = T(0);

        if (params.has_quats) {
            const auto* q_base = seq_quats + K * offsets[6];
            auto* dq_base = d_seq_quats + K * offsets[2];
            const auto q_left = q_base[ref.x];
            const auto q_right = q_base[ref.y];

            auto [d_q_left, d_q_right, d_rel_cur] = quat<dtype>::slerp_backward(
                q_left.q, q_right.q, rel_time, grad_quat[offsets[9]]);
            d_rel_time += d_rel_cur;

            if (valid_delta && params.has_grad_w) {
                const auto [dql, dqr, d_dt] = angular_velocity_short_arc_backward(
                    q_left, q_right, delta_time, grad_w[offsets[12]]);
                d_delta_time += d_dt;
                d_q_left += dql;
                d_q_right += dqr;
            }

            atomicAdd<dtype>(dq_base + ref.x, d_q_left);
            atomicAdd<dtype>(dq_base + ref.y, d_q_right);
        }

        if (params.has_t) {
            const auto* t_base = seq_t + K * offsets[7];
            auto* dt_base = d_seq_t + K * offsets[3];
            const vec3_t t_left = t_base[ref.x];
            const vec3_t t_right = t_base[ref.y];

            auto [d_t_left, d_t_right, d_rel_cur] = lerp_backward(
                t_left, t_right, rel_time, grad_t[offsets[10]]);
            d_rel_time += d_rel_cur;

            if (valid_delta && params.has_grad_v) {
                // v = (t_right - t_left) / delta_time
                const vec3_t d_v = grad_v[offsets[11]];
                const vec3_t diff = t_right - t_left;
                d_t_left -= d_v / delta_time;
                d_t_right += d_v / delta_time;
                d_delta_time -= dot(d_v, diff) / delta_time2;
            }

            atomicAdd<dtype>(dt_base + ref.x, d_t_left);
            atomicAdd<dtype>(dt_base + ref.y, d_t_right);
        }

        T d_query_time = T(0);
        T d_left_time = T(0);
        T d_right_time = T(0);

        if (valid_delta) {
            // rel_time = (query_time - left_time) / delta_time
            d_query_time = d_rel_time / delta_time;
            d_left_time -= d_rel_time / delta_time;
            d_delta_time -= d_rel_time * (query_time - left_time) / delta_time2;
        }

        // delta_time = right_time - left_time
        d_right_time += d_delta_time;
        d_left_time -= d_delta_time;

        d_query_times[offsets[0]] = d_query_time;

        toast_atomic_add(d_time_base + ref.x, d_left_time);
        toast_atomic_add(d_time_base + ref.y, d_right_time);
    }
})


std::tuple<at::Tensor, at::Tensor,
           std::optional<at::Tensor>, std::optional<at::Tensor>>
trajectory_bwd(
    const at::Tensor& query_times,        // (..., Q)
    const at::Tensor& seq_times_in,       // (..., K)
    const std::optional<at::Tensor>& seq_quats_in,
    const std::optional<at::Tensor>& seq_t_in,
    const at::Tensor& ref_indices,        // (..., Q, 2) from the forward
    const std::optional<at::Tensor>& grad_quat_in,
    const std::optional<at::Tensor>& grad_t_in,
    const std::optional<at::Tensor>& grad_v_in,
    const std::optional<at::Tensor>& grad_w_in
) {
    // Gradients arrive in whatever layout autograd produced, e.g. slices of a
    // concatenation, so they get the same treatment as the inputs.
    const auto layout = [](const std::optional<at::Tensor>& t, int64_t row_dims, int64_t alignment) {
        return t ? std::optional(kernel_layout(*t, row_dims, alignment)) : std::nullopt;
    };
    const auto elem = seq_times_in.element_size();
    const auto seq_times = kernel_layout(seq_times_in, 1, elem);
    const auto seq_quats = layout(seq_quats_in, 2, quat_alignment(seq_times_in));
    const auto seq_t = layout(seq_t_in, 2, elem);
    const auto grad_quat = layout(grad_quat_in, 1, quat_alignment(seq_times_in));
    const auto grad_t = layout(grad_t_in, 1, elem);
    const auto grad_v = layout(grad_v_in, 1, elem);
    const auto grad_w = layout(grad_w_in, 1, elem);

    const int64_t num_queries = query_times.size(-1);
    const int64_t num_keyframes = seq_times.size(-1);

    auto batch_shape = query_times.sizes().vec();
    batch_shape.pop_back();
    auto seq_batch = seq_times.sizes().vec();
    seq_batch.pop_back();
    batch_shape = at::infer_size(batch_shape, seq_batch);

    auto iter_shape = batch_shape;
    iter_shape.push_back(num_queries);

    auto opts = seq_times.options();
    auto with = [&](std::initializer_list<int64_t> tail) {
        auto s = batch_shape;
        for (auto d : tail) s.push_back(d);
        return s;
    };

    const bool has_quats = seq_quats.has_value();
    const bool has_t = seq_t.has_value();
    const bool has_grad_v = grad_v.has_value();
    const bool has_grad_w = grad_w.has_value();

    // Gradient accumulators: real storage of keyframe shape, zeroed, then viewed
    // as a per-batch block expanded across the query axis.
    auto d_seq_times_store = at::zeros(with({num_keyframes}), opts);
    auto d_seq_quats_store = has_quats ? at::zeros(with({num_keyframes, 4}), opts)
                                       : at::Tensor();
    auto d_seq_t_store = has_t ? at::zeros(with({num_keyframes, 3}), opts)
                               : at::Tensor();

    auto d_query_times = at::empty(
        [&]{ auto s = iter_shape; s.push_back(1); return s; }(), opts);

    auto d_seq_times_b = as_block_1d(d_seq_times_store, num_queries);
    auto d_seq_quats_b = has_quats ? as_block(d_seq_quats_store, num_queries, num_keyframes)
                                   : dummy_like(iter_shape, 4, opts);
    auto d_seq_t_b = has_t ? as_block(d_seq_t_store, num_queries, num_keyframes)
                           : dummy_like(iter_shape, 3, opts);

    auto q_times = query_times.expand(iter_shape).unsqueeze(-1);
    auto seq_times_b = as_block_1d(seq_times.expand(with({num_keyframes})), num_queries);
    auto seq_quats_b = has_quats
        ? as_block(seq_quats->expand(with({num_keyframes, 4})), num_queries, num_keyframes)
        : dummy_like(iter_shape, 4, opts);
    auto seq_t_b = has_t
        ? as_block(seq_t->expand(with({num_keyframes, 3})), num_queries, num_keyframes)
        : dummy_like(iter_shape, 3, opts);

    auto grad_quat_t = grad_quat.has_value() ? *grad_quat : dummy_like(iter_shape, 4, opts);
    auto grad_t_t = grad_t.has_value() ? *grad_t : dummy_like(iter_shape, 3, opts);
    auto grad_v_t = has_grad_v ? *grad_v : dummy_like(iter_shape, 3, opts);
    auto grad_w_t = has_grad_w ? *grad_w : dummy_like(iter_shape, 3, opts);

    TrajBwdParams params{
        static_cast<uint>(num_keyframes), has_quats, has_t, has_grad_v, has_grad_w
    };

    auto iter = make_tensor_iterator(
        {d_query_times, d_seq_times_b, d_seq_quats_b, d_seq_t_b},
        {q_times, seq_times_b, seq_quats_b, seq_t_b, ref_indices,
         grad_quat_t, grad_t_t, grad_v_t, grad_w_t},
        {-1}, /*check_same_dtype=*/false, /*check_mem_overlap=*/false);

    AT_DISPATCH_FLOATING_TYPES(seq_times.scalar_type(), "trajectory_bwd", [&]() {
        using dtype = dtype_t<scalar_t>;
        using vec3 = typename dtype::vec3;
        using vec4 = typename dtype::vec4;
        constexpr int threads = std::is_same_v<scalar_t, float> ? 256 : 128;
        constexpr int CPU_GRAIN = 512;
        launch_trajectory_bwd<threads, CPU_GRAIN, std::tuple<TrajBwdParams>,
            scalar_t*, scalar_t*, vec4*, vec3*,
            const scalar_t*, const scalar_t*, const quat<dtype>*, const vec3*,
            const int2*, const vec4*, const vec3*, const vec3*, const vec3*
        >(iter, std::make_tuple(params));
    });

    std::optional<at::Tensor> d_quats, d_t;
    if (has_quats) d_quats = d_seq_quats_store;
    if (has_t) d_t = d_seq_t_store;
    return {d_query_times.squeeze(-1), d_seq_times_store, d_quats, d_t};
}

void bind_trajectory(py::module_& m) {
    m.def("trajectory_fwd", &trajectory_fwd,
        py::arg("query_times"), py::arg("seq_times"), py::arg("seq_quats"),
        py::arg("seq_t"), py::arg("seq_mask"), py::arg("use_only_valid_keyframes"),
        py::arg("extrapolate"), py::arg("extrapolation_window"), py::arg("mask_op_and"),
        py::arg("return_velocities"), py::arg("return_indices"), py::arg("return_mask")
    );

    m.def("trajectory_bwd", &trajectory_bwd,
        py::arg("query_times"), py::arg("seq_times"), py::arg("seq_quats"),
        py::arg("seq_t"), py::arg("ref_indices"), py::arg("grad_quat"),
        py::arg("grad_t"), py::arg("grad_v"), py::arg("grad_w")
    );
}
