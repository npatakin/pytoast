#include <vector>

#include <toast/utils/toast_config.h>

#include "math_bind/se3_bind.h"
#include "math_bind/quat_bind.h"
#include "motion_bind/velocities.cuh"
#include "motion_bind/interpolation.cuh"
#include "motion_bind/trajectory.cuh"

#include <torch/extension.h>


std::pair<std::vector<torch::Tensor>, bool> broadcast_args(
    const std::vector<torch::Tensor>& args,
    const std::unordered_set<uint>& mtx_args,
    bool cast_to_feature_contiguous_dims
) {
    bool req_grad = false;
    std::vector<uint> num_feature_dims(args.size(), 1);
    std::vector<bool> cast_to_contiguous(args.size(), false);

    uint num_batch_dims = 0;
    for (uint i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        req_grad = req_grad || arg.requires_grad();
        const auto arg_dim = arg.dim();
        if (arg_dim < 1) {
            throw std::runtime_error(
                "Argument '"+std::to_string(i)+
                "' must be at least 1-dimensional tensor");
        }
        cast_to_contiguous[i] = arg.stride(arg_dim - 1) != 1;

        if (mtx_args.find(i) != mtx_args.end()) {
            num_feature_dims[i] = 2;
            if (arg_dim < 2) {
                throw std::runtime_error(
                    "Argument '"+std::to_string(i)+
                    "' must be at least 2-dimensional tensor");
            }
            cast_to_contiguous[i] = (
                cast_to_contiguous[i] || (arg.stride(arg_dim - 2) != arg.size(-1)));
        }
        cast_to_contiguous[i] = cast_to_contiguous[i] && cast_to_feature_contiguous_dims;

        num_batch_dims = std::max(
            uint(arg_dim - num_feature_dims[i]),
            num_batch_dims
        );
    }
    std::vector<long> target_shape;
    target_shape.reserve(num_batch_dims + 2);
    target_shape.resize(num_batch_dims, 1);

    for (uint i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        const auto arg_dim = arg.dim();
        const auto feat_dim = num_feature_dims[i];
        const auto cur_batch_dim = arg_dim - feat_dim;
        const auto offset = num_batch_dims - cur_batch_dim;

        for (uint j = 0; j < cur_batch_dim; ++j) {
            const auto cur_target_shape = target_shape[offset + j];
            const auto cur_size = arg.size(j);
            if (
                (cur_target_shape != 1) &&
                (cur_size != 1) &&
                (cur_target_shape != cur_size)
            ) {
                throw std::runtime_error(
                    "Could not broadcast: argument " + std::to_string(i)
                    + ", dim " + std::to_string(j));
            }
            if (cur_target_shape == 1) {
                target_shape[offset + j] = cur_size;
            }
        }
    }

    std::vector<torch::Tensor> results;
    results.reserve(args.size());

    for (uint i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        const auto cur_feature_dim = num_feature_dims[i];
        target_shape.resize(num_batch_dims + cur_feature_dim);
        target_shape.back() = arg.size(-1);
        if (cur_feature_dim == 2) {
            target_shape[num_batch_dims] = arg.size(-2);
        }

        // set feature dims
        results.push_back(torch::broadcast_to(
            cast_to_contiguous[i] ? arg.contiguous() : arg, target_shape));
    }

    return {results, req_grad};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

#ifdef TOAST_WITH_CUDA
    m.attr("built_with_cuda") = true;
#else
    m.attr("built_with_cuda") = false;
#endif

    bind_quat(m);
    bind_se3(m);
    bind_velocities(m);
    bind_interpolation(m);
    bind_trajectory(m);

    m.def("broadcast_args",
        broadcast_args,
        py::arg("args"),
        py::arg("mtx_args") = std::unordered_set<uint>{},
        py::arg("to_feature_contiguous")= true
    );
}
