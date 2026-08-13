#ifndef TENSOR_HELPERS_H
#define TENSOR_HELPERS_H

#include <torch/torch.h>


// Checks the last dimension size of each TensorIterator
// operand (outputs first, then inputs).
// Pass -1 to skip the check for a specific operand.
inline void check_operand_last_dims(
    const at::TensorIterator& iter,
    std::initializer_list<int64_t> expected
) {
    TORCH_CHECK(
        (int)expected.size() == iter.ntensors(),
        "check_operand_last_dims: ", expected.size(),
        " sizes given but iterator has ",
        iter.ntensors(), " operands");
    int i = 0;
    for (const int64_t exp : expected) {
        if (exp >= 0) {
            const int64_t actual = iter.tensor(i).size(-1);
            TORCH_CHECK(actual == exp,
                "operand[", i, "] last dim must be ", exp, ", got ", actual);
        }
        ++i;
    }
}

template <typename TensorList, typename LastDimsContainer>
void check_operand_last_dims(
    const TensorList& tensors,
    const LastDimsContainer& expected
) {
    TORCH_CHECK(
        static_cast<int>(expected.size()) == tensors.size(),
        "check_operand_last_dims: ", expected.size(),
        " sizes given but input has ",
        tensors.size(), " operands");

    int i = 0;
    for (const int64_t exp : expected) {
        if (exp >= 0) {
            const int64_t actual = tensors[i].size(-1);
            TORCH_CHECK(actual == exp,
                "input[", i, "] last dim must be ", exp, ", got ", actual);
        }
        ++i;
    }
}


inline bool check_same_dtype(const torch::Tensor& t) {
    return true;
}


template <typename... Tensors>
bool check_same_dtype(
    const torch::Tensor& first,
    const torch::Tensor& second,
    const Tensors&... rest
) {
    if (first.dtype() != second.dtype()) {
        return false;
    }
    return check_same_dtype(first, rest...);
}

inline void check_same_device() {}
inline void check_same_device(const torch::Tensor& t) {}

template <typename... Tensors>
void check_same_device(
    const torch::Tensor& first,
    const torch::Tensor& second,
    const Tensors&... rest
) {
    if (first.device() != second.device()) {
        throw std::runtime_error(
            "Tensors are not the same device: " + first.device().str()
            + " and " + second.device().str());
    }
    check_same_device(first, rest...);
}

template <typename TensorList>
void check_same_device(const TensorList& ts) {
    const auto device = ts.begin()->device();
    for (const auto& t : ts) {
        if (t.device() != device) {
            throw std::runtime_error(
                "Tensors are not the same device: " + device.str()
                + " and " + t.device().str());
        }
    }
}

inline void check_dtypes(const at::ScalarType& dtype, const torch::Tensor& t) {
    if (t.dtype() != dtype) {
        throw std::runtime_error("Dtype mismatch. Expected " +
            std::string(toString(dtype)) +
            ", but given " + std::string(t.dtype().name()));
    }
}

template <typename... Tensors>
void check_dtypes(
    const at::ScalarType& dtype,
    const torch::Tensor& first,
    const Tensors&... rest
) {
    check_dtypes(dtype, first);
    check_dtypes(dtype, rest...);
}

template <typename TensorList>
void check_dtypes(const at::ScalarType& dtype, const TensorList& ts) {
    for (const auto& t : ts) {
        if (t.dtype() != dtype) {
            throw std::runtime_error("Dtype mismatch. Expected " +
                std::string(toString(dtype)) + ", but found: " +
                std::string(t.dtype().name()));
        }
    }
}

inline bool all_contiguous(const torch::Tensor& t) {
    return t.is_contiguous();
}

template <typename... Tensors>
void all_contiguous(const torch::Tensor& first, const Tensors&... rest) {
    if (!all_contiguous(first)) {
        throw std::runtime_error("all input tensors must be contiguous");
    }
    all_contiguous(rest...);
}


template <typename... OutTypes, typename F, typename... Args>
auto map_to_tuple_cast(F&& f, Args&&... args) {
    return std::tuple<OutTypes...>{
        static_cast<OutTypes>(f(std::forward<Args>(args)))...
    };
}


template <typename F, typename... Args>
auto map_to_tuple(F&& f, Args&&... args) {
    return std::make_tuple(f(std::forward<Args>(args))...);
}

template <typename... Args>
auto map_to_pointers(Args&&... args) {
    return map_to_tuple([](const auto& t) {
        return t.data_ptr();
    }, std::forward<Args>(args)...);
}

template <typename... OutTypes, typename ...Args>
auto map_to_pointers_cast(Args&&... args) {
    return map_to_tuple_cast<OutTypes...>([] (const auto& t) {
        return t.data_ptr();
    }, std::forward<Args>(args)...);
}

inline void check_dtype_in(
    const torch::Tensor& t,
    const std::vector<at::ScalarType>& dtypes
) {
    const auto& check_dtype = t.dtype();
    bool matched = false;
    for (const auto& dtype : dtypes) {
        if (dtype == check_dtype) {
            matched = true;
            break;
        }
    }
    if (!matched) {
        std::string allowed_dtypes_str = "";
        for (const auto& dtype : dtypes) {
            allowed_dtypes_str += std::string(toString(dtype)) + ", ";
        }
        throw std::runtime_error(
            "Dtype mismatch. Got: " + std::string(t.dtype().name())
            + " Expected one of: " + allowed_dtypes_str);
    }
};

inline torch::Tensor make_ptr_tensor(const std::vector<torch::Tensor>& tensors) {
    auto ptr_tensor = torch::zeros(
        {static_cast<uint>(tensors.size())},
        at::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto ptr_tensor_ptr = static_cast<char **>(ptr_tensor.data_ptr());
    for (uint i = 0; i != tensors.size(); ++i) {
        ptr_tensor_ptr[i] = static_cast<char *>(tensors[i].data_ptr());
    }
    if (tensors[0].device() != ptr_tensor.device()) {
        ptr_tensor = ptr_tensor.to(tensors[0].device());
    }
    return ptr_tensor;
}



#endif //TENSOR_HELPERS_H
