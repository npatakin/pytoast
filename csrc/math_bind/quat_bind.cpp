#include "quat_bind.h"

void bind_quat(py::module& m) {
    bind_quat_creation(m);
    bind_quat_from_axis_angle(m);
    bind_quat_to_axis_angle(m);
    bind_quat_to_matrix(m);
    bind_quat_from_matrix(m);
    bind_quat_inverse(m);
    bind_quat_std(m);
    bind_quat_mul(m);
    bind_quat_apply(m);
    bind_quat_slerp(m);
    bind_quat_from_euler_angles(m);
}