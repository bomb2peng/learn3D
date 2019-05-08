#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor sub1_cuda(
        at::Tensor faces,
        int bs,
        int nf,
        int size,
        at::Tensor voxels);

at::Tensor sub2_cuda(
        at::Tensor faces,
        int bs,
        int nf,
        int size,
        at::Tensor voxels);

at::Tensor sub3_1_cuda(
        int bs,
        int size,
        at::Tensor voxels,
        at::Tensor visible);

at::Tensor sub3_2_cuda(
        int bs,
        int size,
        at::Tensor voxels,
        at::Tensor visible);


// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor sub1(
        at::Tensor faces,
        int bs,
        int nf,
        int size,
        at::Tensor voxels) {

    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);

    return sub1_cuda(faces, bs, nf, size, voxels);
}

at::Tensor sub2(
        at::Tensor faces,
        int bs,
        int nf,
        int size,
        at::Tensor voxels) {

    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);

    return sub2_cuda(faces, bs, nf, size, voxels);
}

at::Tensor sub3_1(
        int bs,
        int size,
        at::Tensor voxels,
        at::Tensor visible) {

    CHECK_INPUT(voxels);
    CHECK_INPUT(visible);

    return sub3_1_cuda(bs, size, voxels, visible);
}

at::Tensor sub3_2(
        int bs,
        int size,
        at::Tensor voxels,
        at::Tensor visible) {

    CHECK_INPUT(voxels);
    CHECK_INPUT(visible);

    return sub3_2_cuda(bs, size, voxels, visible);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sub1", &sub1, "sub1 (CUDA)");
    m.def("sub2", &sub2, "sub2 (CUDA)");
    m.def("sub3_1", &sub3_1, "sub3_1 (CUDA)");
    m.def("sub3_2", &sub3_2, "sub3_2 (CUDA)");
}