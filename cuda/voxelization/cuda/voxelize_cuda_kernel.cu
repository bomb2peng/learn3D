#include <iostream>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

namespace{
template <typename scalar_t>
__global__ void sub1_cuda_kernel(
        const scalar_t*  faces,
        size_t bs,
        size_t nf,
        int vs,
        int32_t* voxels
        ) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bs * vs * vs) {
        return;
    }

    int y = i % vs;
    int x = (i / vs) % vs;
    int bn = i / (vs * vs);

    //
    for (int fn = 0; fn < nf; fn++){
        const float* face = (float*)&faces[(bn * nf + fn) * 9];
        float y1d = face[3] - face[0];
        float x1d = face[4] - face[1];
        float z1d = face[5] - face[2];
        float y2d = face[6] - face[0];
        float x2d = face[7] - face[1];
        float z2d = face[8] - face[2];
        float ypd = y - face[0];
        float xpd = x - face[1];
        float det = x1d * y2d - x2d * y1d;
        if (det == 0) continue;
        float t1 = (y2d * xpd - x2d * ypd) / det;
        float t2 = (-y1d * xpd + x1d * ypd) / det;
        if (t1 < 0) continue;
        if (t2 < 0) continue;
        if (1 < t1 + t2) continue;
        int zi = floor(t1 * z1d + t2 * z2d + face[2]);

        int yi, xi;
        yi = y;
        xi = x;
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
        yi = y - 1;
        xi = x;
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
        yi = y;
        xi = x - 1;
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
        yi = y - 1;
        xi = x - 1;
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
    }
}

template <typename scalar_t>
__global__ void sub2_cuda_kernel(
        const scalar_t*  faces,
        size_t bs,
        size_t nf,
        int vs,
        int32_t* voxels
        ) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bs * nf) {
        return;
    }

    int fn = i % nf;
    int bn = i / nf;
    const float* face = (float*)&faces[(bn * nf + fn) * 9];
    for (int k = 0; k < 3; k++) {
        int yi = face[3 * k + 0];
        int xi = face[3 * k + 1];
        int zi = face[3 * k + 2];
        if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
            voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
    }
}

template <typename scalar_t>
__global__ void sub3_1_cuda_kernel(
        size_t bs,
        int vs,
        const scalar_t* voxels,
        scalar_t* visible
        ) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bs * vs * vs * vs) {
        return;
    }

    int z = i % vs;
    int x = (i / vs) % vs;
    int y = (i / (vs * vs)) % vs;
    int bn = i / (vs * vs * vs);
    int pn = i;
    if ((y == 0) || (y == vs - 1) || (x == 0) || (x == vs - 1) || (z == 0) || (z == vs - 1)) {
        if (voxels[pn] == 0) visible[pn] = 1;
    }
}

template <typename scalar_t>
__global__ void sub3_2_cuda_kernel(
        size_t bs,
        int vs,
        const scalar_t* voxels,
        scalar_t* visible
        ) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bs * vs * vs * vs) {
        return;
    }

    int z = i % vs;
    int x = (i / vs) % vs;
    int y = (i / (vs * vs)) % vs;
    int bn = i / (vs * vs * vs);
    int pn = i;
    if ((y == 0) || (y == vs - 1) || (x == 0) || (x == vs - 1) || (z == 0) || (z == vs - 1)) return;
    if (voxels[pn] == 0 && visible[pn] == 0) {
        int yi, xi, zi;
        yi = y - 1;
        xi = x;
        zi = z;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y + 1;
        xi = x;
        zi = z;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y;
        xi = x - 1;
        zi = z;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y;
        xi = x + 1;
        zi = z;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y;
        xi = x;
        zi = z - 1;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
        yi = y;
        xi = x;
        zi = z + 1;
        if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
    }
}

}


at::Tensor sub1_cuda(
        at::Tensor faces,
        int bs,
        int nf,
        int size,
        at::Tensor voxels) {

    const int threads = 1024;
    const dim3 blocks ((bs * size * size - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "sub1_cuda", ([&] {
      sub1_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          bs,
          nf,
          size,
          voxels.data<int32_t>());
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sub1_cuda: %s\n", cudaGetErrorString(err));

    return voxels;
}

at::Tensor sub2_cuda(
        at::Tensor faces,
        int bs,
        int nf,
        int size,
        at::Tensor voxels) {

    const int threads = 1024;
    const dim3 blocks ((bs * nf - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(faces.type(), "sub2_cuda", ([&] {
      sub2_cuda_kernel<scalar_t><<<blocks, threads>>>(
          faces.data<scalar_t>(),
          bs,
          nf,
          size,
          voxels.data<int32_t>());
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sub2_cuda: %s\n", cudaGetErrorString(err));

    return voxels;
}

at::Tensor sub3_1_cuda(
        int bs,
        int size,
        at::Tensor voxels,
        at::Tensor visible){

    const int threads = 1024;
    const dim3 blocks ((bs * size * size * size - 1) / threads + 1);

    AT_DISPATCH_INTEGRAL_TYPES(voxels.type(), "sub3_1_cuda", ([&] {
      sub3_1_cuda_kernel<scalar_t><<<blocks, threads>>>(
          bs,
          size,
          voxels.data<scalar_t>(),
          visible.data<scalar_t>());
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sub3_1_cuda: %s\n", cudaGetErrorString(err));

    return visible;
}

at::Tensor sub3_2_cuda(
        int bs,
        int size,
        at::Tensor voxels,
        at::Tensor visible){

    const int threads = 1024;
    const dim3 blocks ((bs * size * size * size - 1) / threads + 1);

    AT_DISPATCH_INTEGRAL_TYPES(voxels.type(), "sub3_2_cuda", ([&] {
      sub3_2_cuda_kernel<scalar_t><<<blocks, threads>>>(
          bs,
          size,
          voxels.data<scalar_t>(),
          visible.data<scalar_t>());
      }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sub3_2_cuda: %s\n", cudaGetErrorString(err));

    return visible;
}