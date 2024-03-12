#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

struct traits {
    static constexpr int kNThreads = 128;
    static constexpr int kGmemThreadsPerRow = 8;
    using Element = cute::half_t;
    using ElementAccum = float;
    using elem_type = cutlass::half_t;

    using GmemLayoutAtomOaccum = Layout<Shape <_8, _16>,  
                                        Stride< _16, _1>>;
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, // (16, 8)
                                  Stride<Int<kGmemThreadsPerRow>, _1>>; // (8, 1)
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint64_t>; 
    // uint64_t或uint128_t不会改变tile的大小和数量，只影响一个线程的拷贝行为是怎么样的
    // 比如要拷贝的元素是fp16时，uint64_t拷贝的行为是拷贝4个fp16元素，uint128_t拷贝的行为是拷贝8个fp16元素
    // 在本例中，uint64_t 的partition_S的Layout是((_4,_2),_8,_2):((_1,_4),2048,_64)
    //          uint128_t的partition_S的Layout是((_8,_1),_8,_2):((_1,_0),2048,_64)
    // 可见uint64_t在一次拷贝执行中，需要重复两次Atom操作，但uint128_t仅需一次
    // 显然，uint128_t的拷贝效率更高
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    // (_16,(_4,_8)):(_8,(_128,_1))
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
};



template <typename traits>
__global__ void print_copytile_partition_s(int* g_mem_ptr) {

    int tidx = threadIdx.x;

    using Element = typename traits::Element;
    typename traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
    typename traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
    typename traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
    auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
    Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(g_mem_ptr) + 0),
                                Shape<Int<128>, Int<128>>{},
                                make_stride(128, _1{}));
    Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(g_mem_ptr) + sizeof(Element) * 128 * 128),
                                Shape<Int<128>, Int<128>>{},
                                make_stride(128, _1{}));
    Tensor tKgKnew = gmem_thr_copy_QKV.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tVgVnew = gmem_thr_copy_QKV.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)

    if (tidx == 0) {
        // print_tensor(gKnew);
        // printf("\n\n");
        print_tensor(tKgKnew);
    }

}


template <typename traits>
void run_kernel() {
    dim3 grid(1, 1, 1);
    dim3 block(traits::kNThreads);

    int* g_mem_ptr;
    cudaMalloc(&g_mem_ptr, 128 * 128 * 128 * sizeof(int));
    print_copytile_partition_s<traits><<<grid, block>>>(g_mem_ptr);
}

int main(int argc, char *argv[]) {

    run_kernel<traits>();

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}