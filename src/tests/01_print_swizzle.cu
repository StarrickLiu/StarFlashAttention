#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

using namespace cute;

int main(int argc, char *argv[]) {
    auto smem_atom_0 = make_layout(make_shape(Int<8>{}, Int<64>{}),
                                        make_stride(Int<64>{}, Int<1>{}));
    // print_latex(smem_atom_0);
    // printf("\n\n");
    auto smem_atom_1 = composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<64>{}),
                                        make_stride(Int<64>{}, Int<1>{})));
    // print_latex(smem_atom_1);
    // printf("\n\n");
    auto smem_atom_2 = composition(
        Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<64>{}),
                                        make_stride(Int<64>{}, Int<1>{})));
    // print_latex(smem_atom_1);

    auto SmemLayoutAtomQ =
        composition(Swizzle<3, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<64>>, // (8, 64)
                           Stride<Int<64>, _1>>{}); // (64, 1)
    auto SmemLayoutQ = tile_to_shape(
        SmemLayoutAtomQ,
        Shape<Int<64>, Int<128>>{}); // (64, 128)

    auto SmemLayoutKV = tile_to_shape(
        SmemLayoutAtomQ,
        Shape<Int<128>, Int<128>>{}); // (128, 128)

    // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
    auto SmemLayoutVtransposed = composition(SmemLayoutKV,
        make_layout(Shape<Int<128>, Int<128>>{}, GenRowMajor{}));

    // print_latex(SmemLayoutQ);
    // print_latex(SmemLayoutKV);
    // print_latex(SmemLayoutVtransposed);
    auto SmemLayoutVtransposedNoSwizzle = get_nonswizzle_portion(SmemLayoutVtransposed);
    print_latex(SmemLayoutVtransposedNoSwizzle);

    static constexpr int kSmemQSize = size(SmemLayoutQ) * sizeof(cutlass::half_t);
    static constexpr int kSmemKVSize = size(SmemLayoutKV) * 2 * sizeof(cutlass::half_t);
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

    printf("\n\nkSmemSize: %d\n\n", kSmemSize);
}