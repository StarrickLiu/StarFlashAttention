#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

int main(int argc, char *argv[]) {
    static constexpr int kNThreads = 128;
    static constexpr int kGmemThreadsPerRow = 8;
    using Element = cute::half_t;
    using ElementAccum = float;
    using GmemLayoutAtomOaccum = Layout<Shape <_8, _16>,  
                                        Stride< _16, _1>>;
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, // (16, 8)
                                  Stride<Int<kGmemThreadsPerRow>, _1>>; // (8, 1)

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
    // (_16,(_8,_8)):(_8,(_128,_1))
    // GmemTiledCopyRotcossin g1;
    // GmemTiledCopyRotcossinCont g2;
    // print_latex(g2);
    Layout gCosCont = make_layout(Shape<Int<8>, Int<8>>{},
                                  Stride<Int<4>, Int<1>>{}); 
    print_layout(gCosCont);
    // print_latex(gCosCont);
}