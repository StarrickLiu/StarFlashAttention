CC_FILES=$(shell find ./src/ -name "*.cc")
CU_FILES=$(shell find ./src/ -name "*.cu")
CPP_FILES=$(shell find ./examples/cpp/ -name "*.cc")
LIB_NAME=libStarFlashAttention.so
LIB_PATH=./build/$(LIB_NAME)
EXECUTABLES=$(CPP_FILES:./examples/cpp/%.cc=./build/%)
BUILD_DIR=./build
OBJ_DIR=$(BUILD_DIR)/obj
CU_OBJ=$(CU_FILES:./src/%.cu=$(OBJ_DIR)/%.o)
DEVICE_LINK_FILE=$(OBJ_DIR)/device_link.o

all: build $(LIB_PATH) $(EXECUTABLES)

build:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(OBJ_DIR)

# Compile flash_attn.cu, assuming utils.cu functions are in utils.h now
$(OBJ_DIR)/flash_attn.o: ./src/flash_attn.cu
	@mkdir -p $(@D) # Ensure the directory exists
	nvcc -c $< -o $@ -O2 -m64 -gencode arch=compute_86,code=sm_86 -std=c++17 -I. --expt-relaxed-constexpr --compiler-options '-fPIC' -rdc=true

# Device link step
$(DEVICE_LINK_FILE): $(CU_OBJ)
	nvcc -dlink $^ -o $@ -gencode arch=compute_86,code=sm_86 -lcublasLt -lcublas --compiler-options '-fPIC' -rdc=true

# Link all object files to a single shared library
$(LIB_PATH): $(CU_OBJ) $(CC_FILES) $(DEVICE_LINK_FILE)
	nvcc -o $@ $^ -shared -gencode arch=compute_86,code=sm_86 -lcublasLt -lcublas

# Compile each C++ file in examples/cpp using the shared library
$(EXECUTABLES): ./build/% : ./examples/cpp/%.cc $(LIB_PATH)
	nvcc -o $@ $< -O2 -m64 -gencode arch=compute_86,code=sm_86 -std=c++17 -I. -I./src -L./build -lnvToolsExt -lStarFlashAttention -lcublas -lcudart

clean:
	rm -rf $(BUILD_DIR)
