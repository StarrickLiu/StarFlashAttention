# 找出src目录下的所有.cu文件，包括子目录
CC_FILES := $(shell find ./src/ -name "*.cu")
# 将找到的.cu文件路径转换为build目录下的目标路径
EXE_FILES := $(patsubst ./src/%.cu,./build/%,$(CC_FILES))

# 默认目标：编译所有
all: $(EXE_FILES)

# 为每个目标确保目录存在
./build/%: ./src/%.cu
	@mkdir -p $(dir $@)
	nvcc -o $@ $< -O2 -arch=sm_86 -std=c++17 -I3rd/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas

# 清理构建文件
clean:
	rm -rf ./build/