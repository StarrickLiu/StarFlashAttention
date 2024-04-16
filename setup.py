from setuptools import setup, Extension
from torch.utils import cpp_extension
from pathlib import Path

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='star_flash_attn',
    version='0.1.0',  # 适当设置您的版本号
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='star_flash_attn',
            sources=['./src/flash_api.cpp'],  # 这里列出需要编译的源代码文件
            extra_objects=[
                Path(this_dir) / 'build/libStarFlashAttention.so'
            ],  # 这里提供完整路径到你的.so文件
            # include_dirs=['./src'],  # 这里可以添加需要包含的头文件目录
            include_dirs=[
                Path(this_dir) / "src",
                Path(this_dir),
            ],
            library_dirs=[],  # 这里可以添加库文件搜索目录
            libraries=[],  # 这里可以添加需要链接的库名（不包括路径和后缀）
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
