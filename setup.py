import platform

from setuptools import find_packages, setup, find_namespace_packages


def torch_urls(version):
    platform_system = platform.system()
    if platform_system == "Windows":
        return f"torch@https://download.pytorch.org/whl/cu90/torch-{version}-cp36-cp36m-win_amd64.whl#"
    return f"torch>={version}"


setup(
    name="lib-poolnet",
    version="v0.0.1",
    description="Ready-to-use Saliency object detection under one common API",
    author="Cinnamon AI Lab",
    url="https://github.com/xing1999/PoolNet",
    packages=find_namespace_packages(),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires=[
        torch_urls("1.4"),
        "numpy",
        "pytorch-lightning",
        "scikit-learn",
    ],
    extras_require={"dev": ["black", "pytest"],},
)
