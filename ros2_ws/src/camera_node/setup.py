from setuptools import find_packages, setup

PKG = "camera_node"

setup(
    name=PKG,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{PKG}"]),
        (f"share/{PKG}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cyk",
    maintainer_email="dev@local",
    description="Orbbec Gemini Pro RGB-D bridge with hardware D2C.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_node = camera_node.camera_node:main",
        ],
    },
)
