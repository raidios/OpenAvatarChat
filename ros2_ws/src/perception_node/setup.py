from setuptools import find_packages, setup

PKG = "perception_node"

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
    description="Pose+track+ReID owner-3D pipeline from RGB-D.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "perception_node = perception_node.perception_node:main",
        ],
    },
)
