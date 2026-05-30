from setuptools import find_packages, setup

PKG = "audio_frontend"

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
    description="Far-field audio frontend bridging M260C 8ch -> /audio/clean.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "audio_frontend_node = audio_frontend.audio_frontend_node:main",
        ],
    },
)
