from setuptools import find_packages, setup


package_name = "recovery_manager"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="Emmanuel Alabi Olasubomi",
    maintainer_email="alabiemmanuel177@gmail.com",
    description="Fail-closed guarded recovery decision manager for Research 2.",
    license="MIT",
    entry_points={
        "console_scripts": ["recovery_manager = recovery_manager.node:main"]
    },
)
