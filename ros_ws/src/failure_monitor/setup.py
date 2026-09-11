from setuptools import find_packages, setup


package_name = "failure_monitor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pyyaml"],
    zip_safe=True,
    maintainer="Emmanuel Alabi Olasubomi",
    maintainer_email="alabiemmanuel177@gmail.com",
    description="Research 2 online failure monitor: frozen predictor on deployable topics.",
    license="MIT",
    entry_points={
        "console_scripts": ["failure_monitor = failure_monitor.node:main"]
    },
)
