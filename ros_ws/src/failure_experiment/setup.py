from glob import glob
from setuptools import find_packages, setup


package_name = "failure_experiment"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "pyyaml"],
    zip_safe=True,
    maintainer="Emmanuel Alabi Olasubomi",
    maintainer_email="alabiemmanuel177@gmail.com",
    description="Research 2 deterministic fault injection overlay.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "signal_proxy = failure_experiment.signal_proxy:main",
            "environment_fault = failure_experiment.environment_fault:main",
            "event_marker = failure_experiment.event_marker:main",
            "topic_health = failure_experiment.topic_health:main",
            "semantic_summary = failure_experiment.semantic_summary:main",
        ]
    },
)
