#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ["numpy >= 1.11", "pandas >= 0.18.0", "scipy"]
TESTS_REQUIRE = ["pytest >= 2.7.1"]

setup(
    name="ripple_detection",
    version="1.5.0",
    license="MIT",
    description=(
        "Tools for finding sharp-wave ripple events (150-250 Hz) "
        "from local field potentials."
    ),
    author="Eric Denovellis",
    author_email="edeno@bu.edu",
    url="https://github.com/Eden-Kramer-Lab/ripple_detection",
    packages=find_packages(),
    package_data={"": ["*.mat"]},
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
