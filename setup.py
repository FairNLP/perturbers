#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

REQUIREMENTS = ["datasets", "transformers", "torch"]

TEST_REQUIREMENTS = ["pytest"]
QUALITY_REQUIREMENTS = ["black", "ruff"]
TRAINING_REQUIREMENTS = ["lightning", "wandb"]

EXTRAS_REQUIREMENTS = {
    "dev": TEST_REQUIREMENTS + QUALITY_REQUIREMENTS,
    "test": TEST_REQUIREMENTS,
    "training": TRAINING_REQUIREMENTS,
}
setup(
    author="FairNLP",
    author_email="info@fairnlp.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Perturber models for neural data augmentation.",
    install_requires=REQUIREMENTS,
    license="Apache Software License 2.0",
    include_package_data=True,
    keywords="perturbers",
    name="perturbers",
    packages=find_packages(include=["perturbers", "perturbers.*"]),
    test_suite="tests",
    tests_require=TEST_REQUIREMENTS,
    extras_require=EXTRAS_REQUIREMENTS,
    url="https://github.com/FairNLP/perturbers",
    version="0.0.0",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    zip_safe=False,
)
