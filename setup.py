from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'transformers[torch]',
    'torch',
    'datasets',
    'numpy',
    'scikit-learn',
    'pandas',
    'scipy',
    'dict_hash'
]

setuptools.setup(
    name="iprompt",
    version="0.01",
    author="Jack Morris, Chandan Singh",
    author_email="",
    description="iPrompt: Explaining Patterns in Data with Language Models via Interpretable Autoprompting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/interpretable-autoprompting",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    install_requires=required_pypi,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)