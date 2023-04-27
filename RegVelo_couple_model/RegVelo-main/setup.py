import setuptools

def get_readme():
    with open("README.md", "rt", encoding="utf-8") as fh:
        return fh.read()

setuptools.setup(
    name='RegVelo',
    version='0.1.4',
    author='Weixu Wang',
    author_email='weixu.wang@helmholtz-muenchen.de',
    description='a deep learning method for regulatory cellular dynamics',
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/theislab/RegVelo',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.9.1',
        'torchdiffeq>=0.2.2',
        'numpy>=1.19.2',
        'scanpy>=1.7.1',
        'anndata>=0.7.5',
        'scipy>=1.5.2',
        'tqdm>=4.32.2',
        'scikit-learn>=0.24.1',
        'leidenalg>=0.8.4',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 4 - Beta",
    ],
    python_requires='>=3.7',
)
