# RegVelo: gene-regulatory-informed dynamics of single cells

Inferring regulatory cellular dynamics through coupled splicing dynamics. If you use our tool in your own work, please cite it as

```
@article{wang:24,
    title = {RegVelo: gene-regulatory-informed dynamics of single cells},
    author = {Wang, Weixu and Hu, Zhiyuan and Weiler, Philipp and Mayes, Sarah and Lange, Marius and Wang, Jingye and Xue, Zhengyuan and Sauka-Spengler, Tatjana Theis, Fabian J.},
    doi = {10.1101/2024.12.11.627935},
    url = { https://doi.org/10.1101/2024.12.11.627935},
    year = {2024},
    publisher = {Cold Spring Harbor Laboratory},
}
```

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

To create and activate a new environment

```bash
conda create -n regvelo-py310 python=3.10 --yes && conda activate regvelo-py310
```

Next, install the package with

```bash
pip install git+https://github.com/theislab/regvelo.git@main
```
