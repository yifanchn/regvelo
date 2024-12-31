# RegVelo: gene-regulatory-informed dynamics of single cells

<img src="https://github.com/theislab/regvelo/blob/main/docs/_static/img/overview_fig.png?raw=true" alt="RegVelo" width="600" />

**RegVelo** is a end-to-end framework to infer regulatory cellular dynamics through coupled splicing dynamics. See our [RegVelo manuscript](https://www.biorxiv.org/content/10.1101/2024.12.11.627935v1) to learn more. If you use our tool in your own work, please cite it as

```
@article{wang2024regvelo,
  title={RegVelo: gene-regulatory-informed dynamics of single cells},
  author={Wang, Weixu and Hu, Zhiyuan and Weiler, Philipp and Mayes, Sarah and Lange, Marius and Wang, Jingye and Xue, Zhengyuan and Sauka-Spengler, Tatjana and Theis, Fabian J},
  journal={bioRxiv},
  pages={2024--12},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
## Getting started
Please refer to the [Tutorials](https://regvelo.readthedocs.io/en/latest/index.html)

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
