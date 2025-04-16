# Splitting the Forward-Backward Algorithm: A Full Characterization. Example code

This repository contains the experimental source code to reproduce the numerical experiments in:

* A. Akerman, E. Chenchene, P. Giselsson, E. Naldi. Splitting the Forward-Backward Algorithm: A Full Characterization. 2025. [ArXiv preprint](https://arxiv.org/abs/2504.10999)

To reproduce the results of the numerical experiments run:
```bash
python3 main.py
```
**Note:** To run `experiment_portfolio_optimization` a [stockdata](https://www.stockdata.org/) account is required. Once registered, add your **API key** in line 50 of `portfolio_optimization.py`.

If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@article{acgn25,
  author = {\AA kerman, Antom and Chenchene, Enis and Giselsson, Pontus and Naldi, Emanuele},
  title = {Splitting the Forward-Backward Algorithm: A Full Characterization},
  pages = {2504.10999},
  journal = {ArXiv},
  year = {2025}
}
```

## Requirements

Please make sure to have the following Python modules installed:

* [numpy>=1.26.4](https://pypi.org/project/numpy/)
* [numpy>=1.13.1](https://pypi.org/project/scipy/)
* [matplotlib>=3.3.4](https://pypi.org/project/matplotlib/)
* [tqdm>=4.66.4](https://pypi.org/project/matplotlib/)
* [networkx>=3.2.1](https://pypi.org/project/networkx/)
* [cvxpy>=1.6.2](https://pypi.org/project/cvxpy/)
* [mosek>=11.0.11](https://pypi.org/project/mosek/)


## License
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
