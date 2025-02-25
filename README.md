# Generalized Fast Krasnoselskii-Mann Method with Preconditioners: Example code

This repository contains the experimental source code to reproduce the numerical experiments in:

* A. Akerman, E. Chenchene, P. Giselsson, E. Naldi. Characterization of Nonexpansive Forward-Backward-type Algorithms with Minimal Memory Requirements. 2025. [ArXiv preprint](https://arxiv.org/abs/xxxx.yyyyy)

To reproduce the results of the numerical experiments run:
```bash
python3 main.py
```

If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@article{bcf24,
  author = {Åkerman, Antom and Chenchene, Enis and Giselsson, Pontus and Naldi, Emanuele},
  title = {Characterization of Nonexpansive Forward--Backward-type Algorithms with Minimal Memory Requirements},
  pages = {xxxx.yyyyy},
  journal = {ArXiv},
  year = {2024}
}
```

## Requirements

Please make sure to have the following Python modules installed, most of which should be standard.

* [numpy>=1.20.1](https://pypi.org/project/numpy/)
* [matplotlib>=3.3.4](https://pypi.org/project/matplotlib/)
* [tqdm>=4.66.4](https://pypi.org/project/matplotlib/)

## Acknowledgments


* The research of EC has been supported by the Austrian Science Fund (FWF), project P 34922-N.
## License
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
