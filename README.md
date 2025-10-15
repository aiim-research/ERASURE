<div align="center">

# ERASURE  
### Redefining Privacy Through Selective Machine Unlearning

<img src="ERASURE_LOGO.png" alt="ERASURE Logo" width="200"/>

[![Star us on GitHub](https://img.shields.io/badge/⭐_Star_Us_If_You_Like_It-181717?style=for-the-badge&logo=github)](https://github.com/aiim-research/ERASURE)

</div>

## 📝 Citation

ERASURE was accepted as IJCAI demo and CIKM resource! 

If you use this Machine Unlearning framework, please cite us:

> @inproceedings{ijcai2025p1255,
  title     = {How to Make Reproducible Research in Machine Unlearning with ERASURE},
  author    = {D'Angelo, Andrea and Savelli, Claudio and Tagliente, Gabriele and Giobergia, Flavio and Baralis, Elena and Stilo, Giovanni},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {11025--11029},
  year      = {2025},
  month     = {8},
  note      = {Demo Track},
  doi       = {10.24963/ijcai.2025/1255},
  url       = {https://doi.org/10.24963/ijcai.2025/1255},
}


## 📜 Table of Contents
* [📘 General Information](#general-information)
* [🧭 Guide](#guide)
* [🧪 Example Workflow](#example-workflow)
* [📊 Resources Provided with the Framework](#resources-provided-with-the-framework)
* [Team Information](#team-information)

## 📘 General Information

ERASURE offers fully extensible built-in components, allowing users to define custom unlearning techniques, integrate custom and synthetic datasets, implement tailored evaluation metrics, and meld seamlessly with state-of-the-art machine learning models.
Additionally, it provides a flexible environment for defining and running unlearning experiments, enabling researchers to evaluate the effectiveness and efficiency of various unlearning methods systematically.

The framework is inside the ```erasure``` folder. A JSON configuration file is everything ERASURE needs to run.

You can find examples of configurations in the ```config``` folder. Specifically, ```example.jsonc``` and ```proof_of_concept.jsonc``` are simple configurations designed as a starting point.

## 🧭 Guide

You can find a step-by-step guide on how to use ERASURE in our IJCAI demo paper: https://www.ijcai.org/proceedings/2025/1255

You may also find the presentation in the ```guide``` folder useful to implement your methods in the ERASURE framework. 



## 🧪 Example Workflow

<div align="center">
 <img src="https://i.imgur.com/gcar8Zz.png" alt="worfklow" width="500"/>
</div>

To run an experiment, you just have to run:

```console
foo@bar:~$ python main.py configs/example.jsonc
```

in your terminal, given that you have dependencies installed.


## 📊 Resources Provided with the Framework


### 🗂️ Datasets

You can quickly implement any Dataset you like in ERASURE. At the moment, ERASURE has built-in DataSource classes for the following sources:

![Static Badge](https://img.shields.io/badge/Datasource-TorchVision-blue)

![Static Badge](https://img.shields.io/badge/Datasource-Hugging%20Face-blue)

![Static Badge](https://img.shields.io/badge/Datasource-UCI%20Repository-blue)

![Static Badge](https://img.shields.io/badge/Datasource-PyTorch%20Geometric-blue)


### 🧩 Built-in Unlearners 

Developing your own custom Unlearner in ERASURE is easy. 

However, benchmarking your Unlearner in an unified environment is just as important. Right now, ERASURE comes built-in with the following Unlearners:


| | | |
|:--:|:--:|:--:|
| ![Static Badge](https://img.shields.io/badge/Unlearner-cfk-red) | ![Static Badge](https://img.shields.io/badge/Unlearner-euk-red) | ![Static Badge](https://img.shields.io/badge/Unlearner-SalUn-red) |
| ![Static Badge](https://img.shields.io/badge/Unlearner-UNSIR-red) | ![Static Badge](https://img.shields.io/badge/Unlearner-SCRUB-red) | ![Static Badge](https://img.shields.io/badge/Unlearner-NegGrad-red) |
| ![Static Badge](https://img.shields.io/badge/Unlearner-Finetuning-red) | ![Static Badge](https://img.shields.io/badge/Unlearner-Bad%20Teaching-red) | ![Static Badge](https://img.shields.io/badge/Unlearner-Gold%20Model-red) |
| ![Static Badge](https://img.shields.io/badge/Unlearner-Fisher%20Forgetting-red) | ![Static Badge](https://img.shields.io/badge/Unlearner-Successive%20Random%20Labels-red) | ![Static Badge](https://img.shields.io/badge/Unlearner-Selective%20Synaptic%20Dampening-red) |
| ![Static Badge](https://img.shields.io/badge/Unlearner-Advanced%20NegGrad-red) |  |  |



## Team Information:
* Prof. Giovanni Stilo, PhD. [project leader/research advisor]
* Flavio Giobergia, PhD. [research advisor]
* Andrea D'Angelo [investigator]
* Claudio Savelli [investigator]
* Gabriele Tagliente [pivotal contributor]