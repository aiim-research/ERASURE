# ERASURE - Redefining Privacy Through Selective Machine Unlearning

![ERASURE Logo](https://raw.githubusercontent.com/aiim-research/ERASURE/refs/heads/main/ERASURE%20LOGO.png?token=GHSAT0AAAAAACXNOTNZVSMFO36TR7DLVFVIZ54TINA)


## Table of Contents
* [General Information](#general-information)
* [Team Information](#team-information)
* [First steps with ERASURE](#first-steps-with-erasure)
* [Resources Provided with the Framework](#resources-provided-with-the-framework)
* [References](#references)

## General Information
The challenge of ensuring privacy and compliance in machine learning models, such as those used in recommendation systems, healthcare, and financial forecasting, is exacerbated by the difficulty of removing specific data from trained models. Machine Unlearning (MU) techniques aim to address this by providing methods to selectively erase data from models, enhancing privacy, trustworthiness, and adaptability. However, the current landscape of MU research is fragmented, with inconsistencies in definitions, methodologies, datasets, and evaluation protocols. 

To tackle these challenges, we introduce ERASURE, a research framework designed to facilitate the research process of machine unlearning, focusing on efficiently and securely removing specific data or knowledge from machine learning models. This framework addresses privacy and compliance concerns by providing a way to systematically evaluate algorithms and methodologies that allow for the selective "erasure" of data, ensuring that the information has no residual influence on model predictions once removed.

ERASURE offers fully extensible built-in components, allowing users to define custom unlearning techniques, integrate custom and synthetic datasets, implement tailored evaluation metrics, and meld seamlessly with state-of-the-art machine learning models.
Additionally, it provides a flexible environment for defining and running unlearning experiments, enabling researchers to evaluate the effectiveness and efficiency of various unlearning methods systematically.


## Team Information:
* Prof. Giovanni Stilo, PhD. [project leader/research advisor]
* Flavio Giobergia, PhD. [research advisor]
* Andrea D'Angelo [investigator]
* Claudio Savelli [investigator]
* Gabriele Tagliente [pivotal contributor]

## First steps with ERASURE
```console
foo@bar:~$ python main.py configs/proof_of_concept.jsonc
```


## Resources Provided with the Framework


### Datasets:

You can quickly implement any Dataset you like in ERASURE. At the moment, ERASURE has built-in DataSource classes for the following sources:

![Static Badge](https://img.shields.io/badge/Datasource-TorchVision-blue)

![Static Badge](https://img.shields.io/badge/Datasource-Hugging%20Face-blue)

![Static Badge](https://img.shields.io/badge/Datasource-UCI%20Repository-blue)

![Static Badge](https://img.shields.io/badge/Datasource-PyTorch%20Geometric-blue)






### Unlearners: 

Developing your own custom Unlearner in ERASURE is easy. 

However, benchmarking your Unlearner in an unified environment is just as important. Right now, ERASURE comes built-in with the following Unlearners:

* ![Static Badge](https://img.shields.io/badge/Unlearner-Gold%20Model-red)
 **GoldModel**: Retrain the model from scratch with a specific (sub)set of the full dataset (usually retain set to evaluate the performance of the model after unlearning).

* ![Static Badge](https://img.shields.io/badge/Unlearner-Fine%20Tuning-red)
 **Fine-Tuning**: Fine-tunes the model with a specific (sub)set of the full dataset (usually retain set).

* ![Static Badge](https://img.shields.io/badge/Unlearner-NegGrad-red)
 **NegGrad** [1]: Negative Gradient (NegGrad) unlearning algorithm. This method fine-tunes the model to forget specific data points by reversing the gradient direction on the forget data. For each sample in the forget set, the stochastic gradient ascent is applied, effectively pushing the model to unlearn the patterns associated with these specific samples.

* ![Static Badge](https://img.shields.io/badge/Unlearner-Advanced%20NegGrad-red)
 **Advanced NegGrad** [2]: Advanced NegGrad is an extension of the NegGrad algorithm that incorporates a loss term to prevent catastrophic forgetting. In particular, Fine-Tuning is applied to the retain set, and NegGrad is applied to the forget set. 

* ![Static Badge](https://img.shields.io/badge/Unlearner-UNSIR-red)
 **UNSIR** [3]:  The method is divided in two phases: 1. In the first phase (Impair), noise is added to perturb the weights of the model. 2. In the second phase (Repair), the model is trained with the retain data to restore its performance. 

    * *Attention*: Since the second phase of the model is a simple finetuning, the UNSIR implementation adopted only perform the first part of the original method, if you want to apply the full UNSIR method, you should apply the second phase (**Fine-Tuning**) manually. 

    * *Attention*: Since the original method is thought for class-unlearning setting, we propose here the modified version proposed by [2].

* ![Static Badge](https://img.shields.io/badge/Unlearner-Bad%20Teaching-red)
 **Bad Teaching** [4]: The method uses a distillation method between two teachers and a student model: The student model and the good teacher are both initialized with the same weights, the ones of the model before unlearning. The bad teacher is initialized with random weights or could be finetuned for few epochs on the retain set. The KL-divergence of the logits between the good teacher and the student are minimized on the retain set, while the KL-divergence of the logits between the bad teacher and the student is minimized on the forget.

* ![Static Badge](https://img.shields.io/badge/Unlearner-SCRUB-red)
**SCRUB** [5]: The method operates in two main stages:

    1. Training on Retain Data: The model (student) is trained using the data to be retained, guided by a frozen version of the model (teacher) to minimize divergence and finetune on retain data.
    2. Divergence on Forget Data: The model then maximizes divergence between its outputs and the teacher's outputs on the data to be forgotten, aiming to remove learned information specific to these samples.

* ![Static Badge](https://img.shields.io/badge/Unlearner-Selective%20Synaptic%20Dampening-red)
 **Selective Synaptic Dampening** [6]: The method dampens model parameters disproportionately important to the forget set. Using the Fisher information matrix, the method identifies the parameters associated with the forget set and dampens them based on their relative importance with respect to the retain set.

* ![Static Badge](https://img.shields.io/badge/Unlearner-UNSIR-red) **UNSIR** [7] The UNSIR method consists of two phases. The original approach was designed to remove entire classes from the model. However, the version presented here has been modified by \cite{choi2023towards} to enable its application to individual samples.
In the first phase (Impair), noise is introduced into the pipeline to perturb the model weights, effectively erasing information about the selected samples. This is achieved by generating a random tensor of the same size for each sample designated for forgetting and passing it through the model. To mimic the model’s random response to the sample being forgotten, the mean squared error (MSE) loss is computed between the original logits produced by the model for the sample and the logits generated using the random tensor.
In the second phase (Repair), the model's performance is restored on the remaining dataset by fine-tuning it with the retained data.

* ![Static Badge](https://img.shields.io/badge/Unlearner-Fisher%20Forgetting-red) **Fisher Forgetting** [8]: The Fisher Forgetting method is designed to determine the optimal noise required to eliminate specific data from the model while preserving only the essential information for the classification task. This is achieved by leveraging derivative information, specifically the diagonal of the Fisher Information Matrix (FIM), which is evaluated based on individual model weights concerning the Retain Set. Gaussian noise with a mean of zero and variance corresponding to the diagonal of the FIM for each weight is then applied. This approach is inspired by a regularization technique to mitigate the effects of catastrophic forgetting.

* ![Static Badge](https://img.shields.io/badge/Unlearner-SalUn-red) **SalUn** [9]: SalUn utilizes saliency maps to selectively unlearn the most active weights linked to the Forget Set. By employing gradient ascent, the model pinpoints and adjusts the most influential weights while preserving the others. A hyperparameter-defined threshold controls the proportion of weights affected. In the original method, Successive Random Labels are applied following the evaluation of the saliency mask.


## References

<!--  taken with Harvard reference style -->
1. Golatkar, A., Achille, A. and Soatto, S., 2019. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In 2020 IEEE. In CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 9301-9309).

2. Choi, D. and Na, D., 2023. Towards machine unlearning benchmarks: Forgetting the personal identities in facial recognition systems. arXiv preprint arXiv:2311.02240.

3. Tarun, A.K., Chundawat, V.S., Mandal, M. and Kankanhalli, M., 2023. Fast yet effective machine unlearning. IEEE Transactions on Neural Networks and Learning Systems.

4. Chundawat, V.S., Tarun, A.K., Mandal, M. and Kankanhalli, M., 2023, June. Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 7210-7217).

5. Kurmanji, M., Triantafillou, P., Hayes, J. and Triantafillou, E., 2024. Towards unbounded machine unlearning. Advances in neural information processing systems, 36.

6. Foster, J., Schoepf, S. and Brintrup, A., 2024, March. Fast machine unlearning without retraining through selective synaptic dampening. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 11, pp. 12043-12051).

7. Tarun, A.K., Chundawat, V.S., Mandal, M. and Kankanhalli, M. (2023) ‘Fast yet effective machine unlearning’, IEEE Transactions on Neural Networks and Learning Systems.

8. Golatkar, A., Achille, A. and Soatto, S. (2020) ‘Eternal sunshine of the spotless net: Selective forgetting in deep networks’, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9304–9312.

9. Fan, C., Liu, J., Zhang, Y., Wong, E., Wei, D. and Liu, S. (2024) ‘SalUn: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation’, arXiv preprint, arXiv:2310.12508. Available at: https://arxiv.org/abs/2310.12508.