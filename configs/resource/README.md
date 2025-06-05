# A Comprehensive Study of Machine Unlearning\\Across Multiple Modalities

A unified study to evaluate machine unlearning techniques across text, tabular, image, and graph classification tasks, using utility, efficacy, and efficiency metrics.


## Table of Contents
* [Experiments](#experiments)
* [Reproducibility](#reproducibility)
* [Quick Start](#quick_start)
* [Complete Results](#results)

## Experiments

In this folder, you will find the configuration files of all the experiments we conducted for this study.

Specifically, you can find information on the parameters used for each method, dataset, models etc. and the criteria for the construction of the Forget Set.

To find information on the construction of the Forget Set, consider, for instance, the file "ag_news.jsonc" in this folder.

In the data section of this configuration file, you will find:

```
{"class":"erasure.data.preprocessing.add_z_label.StringContain", "parameters":{
        "contains":["Real Madrid", "Juventus", "Bayern Monaco", "Arsenal", "Manchester United", "Arsenal", "Chelsea", "Manchester City", "Inter Milan", "Lakers", "Ronaldo", "Messi"]}},
```

This means that we flagged all samples containing these Named Entities, and then constructed the Forget Set accordingly, by including samples flagged as 1, as indicated in the same file, below:

```
    {"class":"erasure.data.datasets.DataSplitter.DataSplitterByZ", "parameters":{"parts_names":["forget","other_ids_full"], "z_labels":[1], "ref_data":"all_shuffled"}},
```

Not all datasets are suitable for this type of Named Entity-based forget set construction, so we default to sampling 20% of the training set when it is not available.

## Reproducibility

All the configuration files needed to reproduce the values on the paper are in this folder and we encourage reproducibility. For more information on how to reproduce all experiments, please visit the root of this Github repository.

## Quick Start

If you are in a rush, to simply reproduce results, for instance for ag_news, you can just run:

```python main.py configs/resource/ag_news.jsonc``` 

from within the root folder.

## Complete Results

Here, we present a table containing the results of all runs we performed, including different random seeding.

