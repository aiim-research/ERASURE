# ERASURE - CIKM 2025 STUDY ACROSS MODALITIES



## Table of Contents
* [Experiments](#experiments)
* [Reproducibility](#reproducibility)

## Experiments

In this folder, you will find the configuration files of all the experiments we conducted for this study.

Specifically, you can find information on the parameters used for each method, dataset, models etc. and the criteria for the construction of the Forget Set.

To find information on the construction of the Forget Set, consider, for instance, the file "ag_news.jsonc" in this folder.

In the data section of this configuration file, you will find:

'''
{"class":"erasure.data.preprocessing.add_z_label.StringContain", "parameters":{
        "contains":["Real Madrid", "Juventus", "Bayern Monaco", "Arsenal", "Manchester United", "Arsenal", "Chelsea", "Manchester City", "Inter Milan", "Lakers", "Ronaldo", "Messi"]}},
'''

This means that we flagged all samples containing these Named Entities, and then constructed the Forget Set accordingly,by including samples flagged as 1, as indicated in the same file, below:

'''
    {"class":"erasure.data.datasets.DataSplitter.DataSplitterByZ", "parameters":{"parts_names":["forget","other_ids_full"], "z_labels":[1], "ref_data":"all_shuffled"}},
'''

Not all datasets are fit for this sort of Named Entity forget set construction, so we default to sampling 20% of the training set when it is not available.

## Reproducibility

All the configuration files needed to reproduce the values on the paper are in this folder and we encourage reproducibility. For more information on how to reproduce all experiments, please visit the root of this Github repository.

