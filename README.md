# EESplit: Early Exit in Deep Neural Networks with Split Computing
This repository is the official implementation of EESplit. The experimental section of EESplit can be divided into two major parts:

Part 1 (Finetuning and predictions): We finetune the ElasticBERT backbone after attaching exits to all the layers on RTE, SST-2, MNLI and MRPC (GLUE) datasets and then obtain prediction as well as confidence values for the evaluation (SciTail, IMDB, Yelp, SNLI, QQP)(GLUE and ELUE datasets except IMDB) i.e. all exit predictions for all samples (num_samples X num_exits)

Part 2: Evaluate EESplit and EESplit-S using the prediction matrix which could be done by running ucb_implementation_using_side_info.py and ucb_implementation_without_using_side_info.py.we plotted the regret, accuracy and cost performance of the EESplit and EESplit-S algorithms which could be done by running Accuracy_cost_plots.ipynb and regret_calculation.ipynb file  
