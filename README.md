# AMOVE
Adaptive offloading in mobile devices with Multi-Exit DNNs
This repository is the official implementation of AMOVE. The experimental section of AMOVE can be divided into two major parts:

Part 1 (Finetuning and predictions): We finetune the ElasticBERT backbone after attaching exits to all the layers on RTE, SST-2, MNLI and MRPC (GLUE) datasets and then obtain prediction as well as confidence values for the evaluation (SciTail, IMDB, Yelp, SNLI, QQP)(GLUE and ELUE datasets except IMDB) i.e. all exit predictions for all samples (num_samples X num_exits)
