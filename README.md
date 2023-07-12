## UBERT: Unsupervised Adaptive Early Exits in BERT

This repository is the official implementation of UBERT.
The experimental section of UBERT can be divided into two major parts:

Part 1 (Finetuning and predictions): We finetune the ElasticBERT backbone after attaching exits to all the layers on RTE, SST-2, MNLI and MRPC (GLUE) datasets and then obtain prediction as well as confidence values for the evaluation (SciTail, IMDB, Yelp, SNLI, QQP)(GLUE and ELUE datasets except IMDB)  i.e. all exit predictions for all samples (num_samples X num_exits)

Part 2: Evaluate UBERT using the prediction matrix which could be done by running "UBERT one exit" file. We have provided the prediction matrix for Yelp dataset as a csv file which could be used to evaluate UBERT(Early_Exit_Confidence_data_yelp____.csv)

For UBERT with 2 exits execute the "UBERT_for_two_layers_reduced" file.

## Requirements
To install requirements 
```setup
pip install -r requirements.txt
```

## Datasets

GLUE datasets are available at: [GLUE Datasets](https://gluebenchmark.com/tasks)

ELUE datasets are available at: [Elue Datasets](http://eluebenchmark.fastnlp.top/#/landing)

Yelp dataset could be found here: [Yelp dataset](https://web.archive.org/web/20220401065200/https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)

IMDb dataset could be created by running "Create_IMDb_tsv_files.ipynb" code.

## Previous Codes Used (Acknowledgement)
Finetuning and pre-training the multi-exit model (Part-1) is closely based on [ElasticBERT](https://github.com/fastnlp/ElasticBERT), we acknowledge and thank the authors for the codebases. 
