Please see git log for code that was re-used versus generated for this project.

# ColBERT on Natural Questions
This branch has been forked from the [ColBERT](https://github.com/stanford-futuredata/ColBERT/tree/main) repo for extension and experiments on new datasets.

## Setup

Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install#linux):
```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-457.0.0-linux-x86_64.tar.gz
```
```bash
tar -xf google-cloud-cli-457.0.0-linux-x86_64.tar.gz
```
```bash
./google-cloud-sdk/install.sh
```

Make a data folder in the project root and navigate to it.

Download the [data](https://ai.google.com/research/NaturalQuestions/download):
```bash
gsutil cp gs://natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz . && gunzip simplified-nq-train.jsonl.gz
```
```bash
gsutil cp gs://natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz . && gunzip nq-dev-all.jsonl.gz
```

## Environment

```bash
pip install -r requirements.txt
```

Make sure you have a compatible CUDA toolkit installed along with a compatible version of pytorch. [Instructions](https://pytorch.org/get-started/locally/)

## Pre-process (training)

Run:
```bash
./utility/preprocess/natural_questions_to_tsv.py --nq_jsonl ./data/simplified-nq-train.jsonl --tsv_file ./data/nq_train_triples.tsv
```

Run from data folder:
```bash
./utility/preprocess/head10.sh ./data/nq-dev-all.jsonl 
```
This last command creates json files for the first 10 line of your test set so that you can inspect your data.

Run from data folder:
```bash
python ../utility/preprocess/generate_llm_challenge.py --nq_jsonl nq-dev-all.jsonl > llm_challenge_prompts.txt
```

## Train

Compile the package:
```bash
pip install .
```

Run the train module:
```bash
python -m colbert.train --accum 1 --triples ./data/nq_train_triples.tsv
```
