# Machine learning for music genre: multifaceted review and experimentation with Audioset: Experiments

Code repository for experiments conducted in the study
**[Machine learning for music genre: multifaceted review and experimentation with Audioset](https://link.springer.com/article/10.1007/s10844-019-00582-9).**

## Abstract

Music genre classification is one of the sub-disciplines of music information retrieval (MIR) with growing popularity among researchers, mainly due to the already open challenges. Although research has been prolific in terms of number of published works, the topic still suffers from a problem in its foundations: there is no clear and formal definition of what genre is. Music categorizations are vague and unclear, suffering from human subjectivity and lack of agreement. In its first part, this paper offers a survey trying to cover the many different aspects of the matter. Its main goal is give the reader an overview of the history and the current state-of-the-art, exploring techniques and datasets used to the date, as well as identifying current challenges, such as this ambiguity of genre definitions or the introduction of human-centric approaches. The paper pays special attention to new trends in machine learning applied to the music annotation problem. Finally, we also include a music genre classification experiment that compares different machine learning models using Audioset.

Read the article here: https://rdcu.be/b87uq

### Demo

I've setup an online demo webapp to show how models trained with this repository work in practice.

https://jramcast.github.io/mgr-app/


### Usage

### Requirements

* **Python 3.6**
* **Pipenv**: The project dependencies are managed using `pipenv`.
To install it, you can follow the [pipenv installation guide](https://pipenv.readthedocs.io/en/latest/install/).
* **Audioset VGGish files**: you need to download Audioset files in order to train the models.

#### 1. Install dependencies

```sh
pipenv install --dev
```

#### 2. Download Audioset VGGish model files

First, create a `downloads` folder in the root of the project, and create an `audioset` folder inside it:

```sh
mkdir downloads && cd downloads
mkdir audioset
```

Now download *class_labels_indices.csv* file to `downloads/` (this file defines dataset labels):

```sh
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
```

You can also download it from [Audioset downloads page](https://research.google.com/audioset/download.html).

Go to `downloads/audioset` and download dataset in csv file:

```sh
cd audioset
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv

```

Finally, [download Audioset features](http://storage.googleapis.com/eu_audioset/youtube_corpus/v1/features/features.tar.gz) and save them in  the folder `downloads/audioset/audioset_v1_embeddings`. Once downloaded and extracted, your file structure should look like this:

```
- downloads/
  - class_labels_indices.csv
  - audioset/
    - balanced_train_segments.csv
    - unbalanced_train_segments.csv
    - eval_segments.csv
    - audioset_v1_embeddings/
        - bal_train/
          - X.tfrecord
          -...
        - unbal_train/
          - X.tfrecord
          -...
        - eval/
          - X.tfrecord
          -...
```

#### 3. Run experiments

Run this command:

```sh
DATA_DIR="./downloads/audioset/audioset_v1_embeddings" python main.py *EXPERIMENT* *OPTIONS*
```

Experiment can be any of: {bayes,deep,lstm,svm,tree}

As options, you can add:

* **--balanced**: to use the balanced Audioset split.
* **--epochs EPOCHS**: only for deep learning experiments.

Example:

```sh
# Train the LSTM RNN model, on the balanced split of Audioset, for 20 epochs
DATA_DIR="./downloads/audioset/audioset_v1_embeddings" python main.py lstm --balanced --epochs 20
```
