# NUSISS Project - US 2024 Presidential Elections Aspect Based Sentiment Analysis

## Background

This is a prototype which scrapes Reddit and Youtube for Top comments about Trump and Kamala Harris.

The pipeline:

1) Scrapes data from both platforms and consolidates the data
2) Preprocesses the data (e.g. fix encoding, remove non-english)
3) Uses pre-trained model to do
   coreference resolution for entities.
4) Uses pre-trained model to do subjectivity classification and remove objective sentences.
5) Tokenize comments into sentences.
6) Use 5 fine tune models to extract entity, aspect related to entity, and sentiment related to aspect.

Running of this code will require the follow to work fully:

1) Download of 5 finetuned model and placing them into `model` folder.
2) Credentials for Reddit and Youtube to be defined in `credentials/reddit.env` for use by respective APIs

## Windows install

Create virtual environment and install requirements

```
python -m venv myVenv
"myVenv/Scripts/activate"
pip install -r requirements.txt
```

Additionally, install spacy-experimental model for coreference resolution

```
python -m pip install -U pip setuptools wheel
python -m pip install spacy-experimental
pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl
```
