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
6) Use 5 fine tuned models to extract entity, aspect related to entity, and sentiment related to aspect.

Running of this code will require the follow to work fully:

1) Download of 5 finetuned model and placing them into `model` folder.
   * [entity model](https://huggingface.co/destonedbob/nusiss-election-project-entity-model-distilbert-base-cased)
   * [aspect seq2seq model](https://huggingface.co/destonedbob/nusiss-election-project-aspect-seq2seq-model-facebook-bart-large)
   * [sentiment seq2seq model](https://huggingface.co/destonedbob/nusiss-election-project-sentiment-seq2seq-model-facebook-bart-large)
   * Other two models are uploaded here in `./model/`
2) Credentials for Reddit and Youtube to be defined in `credentials/reddit.env` for use by respective APIs. Please see the relevant documentations

You may use main.ipynb to do the above. If you do not wish to scrape data, you may create a custom dataframe with sentences in the `Inference` section to test the model.

Note: this code has only been tested on a single windows system with a cuda GPU.

Related Repo: [For Streamlit UI code](https://github.com/destonedbob/nusiss-project-presidential-absa-streamlit-ui)

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
