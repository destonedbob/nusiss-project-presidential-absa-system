# nusiss-project-presidential-absa-system

Windows install
python -m venv myVenv
"myVenv/Scripts/activate"
pip install -r requirements.txt

Install spacy model for coreference resolution
python -m pip install -U pip setuptools wheel
python -m pip install spacy-experimental
pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl

