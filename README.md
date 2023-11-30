# Question-answering-engine

This Python program aims to answer simple questions about one entity (e.g. where was Michael Douglas born, what is the capital of France etc.).

It performs fine-tuning of two BERT models: one that predicts an entity on a question and one that predicts the relation (born, raised, produced).
For example, in the sentence "where was Michael Douglas born", the entity predicting model will return Michael Douglas, if successful. The relation predicting model will classify the question into one of relations categories (where born, when born, where raised, when produced etc.).

BERT fine-tuning is performed using a training dataset of 19481 instances, contained in _data_ folder. The dataset is taken from [wikidata-simplequestions]. 

That way, in a question typed by the user (e.g., when was Michael Douglas born), the two models will predict the entity (Michael Douglas) and the relation where born. A SPARQL query will then be made to [Wikidata] database and the answer (i.e., where was Michael Douglas born), will be returned.

## Requirements

You can the engine on any machine that has Python and pip installed.

## Usage

Start by downloading the current repository and moving to _Question-answering-engine_ directory.
You should have the packages written in [requirements.txt] installed. To install them, type:

```
pip install -r requirements.txt
```

### Run .py

Now, you can run the program. Run:
```
python3 qa.py
```

A box will appear where you can type your question. 

### Run .ipynb

Open qa.ipynb on a notebook environment, such as Google Colab or Kaggle.
There, you will need to upload all the files from [models] and [data] directory.
After the files are uploaded in the environment, simply run all the cells. 
This will set up the models and prepare the question answering engine.
In the last cell, a space will appear. There, type your question, and the answer will be typed below. 

The project was developed as a part of the "Deep Learning for NLP" course, Dept. of Informatics and Telecommunications, NKUA and has been extended afterwards.
