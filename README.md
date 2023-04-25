# Question-answering-engine

This program aims to answer simple questions about one entity (e.g. where was Michael Douglas born, what is the capital of France etc.).

## Usage
Open a notebook environment, such as Google Colab or Kaggle.
There, you will need to upload all the files contained here (two trained models and two text files): https://drive.google.com/drive/folders/1Hy31igylaXcGaYkXZWtAuhinsEpb5527?usp=sharing
After the files are uploaded in the environment, simply run all the cells. 
This will set up the models and prepare the question answering engine.
In the last cell, a space will appear. There, type your question, and the answer will be typed below. 

## Data
Data used to train the models are taken from WikiData(https://www.wikidata.org/wiki/Wikidata:Main_Page), and are retrieved using SPARQL (https://www.w3.org/TR/rdf-sparql-query/).

The project was developed as a part of the "Deep Learning for NLP" course, Dept. of Informatics and Telecommunications, NKUA.
