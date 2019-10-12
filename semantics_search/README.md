For each search engine, to run the search, download all files from the corresponding directory, and then run: </br>
```
pip3 install -r requirements.txt
python "your_python_file" "your_search_query"
```
in the directory with all necessary files. </br>
For **ELMO** and **RuBERT** search engines, the command line is: <br>
```
pip3 install -r requirements.txt
python "your_python_file" "your_search_query" 2>/dev/null
```
* **fastText** – uses [pre-trained **fastText** model](http://vectors.nlpl.eu/repository/11/181.zip) (**ruscorpora_none_fasttextskipgram_300_2_2019**) from [**RusVectores**](https://rusvectores.org/en/models/). You need to download and unzip it to a working directory to run the search.
* **ELMO** – uses [pre-trained **ELMO** model](http://vectors.nlpl.eu/repository/11/196.zip) (**ruwikiruscorpora_lemmas_elmo_1024_2019**) from [**RusVectores**](https://rusvectores.org/en/models/). You need to download and unzip it to a working directory to run the search. Also, you need to download the [pre-computed matrix](https://drive.google.com/open?id=1Dgd7iNjP9YkuD5qyBKKwfxPdxwg5Nyfu) and run ```gzip``` on ```vocab.txt```. </br>
Source for the code to work with ELMO embeddings: https://github.com/ltgoslo/simple_elmo </br>
* **RuBERT** – uses [pre-trained **BERT** model](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz). You need to download and unzip it to a working directory to run the search. Also, you need to download the [pre-computed matrix](https://drive.google.com/open?id=1_iEe-7Ibx_L5e-aODXD3i4avsUoz6lf1).</br>
</br>
