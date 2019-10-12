For each search engine, to run the search, download all files from the corresponding directory, and then run: </br>
```
pip3 install -r requirements.txt
python "your_python_file" "your_search_query"
```
in the directory with all necessary files. </br>
* **fastText** – uses [pre-trained **fastText** model](http://vectors.nlpl.eu/repository/11/181.zip) (**ruscorpora_none_fasttextskipgram_300_2_2019**) from [**RusVectores**](https://rusvectores.org/en/models/). You need to download and unzip it to a working directory to run the search.
* **ELMO** – uses [pre-trained **ELMO** model](http://vectors.nlpl.eu/repository/11/196.zip) (**ruwikiruscorpora_lemmas_elmo_1024_2019**) from [**RusVectores**](https://rusvectores.org/en/models/). You need to download and unzip it to a working directory to run the search. Also, you need to run ```gzip``` on ```vocab.txt```. </br>
Source for the code to work with ELMO embeddings: https://github.com/ltgoslo/simple_elmo
* **RuBERT** – uses [pre-trained **BERT** model](http://docs.deeppavlov.ai/en/master/features/models/bert.html). </br>
</br>
