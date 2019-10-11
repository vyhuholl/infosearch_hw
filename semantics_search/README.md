* **fastText** – uses [pre-trained **fastText** model](http://vectors.nlpl.eu/repository/11/181.zip) (**ruscorpora_none_fasttextskipgram_300_2_2019**) from [**RusVectores**](https://rusvectores.org/en/models/).
* **ELMO** – uses [pre-trained **ELMO** model](http://vectors.nlpl.eu/repository/11/196.zip) (**ruwikiruscorpora_lemmas_elmo_1024_2019**) from [**RusVectores**](https://rusvectores.org/en/models/).
* **RuBERT** – uses [pre-trained **BERT** model](http://docs.deeppavlov.ai/en/master/features/models/bert.html). <br>
<br>
For each search engine, to run the search, download all files from the corresponding directory, and then run:
```
pip3 install -r requirements.txt
python <python_file> "your_search_query"
```
in the directory with all necessary files. </br>
