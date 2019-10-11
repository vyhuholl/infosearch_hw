# infosearch_hw

My assignments for this course: https://github.com/hse-infosearch/infosearch
### friends_search
A simple search engine on a collection of documents (the "Friends" script). To run the search, do: </br>
```
pip3 install -r requirements.txt
python friends_search.py "your_search_query"
```
in the directory with *friends_search.py*, *FriendsSearch.pkl* and *requirements.txt*. </br>
### bm25_search
Another search engine on the [**Quora question pairs** dataset](https://www.kaggle.com/loopdigga/quora-question-pairs-russian)<sup>1</sup>, using [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) ranking function. To run the search, first download [BMSearchEngine.pkl](https://drive.google.com/open?id=1o6kBTsrcZ4SFGcWaN21L_eV4Vevd8y6x) and then run: <br>
```
pip3 install -r requirements.txt
python bm25_search.py "your_search_query"
```
in the directory with *bm25_search.py*, *BMSearchEngine.pkl* and *requirements.txt*. </br>

### semantics_search
More search engines on the [**Quora question pairs** dataset](https://www.kaggle.com/loopdigga/quora-question-pairs-russian)<sup>1</sup>, using different [word embeddinds](https://en.wikipedia.org/wiki/Word_embedding). For more info, see README.md at the directory.<br>
<br>
<sup>1</sup> A dataset of Quora question pairs. Here, we trained a search engine on "question1" column and used "question2" column to evaluate the quality of our search engine.
