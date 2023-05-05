### GloVe/experimental

Additional functionality added to GloVe repo, to compare embedding initialisations across a range of text corpora. The focus of this project is child-directed speech, and as such many of the corpora of interest are child-oriented. Others (enwik8/text8) are used as a point of comparison. Scripts for processing corpora into an appropriate format for the GloVe algorithm (text pre-processing; one document per line etc) are found in experimental/corpora.

Much of this code exists to run repeated instantiations of the GloVe algorithm on different subsets of corpora, to get estimates of embedding stability across corpus samples. Such sampling and embedding scripts are found in experimental/multi_fold

Comparisons of resultant embeddings to the pre-trained embeddings from <a href=https://nlp.stanford.edu/projects/glove/>Stanford NLP</a> are also key here. These are included as text files in experimental/assets. As we were interested in statistics across early/late acquired concepts differentially, we also use age-of-acquisition data from Frank et al. This is included in experimental/assets