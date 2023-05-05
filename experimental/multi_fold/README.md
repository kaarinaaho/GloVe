### Embedding corpus subsets

Code here relates to running the GloVe algorithm repeatedly on subsets of the input corpus, to ascertain the stability of inferred embeddings across samples of a certain size. Usage of this code is recommended as follows:

 - Ensure that corpus of interest has been generated and exists in experimental/corpora/processed
 - Input corpus filename in run_multifold.py
 - Select the sampling type: either:
    - "nfold": split the corpus into n folds and generate n samples from the corpus, each of which randomly excludes one fold. 
    - "fixedsize": generate samples of a fixed size from a corpus. Number of samples is specified.
 - A subdirectory for the samples generated from the corpus will be generated, containing subdirectories for splitcorpora, vocab, vectors and co-occurrences each
 - Analysis of the generated samples can be conducted by running analyse_multifold.py


 ## run_multifold

 # SplitEmbGenerator()


