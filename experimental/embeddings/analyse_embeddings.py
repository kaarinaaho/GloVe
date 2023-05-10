import random
import numpy as np
import os, sys
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import math

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
import utils


class Analyser():
    def __init__(self, fold_folder):
        self.fold_folder = fold_folder
        self.results_folder = os.path.join(
            Path(__file__).parent, f"{self.fold_folder}/results"
        )
        self.aoa = utils.load_aoa_data()
        wdnimg, ___, og_vocab = utils.embs_quickload(50)
        self.wdnimg = wdnimg
        self.og_vocab = og_vocab
        vecs, vocab = self._load_vectors()

        # Filter for only items present in original embs
        vocab_reduced = [x for x in self.og_vocab if x in vocab]
        new_idx = [vocab.index(x) for x in vocab_reduced]
        self.vecs = vecs[new_idx, :]
        self.vocab = vocab_reduced


    def _load_vectors(self):
        print("Loading vectors...")

        # Load vector file
        vecs = np.loadtxt(
            os.path.join(self.results_folder, "vectors.txt"),
            usecols = [x for x in range(1,50)]
            )

        # Load corresponding vocab file
        textfile = open(
            os.path.join(self.results_folder, "vocab.txt"),
            "r")
        vocab = textfile.read().split('\n')
        vocab = [x.split(" ")[0] for x in vocab]
        print("Vectors loaded")

        return vecs, vocab


    def _upper_triu_for_vocab(self, emb, emb_vocab, select_vocab):

        index = [emb_vocab.index(x) for x in select_vocab if x in emb_vocab]
        idxed_emb = emb[index][:]
        #cos_sim = cosine_similarity(idxed_emb)
        sim = euclidean_distances(idxed_emb)
        upper = sim[np.triu_indices(len(select_vocab),1)]

        return index, idxed_emb, sim, upper


    def _compare_to_emb_corr(self, emb1, emb1_vocab, emb2, emb2_vocab, aoa):

        intersect_vocab = [x for x in emb1_vocab if x in emb2_vocab]

        __, __, __, upper1 = self._upper_triu_for_vocab(
            emb1, emb1_vocab, intersect_vocab)
        __, __, __, upper2 = self._upper_triu_for_vocab(
            emb2, emb2_vocab, intersect_vocab)

        corr, p = pearsonr(upper1, upper2)

        intersect_vocab_aoa = [x for x in intersect_vocab if x in list(aoa["concept_i"])]

        __, __, __, upper1 = self._upper_triu_for_vocab(
            emb1, emb1_vocab, intersect_vocab_aoa)
        __, __, __, upper2 = self._upper_triu_for_vocab(
            emb2, emb2_vocab, intersect_vocab_aoa)

        corr2, p = pearsonr(upper1, upper2)

        return corr, corr2 

    def _jaccard_sim(self, x, y):

        intersection = [a for a in y if y in x]
        union = list(set(x + y))

        return len(intersection)/len(union)

    def choose(self, n, r):

        """ N choose r """

        return (math.factorial(n))/(math.factorial(n-r) * math.factorial(r))


    def get_expected_jaccard(self, n, m):

        to_add = []
        for k in range(m+1):
            add = ((self.choose(m, k) * self.choose(n-m, m-k))/self.choose(n,m)) * (k/(2*m - k))
            to_add.append(add)

        return np.sum(to_add)


    def _compare_to_emb_neighbour(self, emb1, emb1_vocab, emb2, emb2_vocab, aoa, nn=10):
        """Mean jaccard scores"""
        intersect_vocab = [x for x in emb1_vocab if x in emb2_vocab]

        emb1 = emb1[[emb1_vocab.index(x) for x in intersect_vocab],:]
        emb2 = emb2[[emb2_vocab.index(x) for x in intersect_vocab],:]

        # Get list of nearest neighbours in same vocab space
        sim1 = euclidean_distances(emb1)
        sim2 = euclidean_distances(emb2)

        # Sort to get index of items in each ranking position
        sort1 = np.argsort(sim1, axis=1)
        sort2 = np.argsort(sim2, axis=1)

        # Getjaccard similarity between lists
        top_n_1 = sort1[:, 1:1+nn]
        top_n_2 = sort2[:, 1:1+nn]

        jaccards = []
        for i in range(top_n_1.shape[0]):
            js = self._jaccard_sim(top_n_1[i,:], top_n_2[i,:])
            jaccards.append(js)

        expected_jac_full = self.get_expected_jaccard(len(intersect_vocab), nn)
        
        intersect_idxs_aoa = [i for i, x in enumerate(intersect_vocab)
                                if x in list(aoa["concept_i"])]


        emb1 = emb1[intersect_idxs_aoa,:]
        emb2 = emb2[intersect_idxs_aoa,:]

        # Get list of nearest neighbours in same vocab space
        sim1 = euclidean_distances(emb1)
        sim2 = euclidean_distances(emb2)

        # Sort to get index of items in each ranking position
        sort1 = np.argsort(sim1, axis=1)
        sort2 = np.argsort(sim2, axis=1)

        # Getjaccard similarity between lists
        top_n_1 = sort1[:, 1:1+nn]
        top_n_2 = sort2[:, 1:1+nn]

        jaccards_aoa = []
        for i in range(top_n_1.shape[0]):
            js = self._jaccard_sim(top_n_1[i,:], top_n_2[i,:])
            jaccards_aoa.append(js)

        expected_jac_aoa = self.get_expected_jaccard(len(intersect_idxs_aoa), nn)


        # Repeat for AoA words only
        return np.mean(jaccards), np.mean(jaccards_aoa), expected_jac_full, expected_jac_aoa


    def get_corr_v_og(self):
        # Get pairwise correlation of folds with original GloVe

        corr_full, corr_aoa = self._compare_to_emb_corr(
            self.vecs, self.vocab, self.wdnimg, self.og_vocab, self.aoa
            )

        return corr_full, corr_aoa


    def get_corr_v_other(self, other_vocab_path, other_vec_path):
        vecs_other = np.loadtxt(
            other_vec_path,
            usecols = [x for x in range(1,50)]
        )
        textfile = open(
            other_vocab_path,
            "r")
        vocab_other = textfile.read().split('\n')
        vocab_other = [x.split(" ")[0] for x in vocab_other]

        # Filter for items in og embedding
        idxs = [i for i, x in enumerate(vocab_other) if x in self.og_vocab]
        vocab_other = [vocab_other[x] for x in idxs]
        vecs_other = vecs_other[idxs,:]
                
        corr_full, corr_aoa = self._compare_to_emb_corr(
            self.vecs, self.vocab, vecs_other, vocab_other, self.aoa
            )

        return corr_full, corr_aoa



    def get_nn_v_og(self, nn=10):

        jac_full, jac_aoa, exp_jac_full, exp_jac_aoa = self._compare_to_emb_neighbour(
            self.vecs, self.vocab, self.wdnimg, self.og_vocab, self.aoa, nn
            )

        return jac_full, jac_aoa, exp_jac_full, exp_jac_aoa





if __name__ == "__main__":

    analyse = Analyser(
        os.path.join(Path(__file__).parent, "childes")
        )

    jac_full, jac_aoa, exp_jac_full, exp_jac_aoa = analyse.get_nn_v_og()
    print(jac_full, jac_aoa, exp_jac_full, exp_jac_aoa)


    analyse = Analyser(
    os.path.join(Path(__file__).parent, "enwik8")
    )

    jac_full, jac_aoa, exp_jac_full, exp_jac_aoa = analyse.get_nn_v_og()
    print(jac_full, jac_aoa, exp_jac_full, exp_jac_aoa)