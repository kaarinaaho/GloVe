import random
import numpy as np
import os, sys
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
import utils


class FoldAnalyser():
    def __init__(self, fold_folder):
        self.fold_folder = fold_folder
        self.results_folder = os.path.join(
            Path(__file__).parent, f"{self.fold_folder}/results"
        )
        self.aoa = utils.load_aoa_data()
        wdnimg, ___, og_vocab = utils.embs_quickload(50)
        self.wdnimg = wdnimg
        self.og_vocab = og_vocab
        vec_list, all_fold_vocab = self._load_all_vectors()

        # Filter for only items present in original embs
        all_fold_reduced = [x for x in self.og_vocab if x in all_fold_vocab]
        all_fold_idx = [all_fold_vocab.index(x) for x in all_fold_reduced]
        self.vec_list = [x[all_fold_idx, :] for x in vec_list]
        self.all_fold_vocab = all_fold_reduced


    def _load_all_vectors(self):
        print("Loading vectors...")
        # Load all vector file names
        vector_folder = os.path.join(
            self.results_folder, 'vectors'
        )
        vec_list = []
        vec_files = [x for x in os.listdir(vector_folder)
                    if (x.split(".")[-1] == "txt")]

        # Load all vocab file names
        vocab_folder = os.path.join(
            self.results_folder, 'vocab'
        )
        vocab_files = [x for x in os.listdir(vocab_folder)]

        # Get words which occur in all vocabularies
        all_fold_vocab = self._get_intersection_vocab(
            vocab_folder, vocab_files
            )

        for i, f in enumerate(vec_files):
            print(f"Loading file {i}/{len(vec_files)}: {f}")
            # Load vector file
            vecs = np.loadtxt(
                os.path.join(vector_folder, f),
                usecols = [x for x in range(1,50)]
                )

            # Load corresponding vocab file
            n = f.split(".")[0]
            n = n.replace("split", "")

            textfile = open(
                os.path.join(vocab_folder, f"{n}.txt"),
                "r")
            vocab = textfile.read().split('\n')
            vocab = [x.split(" ")[0] for x in vocab]

            # Words are in same order for each vector set
            vocab_idxs = [vocab.index(x) for x in all_fold_vocab]

            vecs = vecs[vocab_idxs,:]
            vec_list.append(vecs)
        print("Vectors loaded")

        return vec_list, all_fold_vocab


    def _get_intersection_vocab(self, vocab_folder, vocab_files):
        # Initialise vocab    
        textfile = open(
            os.path.join(vocab_folder, vocab_files[0]),
            "r")
        intersect_vocab = textfile.read().split('\n')
        intersect_vocab = [x.split(" ")[0] for x in intersect_vocab]
        print("Loading intersection vocabularies...")
        print(f"1/{len(vocab_files)}")
        for i, f in enumerate(vocab_files[1:]):
            print(f"{i+2}/{len(vocab_files)}")
            textfile = open(
                os.path.join(vocab_folder, f),
                "r")
            vocab1 = textfile.read().split('\n')
            vocab1 = [x.split(" ")[0] for x in vocab1]

            # Filter for words in both intersect and current
            intersect_vocab = [x for x in intersect_vocab if x in vocab1]
        print("Vocabulary intersections loaded.")
        return intersect_vocab


    def _upper_triu_for_vocab(self, emb, emb_vocab, select_vocab):

        index = [emb_vocab.index(x) for x in select_vocab if x in emb_vocab]
        idxed_emb = emb[index][:]
        #cos_sim = cosine_similarity(idxed_emb)
        sim = euclidean_distances(idxed_emb)
        upper = sim[np.triu_indices(len(select_vocab),1)]

        return index, idxed_emb, sim, upper


    def _compare_to_emb(self, emb1, emb1_vocab, emb2, emb2_vocab, aoa):

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


    def get_pairwise_fold_corr(self):
        # Requires generation of list of all fold vectors
        corr_full_list = []
        corr_aoa_list = []
        print("Begin pairwise fold correlations:")
        n_tot_corr = sum(range(len(self.vec_list)))
        i=1
        for p, x in enumerate(self.vec_list):
            for q in range(p+1, len(self.vec_list)):
                print(f"Pair ({p}, {q}), #{i}/{n_tot_corr}")
                pw1 = euclidean_distances(self.vec_list[p])
                pw2 = euclidean_distances(self.vec_list[q])

                # Get correlation for all items
                corr_full = pearsonr(
                    pw1[np.triu_indices(len(self.all_fold_vocab), 1)],
                    pw2[np.triu_indices(len(self.all_fold_vocab), 1)])

                # Get correlation for AoA items
                aoa_idx = [
                    i for i, x in enumerate(self.all_fold_vocab)
                    if x in list(self.aoa["concept_i"])
                    ]
                pw_1 = euclidean_distances(
                    self.vec_list[p][aoa_idx,:]
                    )
                pw_2 = euclidean_distances(
                    self.vec_list[q][aoa_idx,:]
                    )
                
                corr_aoa = pearsonr(pw_1[np.triu_indices(len(aoa_idx), 1)],
                                    pw_2[np.triu_indices(len(aoa_idx), 1)])

                corr_full_list.append(corr_full[0])
                corr_aoa_list.append(corr_aoa[0])
                i += 1

        return corr_full_list, corr_aoa_list


    def get_fold_v_og_corr(self):
        # Get pairwise correlation of folds with original GloVe

        corr_og_list_full = []
        corr_og_list_aoa = []
        for x in self.vec_list:
                
            corr_full, corr_aoa = self._compare_to_emb(
                x, self.all_fold_vocab, self.wdnimg, self.og_vocab, self.aoa
                )
            corr_og_list_full.append(corr_full)
            corr_og_list_aoa.append(corr_aoa)

        return corr_og_list_full, corr_og_list_aoa


    def get_fold_v_other_corr(self, other_vocab_path, other_vec_path):
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


        corr_list_full = []
        corr_list_aoa = []
        for x in self.vec_list:
                
            corr_full, corr_aoa = self._compare_to_emb(
                x, self.all_fold_vocab, vecs_other, vocab_other, self.aoa
                )
            corr_list_full.append(corr_full)
            corr_list_aoa.append(corr_aoa)

        return corr_list_full, corr_list_aoa


if __name__ == "__main__":

    analyse = FoldAnalyser(
        os.path.join(Path(__file__).parent, "childes_nfold_20")
        )
    full_pw, aoa_pw = analyse.get_pairwise_fold_corr()
    full_og, aoa_og = analyse.get_fold_v_og_corr()
    full_text8, aoa_text8 = analyse.get_fold_v_other_corr(
        "/Users/apple/Documents/GitHub/GloVe_STATICCLONE/vocab_text8.txt",
        "/Users/apple/Documents/GitHub/GloVe_STATICCLONE/vectors_text8.txt"
    )

    print(np.mean(full_pw), np.mean(aoa_pw))
    print(np.mean(full_og), np.mean(aoa_og))
    print(np.mean(full_text8), np.mean(aoa_text8))