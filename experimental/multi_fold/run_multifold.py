import subprocess
import random
import numpy as np
from pathlib import Path
import os, sys
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import shutil

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
from utils import embs_quickload


from scipy.stats import spearmanr, pearsonr


class SplitEmbGenerator:
    def __init__(self, corpusfile, split_type="nfold", n_folds=20,
                n_documents=None, max_docsize=None):
        """ Init. """

        # Strip corpus name of filepath/extensions
        raw_name = self._get_raw_name(corpusfile)
        self.corpusname = raw_name

        # Load original corpus from corpora/processed directory
        self.corpus_fp = os.path.join(
            Path(__file__).parent.parent, 
            f"corpora/processed/{raw_name}.txt")
        
        self.n_folds = n_folds

        self.results_folder = os.path.join(
            Path(__file__).parent, f"{raw_name}_{split_type}_{n_folds}/results"
            )
        
        # If directory does not exist, create
        if not os.path.exists(Path(self.results_folder).parent):
            os.mkdir(Path(self.results_folder).parent)

        else:
            if (
                os.path.exists(os.path.join(self.results_folder, "splits"))
                ) | (
                os.path.exists(os.path.join(self.results_folder, "vocab"))
                ) | (
                os.path.exists(os.path.join(self.results_folder, "vectors"))
                ) | (
                os.path.exists(os.path.join(self.results_folder, "coocc"))
            ):
                reply = input(f"Folders in {self.results_folder} will be overwritten. Enter 'y' to continue")

                assert(reply == "y")
        if not os.path.exists(Path(self.results_folder)):
            os.mkdir(self.results_folder)
        
        for f in ["splits", "vocab", "vectors", "coocc"]:
            p = os.path.join(self.results_folder, f)
            if os.path.exists(p):
                shutil.rmtree(p)
            os.mkdir(p)
        
        self.split_type = split_type
        if (split_type == "fixedsize"):
            if ((n_documents is None)):
                raise ValueError("n_documents must be specified in order to run a fixedsize split")
            self.n_documents = n_documents
            self.max_docsize = max_docsize

    def _get_raw_name(self, x):
        # Remove filepath
        if "/" in x:
            x = x.split("/")[-1]
        # Remove extension
        if "." in x:
            x = x.split(".")[0]
        return x

    def generate_split_corpora(self):
        output_folder = os.path.join(
            self.results_folder, "splits"
        )

        # Open original corpus file
        with open(self.corpus_fp, "r") as f:
            if self.split_type == "nfold":
                self._generate_fold_corpora(output_folder, f)

            elif self.split_type == "fixedsize":
                self._generate_sample_corpora(output_folder, f)
        f.close()

    def _generate_fold_corpora(self, output_folder, f):
        """Split complete corpus into n folds."""


        lines = f.readlines()

        n_lines = len(lines)
        min_lines_per_fold = int(np.floor(n_lines/self.n_folds))
        # Remainder folds need an additional line
        n_w_additional_line = int(n_lines % self.n_folds)

        # Each sublist is a set of lines to exclude
        lines_list = []
        all_lines = [x for x in range(n_lines)]
        for fold in range(self.n_folds):
            n_add = min_lines_per_fold

            # Add additional line to required number of lines
            if fold < n_w_additional_line:
                n_add += 1
            
            # Sample the lines to be excluded for fold
            sample = random.sample(all_lines, n_add)
            lines_list.append(sample)
            
            # Exclude each line from only one fold
            all_lines = [x for x in all_lines if x not in sample]

        # For each fold, get index of lines to exclude
        for i, exc_idx in enumerate(lines_list):
            # Get all lines in corpus but those in exc index
            corpus_split = [lines[x] for x in range(n_lines)
                            if x not in exc_idx]
            with open(
                os.path.join(
                    output_folder, f"{self.corpusname}_split{i}.txt"
            ),'w') as out:
                for line in corpus_split:
                    out.write(f"{line}")

    def _generate_sample_corpora(self, output_folder, f):

        lines = f.readlines()
        n_lines = len(lines)

        # Random sort of n_lines
        rand_sort = random.sample(list(range(n_lines)), n_lines)

        # Collect list of sampled line sets
        lines_list = []

        for x in range(self.n_folds):
            i = 0

            # Collect list for this sample
            i_list = []
            
            # Select as many documents as specified for each set
            while len(i_list) < self.n_documents:

                # Get line and split into tokens
                cand = lines[rand_sort[i]]
                cand_list = cand.split(" ")
                
                if (self.max_docsize is not None):
                    # If doc is over max doc size, take a window
                    if len(cand_list) > self.max_docsize:
                        # get min starting value for window
                        min_start = len(cand_list) - self.max_docsize

                        #randomly select starting value for window
                        start = random.randint(0,min_start)

                        # get subset of doc
                        sub = cand_list[start:start+self.max_docsize]

                    else:
                        sub = cand_list

                else:
                    sub = cand_list

                #re-string
                sub = " ".join(sub)

                # Add line to list for this sample
                i_list.append(sub)

                # Increment number of selected lines
                i += 1
            # When sample complete, append sample to list of samples
            lines_list.append(i_list)

        for i, corpus_split in enumerate(lines_list):
            with open(
                os.path.join(
                    output_folder, f"{self.corpusname}_split{i}.txt"
            ),'w') as out:
                for line in corpus_split:
                    out.write(f"{line}")

    def run_embedding_per_fold(self):

        # Glove script path;
        glove_script = os.path.join(
            Path(__file__).parent.parent.parent,
            "demo_amended.sh")

        for i in range(self.n_folds):
            # Get output path names relative to glove script folder
            corpus_fn = f"experimental/multi_fold/{self.corpusname}_{self.split_type}_{self.n_folds}/results/splits/{self.corpusname}_split{i}.txt"
            vocab_fn = f"experimental/multi_fold/{self.corpusname}_{self.split_type}_{self.n_folds}/results/vocab/{i}.txt"
            coocc_fn = f"experimental/multi_fold/{self.corpusname}_{self.split_type}_{self.n_folds}/results/coocc/{i}.bin"
            coocc_shuf_fn = f"experimental/multi_fold/{self.corpusname}_{self.split_type}_{self.n_folds}/results/coocc/{i}.shuf.bin"
            vec_fn = f"experimental/multi_fold/{self.corpusname}_{self.split_type}_{self.n_folds}/results/vectors/{i}"


            # input file names as relative, see if this works
            try:
                output = subprocess.check_call(
                    [glove_script,
                    corpus_fn, vocab_fn, coocc_fn, coocc_shuf_fn, vec_fn])

            except subprocess.CalledProcessError as e:
                output = e.output
                print(output)


if __name__ == "__main__":

    generator = SplitEmbGenerator("enwiki8.txt")
    generator.generate_split_corpora()
    generator.run_embedding_per_fold()
