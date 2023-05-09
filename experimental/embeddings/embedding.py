
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


class EmbeddingInference():
    def __init__(self, corpusfile):
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
            Path(__file__).parent, f"/results"
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
                reply = input(f"Files in {self.results_folder} may be overwritten. Enter 'y' to continue")

                assert(reply == "y")
        if not os.path.exists(Path(self.results_folder)):
            os.mkdir(self.results_folder)
        
        for f in ["splits", "vocab", "vectors", "coocc"]:
            p = os.path.join(self.results_folder, f)
            if os.path.exists(p):
                shutil.rmtree(p)
            os.mkdir(p)

    def _get_raw_name(self, x):
        # Remove filepath
        if "/" in x:
            x = x.split("/")[-1]
        # Remove extension
        if "." in x:
            x = x.split(".")[0]
        return x

    def run_embedding(self):

        # Glove script path;
        glove_script = os.path.join(
            Path(__file__).parent.parent.parent,
            "demo_amended.sh")

        for i in range(self.n_folds):
            # Get output path names relative to glove script folder
            corpus_fn = f"experimental/corpora/processed/{self.corpusname}.txt"
            vocab_fn = f"experimental/embeddings/results/vocab/{self.corpusname}_vocab.txt"
            coocc_fn = f"experimental/embeddings/results/cooc/{self.corpusname}_cooc.bin"
            coocc_shuf_fn = f"experimental/embeddings/results/cooc/{self.corpusname}_cooc.shuf.bin"
            vec_fn = f"experimental/embeddings/results/vectors/{self.corpusname}_vectors.txt"

            try:
                output = subprocess.check_call(
                    [glove_script,
                    corpus_fn, vocab_fn, coocc_fn, coocc_shuf_fn, vec_fn])

            except subprocess.CalledProcessError as e:
                output = e.output
                print(output)


if __name__ == "__main__":

    emb = EmbeddingInference("enwik8")
    emb.run_embedding()
