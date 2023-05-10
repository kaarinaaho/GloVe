
import numpy as np
import pandas as pd
from pathlib import Path
import os

def embs_quickload(n_word_dim=300):
    # Quickload
    if n_word_dim==300:
        wdnimg = np.loadtxt(
            os.path.join(Path(__file__).parent, "assets/embeddings/wdnimg300.txt")
            )
    elif n_word_dim==50:
        wdnimg = np.loadtxt(
            os.path.join(Path(__file__).parent, "assets/embeddings/wdnimg50.txt")
            )
    imgnwd = np.loadtxt(
            os.path.join(Path(__file__).parent, "assets/embeddings/imgnwd50.txt")
            )

    wdnimg = scale_array(wdnimg, -1, 1)
    imgnwd = scale_array(imgnwd, -1, 1)


    textfile = open(
        os.path.join(Path(__file__).parent, "assets/embeddings/vocab.txt"), "r"
        )
    vocab = textfile.read().split('\n')

    return wdnimg, imgnwd, vocab


def scale_array(a, newmin, newmax):

    newrng = newmax-newmin
    a = ((a-np.amin(a, axis=0))/(np.amax(a, axis=0)-np.amin(a, axis=0)) * newrng) + newmin

    return a


def load_aoa_data(acquisition_threshold=None):

    aoa_fp = os.path.join(Path(__file__).parent, "assets/aoa/aoa_data_EN.csv")
    
    # Import aoa data and match column names
    aoa = pd.read_csv(aoa_fp, header=0, index_col=0)
    aoa = aoa.rename(columns={"definition": "concept_i"})

    if acquisition_threshold is not None:
        # Find age (in months) at which threshold acquisition is achieved
        mins = pd.melt(
            aoa, id_vars=["concept_i", "category", "type"],
            var_name="age", value_name="percent"
            )
        mins["above_thresh"] = mins["percent"] >= acquisition_threshold
        mins = mins[["concept_i", "age"]][mins["above_thresh"] == True]
        mins = mins.groupby("concept_i").agg("min")
        mins = mins.rename(columns={"age": "age_thresh_ac"})

        aoa = aoa.merge(mins, how="inner", on="concept_i")
        aoa["thresh"] = acquisition_threshold
        aoa = aoa.sort_values(by=["age_thresh_ac"])
        aoa.reset_index(inplace=True, drop=True)

    return aoa

