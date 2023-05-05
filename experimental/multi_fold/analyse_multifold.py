import numpy as np
import os



def get_all_vectors(folder):

    vec_list = []

    vec_files = [x for x in os.listdir(folder)
                if (x.split("_")[-1] == "vectors.txt.txt")
                and("full" not in x)]


    vocab_files =  [x for x in os.listdir(folder)
                    if ("vocab" in x) and ("full" not in x)]

    textfile = open(
                os.path.join(folder, vocab_files[0]),
                "r")
    intersect_vocab = textfile.read().split('\n')
    intersect_vocab = [x.split(" ")[0] for x in intersect_vocab]
    for i, f in enumerate(vocab_files[1:]):
        print(i)
        textfile = open(
            os.path.join(folder, f),
            "r")
        vocab1 = textfile.read().split('\n')
        vocab1 = [x.split(" ")[0] for x in vocab1]
        intersect_vocab = [x for x in intersect_vocab if x in vocab1]

    for f in vec_files:
        print(f)
        fs = f.split("_")[-1]
        n = f.split("_")[-2]
        n = n.replace("split", "")

        vecs = np.loadtxt(
            os.path.join(folder, f),
            usecols = [x for x in range(1,50)]
            )

        textfile = open(
            os.path.join(folder, f"vocab_{n}.txt"),
            "r")
        vocab2 = textfile.read().split('\n')
        vocab2 = [x.split(" ")[0] for x in vocab2]

        vocab_idxs_2 = [vocab2.index(x) for x in intersect_vocab]

        vecs = vecs[vocab_idxs_2,:]
        vec_list.append(vecs)

    return vec_list, intersect_vocab

#generate_fold_corpora("/Users/apple/Documents/GitHub/GloVe/multi_fold/multi_fold_enwik8/enwiki8_compiled.txt")
#run_embedding_for_og("/Users/apple/Documents/GitHub/GloVe/multi_fold/multi_fold_enwik8/enwiki8_compiled.txt")
#run_embedding_per_fold()



def upper_triu_for_vocab(emb, emb_vocab, select_vocab):

    index = [emb_vocab.index(x) for x in select_vocab if x in emb_vocab]
    idxed_emb = emb[index][:]
    #cos_sim = cosine_similarity(idxed_emb)
    cos_sim = euclidean_distances(idxed_emb)
    upper = cos_sim[np.triu_indices(len(select_vocab),1)]

    return index, idxed_emb, cos_sim, upper


def compare_to_emb(emb1, emb1_vocab, emb2, emb2_vocab, aoa):

    intersect_vocab = [x for x in emb1_vocab if x in emb2_vocab]

    __, __, __, upper1 = upper_triu_for_vocab(emb1, emb1_vocab, intersect_vocab)
    __, __, __, upper2 = upper_triu_for_vocab(emb2, emb2_vocab, intersect_vocab)

    corr = pearsonr(upper1, upper2)
    print(f"Correlation for all items in vocab: {corr}")

    intersect_vocab_aoa = [x for x in intersect_vocab if x in list(aoa["concept_i"])]

    __, __, __, upper1 = upper_triu_for_vocab(emb1, emb1_vocab, intersect_vocab_aoa)
    __, __, __, upper2 = upper_triu_for_vocab(emb2, emb2_vocab, intersect_vocab_aoa)

    corr2 = pearsonr(upper1, upper2)
    print(f"Correlation for early AoA in vocab: {corr2}")   

    return corr, corr2 


vec_list, vocab = get_all_vectors("/Users/apple/Documents/GitHub/GloVe/multi_fold/multi_fold_enwik8")
print("vec_list done")

wdnimg, ___, og_vocab = embs_quickload(50)
aoa = load_aoa_data()

# Get intersection with words of interest in intersection
vocab = [x for x in og_vocab if x in vocab]
intersect_idx = [vocab.index(x) for x in vocab]
vec_list = [x[intersect_idx,:] for x in vec_list]


## Get pairwise correlations of folds
"""
corr_full_list = []
corr_aoa_list = []
for i, p in enumerate(vec_list):
    for q in range(p+1, len(vec_list)):
        print(p,q)
        pw1 = euclidean_distances(vec_list[p])
        pw2 = euclidean_distances(vec_list[q])

        #corr_full, corr_aoa = compare_to_emb(vec_list[p], vocab, vec_list[q], vocab, aoa)
        corr_full = pearsonr(
            pw1[np.triu_indices(len(vocab), 1)],
            pw2[np.triu_indices(len(vocab), 1)])

        aoa_idx = [
            i for i, x in enumerate(vocab) if x in list(aoa["concept_i"])
            ]
        pw_1 = euclidean_distances(
            vec_list[p][aoa_idx,:]
            )
        pw_2 = euclidean_distances(
            vec_list[q][aoa_idx,:]
            )
        
        corr_aoa = pearsonr(pw_1[np.triu_indices(len(aoa_idx), 1)], pw_2[np.triu_indices(len(aoa_idx), 1)])

        corr_full_list.append(corr_full[0])
        corr_aoa_list.append(corr_aoa[0])
print(f"Mean pairwise cross-fold correlations (all): {np.mean(corr_full_list)}; (AoA): {np.mean(corr_aoa_list)}")
"""


# Get pairwise correlation of folds with original GloVe

corr_og_list_full = []
corr_og_list_aoa = []
for p in range(len(vec_list)):
        
    corr_full, corr_aoa = compare_to_emb(
        vec_list[p], vocab, wdnimg, og_vocab, aoa
        )
    corr_og_list_full.append(corr_full[0])
    corr_og_list_aoa.append(corr_aoa[0])

print(f"Mean fold correlations w og (all): {np.mean(np.array(corr_og_list_full))}; (AoA): {np.mean(np.array(corr_og_list_aoa))}")


# Correlation of full enwik8 corpus with original GloVe
"""
folder="/Users/apple/Documents/GitHub/GloVe/multi_fold/multi_fold_enwik8"
f="enwik8_full_vectors.txt"

vecs_full_enwik8 = np.loadtxt(
    os.path.join(folder, f),
    usecols = [x for x in range(1,50)]
)

f = "vocab_full.txt"
textfile = open(
            os.path.join(folder, f),
            "r")
vocab_enwik8 = textfile.read().split('\n')
vocab_enwik8 = [x.split(" ")[0] for x in vocab_enwik8]

corr_full, corr_aoa = compare_to_emb(
    vecs_full_enwik8, vocab_enwik8, wdnimg, og_vocab, aoa
    )
print(f"Full correlation w og (all): {corr_full[0]}; (AoA): {corr_aoa[0]}")
"""


# Correlation of full enwik8 corpus with text8
"""
vecs_full_text8 = np.loadtxt(
    "/Users/apple/Documents/GitHub/GloVe/vectors_text8.txt",
    usecols = [x for x in range(1,50)]
)
textfile = open(
            "/Users/apple/Documents/GitHub/GloVe/vocab_text8.txt",
            "r")
vocab_text8 = textfile.read().split('\n')
vocab_text8 = [x.split(" ")[0] for x in vocab_text8]


intersect_vocab = [x for x in og_vocab if (x in vocab_text8) and (x in vocab_enwik8)]
idxs_enwik8 = [vocab_enwik8.index(x) for x in intersect_vocab]
idxs_text8 = [vocab_text8.index(x) for x in intersect_vocab]


corr_full, corr_aoa = compare_to_emb(vecs_full_enwik8[idxs_enwik8,:], intersect_vocab, vecs_full_text8[idxs_text8,:], intersect_vocab, aoa)
print(f"Full correlation w text8 (all): {corr_full[0]}; (AoA): {corr_aoa[0]}")
"""