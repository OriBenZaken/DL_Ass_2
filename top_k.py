import utils1 as ut1
import torch
from heapq import nlargest
import torch.nn.functional
import math
import numpy as np
from numpy import linalg as la


def main():
    print most_similar("dog", 5)
    print most_similar("england", 5)
    print most_similar("john", 5)
    print most_similar("explode", 5)
    print most_similar("office", 5)



def most_similar(word, k):

    word_embedding_dict = ut1.WORD_EMBEDDINGS_DICT
    u = word_embedding_dict[word]
    words_distances = []
    for one_word in word_embedding_dict:
        calc = cosine_distance(u, word_embedding_dict[one_word])
        words_distances.append([one_word, calc])

    words_distances = sorted(words_distances, key=get_distance)
    top_k = sorted(words_distances, key=get_distance,reverse=True)[0:k]
    top_k = [item[0] for item in top_k]
    return top_k

def cosine_distance(u, v):
    d = np.max([float(la.norm(u, 2) * la.norm(v,2)), 1e-8])
    n = np.dot(u, v)
    return  n / d

def get_distance(word_and_distance):
    return word_and_distance[1]

if __name__ == "__main__":
    main()