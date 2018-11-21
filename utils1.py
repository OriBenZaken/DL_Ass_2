import numpy as np
import torch as tc
from itertools import izip
EMBEDDING_VECTOR_SIZE = 50
START = '*START*'
END = '*END*'

def get_word_embeddings_dict_from_file(words_file, vector_file):
    word_embeddings_dict = {}
    for word, vector_line in izip(open(words_file), open(vector_file)):
        word = word.strip("\n").strip()
        vector_line = vector_line.strip("\n").strip().split(" ")
        word_embeddings_dict[word] = np.asarray(map(float,vector_line))

    word_embeddings_dict[START] = np.random.uniform(-1,1,[1, EMBEDDING_VECTOR_SIZE])
    word_embeddings_dict[END] = np.random.uniform(-1,1,[1, EMBEDDING_VECTOR_SIZE])
    return word_embeddings_dict

WORD_EMBEDDINGS_DICT = get_word_embeddings_dict_from_file('vocab.txt', 'wordVectors.txt')
TAGS_SET = set()
WORDS_SET = set()
WORD_TO_INDEX = {}
INDEX_TO_WORD = {}
TAG_TO_INDEX = {}
INDEX_TO_TAG = {}

def read_train_data(file_name):
    global WORDS_SET, TAGS_SET
    tagged_sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence_and_tags = []
        for line in content:
            if line == "\n":
                tagged_sentences.append(sentence_and_tags)
                sentence_and_tags =[]
                continue
            line = line.strip("\n").strip()
            word, tag  = line.split(" ")
            TAGS_SET.add(tag)
            WORDS_SET.add(word)
            sentence_and_tags.append((word, tag))
    return tagged_sentences

def load_indexers(word_set, tags_set):
    global  WORD_TO_INDEX, INDEX_TO_WORD, TAG_TO_INDEX, INDEX_TO_TAG
    word_set.update(set([START, END]))
    WORD_TO_INDEX = {word : i for i, word in enumerate(word_set)}
    INDEX_TO_WORD = {i : word for word, i in WORD_TO_INDEX.iteritems()}
    TAG_TO_INDEX = {tag : i for i, tag in enumerate(tags_set)}
    INDEX_TO_TAG = {i : tag for tag, i in TAG_TO_INDEX.iteritems()}



def get_windows_and_tags(tagged_sentences):
    concat_words = []
    tags = []
    for sentence in tagged_sentences:
        pad_s = [(START, START), (START, START)]
        pad_s.extend(sentence)
        pad_s.extend([(END, END), (END, END)])
        for i, (word,tag) in enumerate(pad_s):
            if word!=START and word !=END:
                liz = pad_s[i-2][0]

                win = [WORD_TO_INDEX[pad_s[i-2][0]], WORD_TO_INDEX[pad_s[i-1][0]],
                       WORD_TO_INDEX[word], WORD_TO_INDEX[pad_s[i+1][0]], WORD_TO_INDEX[pad_s[i+2][0]]]
                concat_words.append(win)
                tags.append(tag)

    return concat_words,tags


    pass

def get_train_data(file_name):
    global WORDS_SET, TAGS_SET
    tagged_sentences_list = read_train_data(file_name)
    load_indexers(WORDS_SET, TAGS_SET)
    concat, tags = get_windows_and_tags(tagged_sentences_list)
    x = 'fuck'



get_train_data('pos/train')

