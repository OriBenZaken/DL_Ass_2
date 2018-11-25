import numpy as np
import torch as tc
from itertools import izip

#globals
EMBEDDING_VECTOR_SIZE = 50
START = '*START*'
END = '*END*'
UNK = "UUUNKKK"
TAGS_SET = set()
WORDS_SET = set()
WORD_TO_INDEX = {}
INDEX_TO_WORD = {}
TAG_TO_INDEX = {}
INDEX_TO_TAG = {}

def get_word_embeddings_dict_from_file(words_file, vector_file):
    word_embeddings_dict = {}
    for word, vector_line in izip(open(words_file), open(vector_file)):
        word = word.strip("\n").strip()
        vector_line = vector_line.strip("\n").strip().split(" ")
        word_embeddings_dict[word] = np.asanyarray(map(float,vector_line))

    #word_embeddings_dict[START] = np.random.uniform(-1,1,[1, EMBEDDING_VECTOR_SIZE])
    #word_embeddings_dict[END] = np.random.uniform(-1,1,[1, EMBEDDING_VECTOR_SIZE])
    return word_embeddings_dict



def read_tagged_data(file_name, is_dev = False):
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
            line = line.strip("\n").strip().strip("\t")
            word, tag  = line.split()
            if not is_dev:
                TAGS_SET.add(tag)
                WORDS_SET.add(word)
            sentence_and_tags.append((word, tag))
    if not is_dev:
        TAGS_SET.add(UNK)
        WORDS_SET.add(UNK)
    return tagged_sentences

def read_not_tagged_data(file_name):
    sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence = []
        for line in content:
            if line == "\n":
                sentences.append(sentence)
                sentence =[]
                continue
            w = line.strip("\n").strip()
            sentence.append(w)
    return sentences

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
                win = get_word_indices_window(pad_s[i - 2][0], pad_s[i - 1][0], word,
                                              pad_s[i + 1][0], pad_s[i + 2][0])
                # win = [WORD_TO_INDEX[pad_s[i-2][0]], WORD_TO_INDEX[pad_s[i-1][0]],
                #        WORD_TO_INDEX[word], WORD_TO_INDEX[pad_s[i+1][0]], WORD_TO_INDEX[pad_s[i+2][0]]]
                concat_words.append(win)
                tags.append(TAG_TO_INDEX[tag])
    return concat_words, tags

def get_windows(sentences):
    concat_words = []
    for sentence in sentences:
        pad_s = [START,START]
        pad_s.extend(sentence)
        pad_s.extend([END,END])
        for i, (word) in enumerate(pad_s):
            if word != START and word != END:
                win = get_word_indices_window(pad_s[i - 2],pad_s[i - 1],word,pad_s[i + 1],pad_s[i + 2])
                # win = [WORD_TO_INDEX[pad_s[i - 2]], WORD_TO_INDEX[pad_s[i - 1],pad_s[i + 1]],
                #        WORD_TO_INDEX[word], WORD_TO_INDEX[pad_s[i + 1]], WORD_TO_INDEX[pad_s[i + 2]]]
                concat_words.append(win)
    return concat_words

def get_word_indices_window(w1,w2,w3,w4,w5):
    win = []
    win.append(get_word_index(w1))
    win.append(get_word_index(w2))
    win.append(get_word_index(w3))
    win.append(get_word_index(w4))
    win.append(get_word_index(w5))
    return win

def get_word_index(w):
    if w in WORD_TO_INDEX:
        return WORD_TO_INDEX[w]
    else:
        return WORD_TO_INDEX[UNK]

def get_tagged_data(file_name,is_dev = False):
    global WORDS_SET, TAGS_SET
    tagged_sentences_list = read_tagged_data(file_name, is_dev)
    if not is_dev:
        load_indexers(WORDS_SET,TAGS_SET)
    concat, tags = get_windows_and_tags(tagged_sentences_list)
    return concat, tags

def get_not_tagged_data(file_name):
    global WORDS_SET, TAGS_SET
    sentences_list = read_not_tagged_data(file_name)
    concat = get_windows(sentences_list)
    return concat


WORD_EMBEDDINGS_DICT = get_word_embeddings_dict_from_file('vocab.txt', 'wordVectors.txt')

#load_indexers(WORDS_SET, TAGS_SET)

