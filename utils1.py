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
    """
    get_word_embeddings_dict_from_file function.
    reads the words and word embeddings vectors and fills word_embeddings_dict
    :param words_file: name of the file containing the words
    :param vector_file: name of the file containing the vectors of the words
    from the words_file
    :return: word embeddings dictionary
    """
    word_embeddings_dict = {}
    for word, vector_line in izip(open(words_file), open(vector_file)):
        word = word.strip("\n").strip()
        vector_line = vector_line.strip("\n").strip().split(" ")
        word_embeddings_dict[word] = np.asanyarray(map(float,vector_line))
    return word_embeddings_dict

def read_tagged_data(file_name, is_dev = False):
    """
    read_tagged_data function.
    reads dev and train from files and returns list of tagged sentences.
    in case we read the train, we also fill the WORDS_SET and TAGS_SET
    :param file_name: name of file to read.
    :param is_dev: indicates if the file is validation file.
    :return: list of tagged sentences
    """
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
    """
    read_not_tagged_data function.
    reads the test file.
    :param file_name: test file name.
    :return: list of sentences.
    """
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
    """
    load_indexers function.
    creates our dicts that helps us to manage the data.
    :param word_set: our words.
    :param tags_set: our tags.
    """
    global  WORD_TO_INDEX, INDEX_TO_WORD, TAG_TO_INDEX, INDEX_TO_TAG
    word_set.update(set([START, END]))
    WORD_TO_INDEX = {word : i for i, word in enumerate(word_set)}
    INDEX_TO_WORD = {i : word for word, i in WORD_TO_INDEX.iteritems()}
    TAG_TO_INDEX = {tag : i for i, tag in enumerate(tags_set)}
    INDEX_TO_TAG = {i : tag for tag, i in TAG_TO_INDEX.iteritems()}



def get_windows_and_tags(tagged_sentences):
    """
    get_windows_and_tags function.
    :param tagged_sentences: examples.
    :return: concat of five window of words and tags.
    """
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
                concat_words.append(win)
                tags.append(TAG_TO_INDEX[tag])
    return concat_words, tags

def get_windows(sentences):
    """
    get_windows function.
    :param sentences list.
    :return: concat of five window of words.
    """
    concat_words = []
    for sentence in sentences:
        pad_s = [START,START]
        pad_s.extend(sentence)
        pad_s.extend([END,END])
        for i, (word) in enumerate(pad_s):
            if word != START and word != END:
                win = get_word_indices_window(pad_s[i - 2],pad_s[i - 1],word,pad_s[i + 1],pad_s[i + 2])
                concat_words.append(win)
    return concat_words

def get_word_indices_window(w1,w2,w3,w4,w5):
    """
    get_word_indices_window function.
    :param w1: word1
    :param w2: word2
    :param w3: word3
    :param w4: word4
    :param w5: word5
    :return: concat words window indices.
    """
    win = []
    win.append(get_word_index(w1))
    win.append(get_word_index(w2))
    win.append(get_word_index(w3))
    win.append(get_word_index(w4))
    win.append(get_word_index(w5))
    return win

def get_word_index(w):
    """
    get_word_index function.
    :param w: requested word index.
    :return: word index if its in words set, or unk index.
    """
    if w in WORD_TO_INDEX:
        return WORD_TO_INDEX[w]
    else:
        return WORD_TO_INDEX[UNK]

def get_tagged_data(file_name,is_dev = False):
    """
    get_tagged_data function.
    :param file_name: file name of the requested data for dev or train.
    :param is_dev:
    :return: data and tags
    """
    global WORDS_SET, TAGS_SET
    tagged_sentences_list = read_tagged_data(file_name, is_dev)
    if not is_dev:
        load_indexers(WORDS_SET,TAGS_SET)
    concat, tags = get_windows_and_tags(tagged_sentences_list)
    return concat, tags

def get_not_tagged_data(file_name):
    """
    get_not_tagged_data function.
    :param file_name: file name of the requested data for test.
    :return: data
    """
    global WORDS_SET, TAGS_SET
    sentences_list = read_not_tagged_data(file_name)
    concat = get_windows(sentences_list)
    return concat


WORD_EMBEDDINGS_DICT = get_word_embeddings_dict_from_file('vocab.txt', 'wordVectors.txt')


