## https://github.com/sujitpal/dl-models-for-qa
## convert python2 > 3
## it is available without keras

# -*- coding: utf-8 -*-
from __future__ import division, print_function
import collections
import re
import nltk
import numpy as np
import os
from functools import reduce

def pad_sequences(sentences, maxlen=500, value = 0):
    import numpy as np
    padded_sentences =  np.full((len(sentences), maxlen), value)
    for idx, sentence in enumerate(sentences):
        new_sentence = np.full((1, maxlen), value)
        if (len(sentence) > maxlen):
            new_sentence = sentence[:maxlen]
        else:
            new_sentence = np.append([value]*(maxlen-len(sentence)), sentence)
    
        padded_sentences[idx] = new_sentence
    return padded_sentences

def get_files_for_task(task_nbr, babi_dir):
    filenames = os.listdir(babi_dir)
    task_files = list(filter(lambda x: re.search("qa%d_" % (task_nbr), x), filenames))
    print (task_files)
    assert(len(task_files) == 2)
    train_file = list(filter(lambda x: re.search("_train.txt", x), task_files))[0]
    test_file = list(filter(lambda x: re.search("_test.txt", x), task_files))[0]
    return train_file, test_file

def get_stories(taskfile, only_support=False):
    data = []
    story_sents = []
    ftask = open(taskfile, "r")
    for line in ftask:
        line = line.strip()
        nid, line = line.split(" ", 1)
        if int(nid) == 1:
            # new story
            story_sents = []
        if "\t" in line:
            # capture question, answer and support
            q, a, support = line.split("\t")
            q = nltk.word_tokenize(q)
            if only_support:
                # only select supporting sentences
                support_idxs = [int(x)-1 for x in support.split(" ")]
                story_so_far = []
                for support_idx in support_idxs:
                    story_so_far.append(story_sents[support_idx])
            else:
                story_so_far = [x for x in story_sents]
            story = reduce(lambda a, b: a + b, story_so_far)
            data.append((story, q, a))
        else:
            # only capture story
            story_sents.append(nltk.word_tokenize(line))
    ftask.close()
    return data

def build_vocab(daten):
    counter = collections.Counter()
    for data in daten:
        for story, question, answer in data:
            for w in story:
                counter[w] += 1
            for w in question:
                counter[w] += 1
            for w in [answer]:
                counter[w] += 1
    # don't throw away anything because we don't have many words
    # in the synthetic dataset.
    # also we want to reserve 0 for pad character, so we offset the
    # indexes by 1.
    words = [wordcount[0] for wordcount in counter.most_common()]
    word2idx = {w: i+1 for i, w in enumerate(words)}
    return word2idx

def get_maxlens(daten):
    """ Return the max number of words in story and question """
    data_comb = []
    for data in daten:
        data_comb.extend(data)
    story_maxlen = max([len(x) for x, _, _ in data_comb])
    question_maxlen = max([len(x) for _, x, _ in data_comb])
    return story_maxlen, question_maxlen

def vectorize(data, word2idx, story_maxlen, question_maxlen):
    """ Create the story and question vectors and the label """
    Xs, Xq, Y = [], [], []
    for story, question, answer in data:
        xs = [word2idx[word] for word in story]
        xq = [word2idx[word] for word in question]
        y = np.zeros(len(word2idx) + 1)
        y[word2idx[answer]] = 1
        Xs.append(xs)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(Xs, maxlen=story_maxlen), 
            pad_sequences(Xq, maxlen=question_maxlen),
            np.array(Y))

