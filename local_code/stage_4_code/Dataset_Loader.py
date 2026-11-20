'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
#
from local_code.base_class.dataset import dataset
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset
import glob
import os
import re


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    vocab = None
    max_len = 300


    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def tokenize(self, text):
        text = text.lower().replace('<br />', ' ')
        tokens = re.findall(r"[a-z0-9']+", text)
        return tokens[:self.max_len]

    def build_vocab(self, tokenized_texts, min_freq=2, max_size=None):
        """
        Build a token->id mapping from a list of token lists.

        tokenized_texts: List[List[str]]  (e.g. positive_texts + negative_texts)
        min_freq:        drop tokens seen fewer than this many times
        max_size:        optional cap on vocab size (including specials)
        """
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        # Reserve special IDs first so they never shift
        vocab = {"<pad>": 0, "<unk>": 1}

        # Sort by frequency desc, then alphabetically for determinism
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        for token, freq in items:
            if freq < min_freq:
                continue
            if max_size is not None and len(vocab) >= max_size:
                break
            vocab[token] = len(vocab)

        return vocab

    def load(self):
        X = []
        Y = []
        positive_texts = []
        negative_texts = []

        story_files = glob.glob(os.path.join(self.dataset_source_folder_path, "pos/*.txt"))
        for f in story_files:
            with open(f, "r", encoding="utf-8") as fh:
                positive_texts.append(self.tokenize(fh.read()))
        story_files = glob.glob(os.path.join(self.dataset_source_folder_path, "neg/*.txt"))
        for f in story_files:
            with open(f, "r", encoding="utf-8") as fh:
                negative_texts.append(self.tokenize(fh.read()))

        if self.vocab is None:
            vocab = self.build_vocab(positive_texts + negative_texts)
        else:
            vocab = self.vocab
        for review in positive_texts:
            X.append([vocab.get(t, vocab['<unk>']) for t in review])
            Y.append(1)
        for review in negative_texts:
            X.append([vocab.get(t, vocab['<unk>']) for t in review])
            Y.append(0)

        return {'X': X, 'Y': Y, 'dictionary': vocab}