from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import sys
from pkgutil import simplegeneric
from transformers import XLNetTokenizer, XLNetModel
from keras.preprocessing.sequence import pad_sequences
from torch import nn
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset


class Clinic(CombinationMetaDataset):

    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 use_vinyals_split=True, transform=None, target_transform=None,
                 dataset_transform=None, class_augmentations=None, download=False):
        dataset = ClinicDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test,
            use_vinyals_split=use_vinyals_split, transform=transform,
            meta_split=meta_split, class_augmentations=class_augmentations,
            download=download)
        super(Clinic(root), self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class ClinicDataset(Dataset):

    def __init__(self, index, data, classLabel):

        """
          Given index, data, classLabel return a Dataset for only that class.
          Data is a dataframe with just classLabel entries.
        """
        self.index = index
        self.data = data
        self.classLabel = classLabel
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.max_len = 64

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):

        text = str(self.data["text"][item])
        label = self.data["label"][item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = pad_sequences(encoding['input_ids'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids)

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask)

        return [input_ids, attention_mask.flatten(), torch.tensor(label, dtype=torch.long)]


class ClinicClassDataset(ClassDataset):

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None,
                 class_augmentations=None, download=False):
        super(ClinicClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        # Only populate the dataframe the first time around
        if not self.df:
          self.df = self.process_data()

        self.df = self.df.assign(id=self.df.index.values)
        self._data = None
        self._labels = None
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        """
          This should go ahead and get the class from the dataset and pass that through ClinicDataset
        """
        data = self.df[index]
        classLabel = data['label']

        return ClinicDataset(index, data, classLabel)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.split_filename, 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def process_data(self):
        """
            Subset tells you whether to return train, val or test.
        """
        # Open a file: file
        fileName = os.getcwd() + '/data_small.json'
        file = open(fileName, mode='r')
        all_of_it = file.read()
        label_mapping = {}

        @simplegeneric
        def get_items(obj):
            while False: # no items, a scalar object
                yield None

        @get_items.register(dict)
        def _(obj):
            return obj.items() # json object. Edit: iteritems() was removed in Python 3

        @get_items.register(list)
        def _(obj):
            return enumerate(obj) # json array

        def strip_whitespace(json_data):
            for key, value in get_items(json_data):
                if hasattr(value, 'strip'): # json string
                    json_data[key] = value.strip()
                else:
                    strip_whitespace(value) # recursive call

        data = json.loads(all_of_it) # read json data from standard input
        strip_whitespace(data)

        labels = []
        for text, label in data[self.subset]:
            labels.append(label)

        label_set = set(labels)
        label_mapping = {}

        index = 0
        for label in label_set:
            label_mapping[label] = index
            index += 1

        # Convert into dataframe
        embedded = []

        for text, label in data[self.subset]:
            row = {"text": text, "label": label_mapping[label]}
            embedded.append(row)

        df = pd.DataFrame(embedded)

        return df



