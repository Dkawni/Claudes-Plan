'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

<<<<<<< Updated upstream
from local_code.base_class.dataset import dataset
import pickle
import numpy as np
=======
import pickle
import numpy as np
from local_code.base_class.dataset import dataset
>>>>>>> Stashed changes


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
<<<<<<< Updated upstream
    dataset_source_file_name = None   # 'MNIST', 'ORL' or 'CIFAR'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

## change to numpy arrays
    def to_numpy_arrays(self, instances):
        X, y = [], []
        for inst in instances:
            img = np.array(inst['image'], dtype=np.float32) / 255.0
            if self.dataset_source_file_name == 'MNIST':
                img = img[np.newaxis, :, :]
            elif self.dataset_source_file_name == 'ORL':
                img = img[:, :, 0][np.newaxis, :, :]
            else:
                img = img.transpose(2, 0, 1)
            X.append(img)
            y.append(int(inst['label']))
        X = np.stack(X)
        y = np.array(y)
        if self.dataset_source_file_name == 'ORL':
            y = y - 1
        return X, y


    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        train_X, train_y = self.to_numpy_arrays(data['train'])
        test_X, test_y = self.to_numpy_arrays(data['test'])
        return {'train': {'X': train_X, 'y': train_y},
                'test': {'X': test_X, 'y': test_y}}
=======
    dataset_source_file_name = None
    dName = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.dName = dName
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        if self.dName == 'train':
            for instance in data['train']:
                image = instance['image']
                image = image / 255.0
                X.append(image.flatten())
                y.append(instance['label'])
        else:
            if self.dName == 'test':
                for instance in data['test']:
                    image = instance['image']
                    image = image / 255.0
                    X.append(image.flatten())
                    y.append(instance['label'])
        return {'X': X, 'y': y}
>>>>>>> Stashed changes
