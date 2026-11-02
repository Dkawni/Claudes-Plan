'''
Dataset loader variant that can return a class-balanced subsample.

Kernel SVM (sklearn's SVC) scales roughly quadratically with the number of
training samples, so fitting it on all 60,000 MNIST images takes hours. This
loader allows a baseline to be trained on a smaller, stratified subset so the
comparison scripts finish in reasonable time. The subset keeps the original
class proportions, and the seed is fixed so runs are reproducible.
'''

from sklearn.model_selection import train_test_split

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader


class Dataset_Subsample(Dataset_Loader):
    # None (default) means "use every row"
    max_samples = None
    random_state = 1

    def load(self):
        data = super().load()

        total = len(data['y'])
        if not self.max_samples or self.max_samples >= total:
            return data

        # stratify so every digit stays represented in its original proportion
        keep_index, _ = train_test_split(
            range(total),
            train_size=self.max_samples,
            stratify=data['y'],
            random_state=self.random_state,
        )
        keep_index = sorted(keep_index)

        print('subsampling {} -> {} rows (stratified)'.format(
            total, len(keep_index)))

        return {
            'X': [data['X'][i] for i in keep_index],
            'y': [data['y'][i] for i in keep_index],
        }
