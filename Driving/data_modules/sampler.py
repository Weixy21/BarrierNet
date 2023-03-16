import torch


class ImbalanceDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self._dataset = dataset
        self._num_samples = len(dataset)
        self._idcs = list(range(self._num_samples))

    @property
    def num_samples(self):
        return self._num_samples

    def __iter__(self):
        return (self._idcs[i] for i in torch.)

    def __len__(self):
        return self.num_samples
