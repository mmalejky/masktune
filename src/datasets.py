import torch
from torchvision.datasets import MNIST
from tqdm import tqdm


class SpuriousMNIST(MNIST):

    classes = ['0 - first five digits', '1 - last five digits']
    
    spurious_feature = torch.zeros((3, 4, 4), dtype=torch.uint8)
    spurious_feature[2, :, :] = 255 # blue square

    def __init__(self, test_set=None, mask_func=None, class_mix_ratio=0.01, **kwargs):
        super().__init__(**kwargs)
        self.test_set = test_set

        self.data.unsqueeze_(1)
        self.data = self.data.repeat(1, 3, 1, 1)
        self.targets = (self.targets >= 5).long()
        
        if not self.train and not self.test_set:
            raise ValueError('Must specify test set when testing')
        if self.train:
            if self.test_set: raise ValueError('Cannot specify test set when training')
            temp = torch.ones(len(self.targets), dtype=torch.bool)
            temp[:int(len(temp)*class_mix_ratio)] = False
            spurious_shuffle = temp[torch.randperm(len(temp))]
            # whether particular index will got spurious feature based on its class
            spurious_feature_occurence = torch.logical_xor(self.targets, spurious_shuffle)
        elif self.test_set == 'raw':
            spurious_feature_occurence = torch.zeros(len(self.targets), dtype=torch.bool)
        elif self.test_set == 'malicious':
            spurious_feature_occurence = self.targets.bool()
        else:
            raise ValueError(f'Unknown test set: {self.test_set}')

        self.data[spurious_feature_occurence.nonzero(as_tuple=True)[0], :, 0:4, 0:4] = self.spurious_feature
        
        if mask_func is not None:
            # split data in batches of 128 and apply mask_func to every batch
            for i in tqdm(range(0, len(self.data), 128), desc='Masking data'):
                masked_data, _ = mask_func(self.data[i:i+128])
                self.data[i:i+128] = masked_data

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

