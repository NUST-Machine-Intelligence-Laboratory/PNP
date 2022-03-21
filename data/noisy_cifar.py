import numpy as np
from numpy.testing import assert_array_almost_equal
import torchvision
from PIL import Image


# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def generate_noise_matrix(noise_type, closeset_noise_ratio, openset_noise_ratio=0.8, nb_classes=10):
    """

    Example of the noise transition matrix (closeset_ratio = 0.3):
        - Symmetric:
            -                               -
            | 0.7  0.1  0.1  0.1  0.0  0.0  |
            | 0.1  0.7  0.1  0.1  0.0  0.0  |
            | 0.1  0.1  0.7  0.1  0.0  0.0  |
            | 0.1  0.1  0.1  0.7  0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            -                               -
        - Asymmetric (Pairflip)
            -                               -
            | 0.7  0.3  0.0  0.0  0.0  0.0  |
            | 0.0  0.7  0.3  0.0  0.0  0.0  |
            | 0.0  0.0  0.7  0.3  0.0  0.0  |
            | 0.3  0.0  0.0  0.7  0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            | 0.25 0.25 0.25 0.25 0.0  0.0  |
            -                               -

    """
    assert closeset_noise_ratio > 0.0, 'noise rate must be greater than 0.0'
    assert 0.0 <= openset_noise_ratio < 1.0, 'the ratio of out-of-distribution class must be within [0.0, 1.0)'
    closeset_nb_classes = int(nb_classes * (1 - openset_noise_ratio))
    # openset_nb_classes = nb_classes - closeset_nb_classes
    if noise_type == 'symmetric':
        P = np.ones((nb_classes, nb_classes))
        P = (closeset_noise_ratio / (closeset_nb_classes - 1)) * P
        for i in range(closeset_nb_classes):
            P[i, i] = 1.0 - closeset_noise_ratio
        for i in range(closeset_nb_classes, nb_classes):
            P[i, :] = 1.0 / closeset_nb_classes
        for i in range(closeset_nb_classes, nb_classes):
            P[:, i] = 0.0
    elif noise_type == 'asymmetric':
        P = np.eye(nb_classes)
        P[0, 0], P[0, 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        for i in range(1, closeset_nb_classes - 1):
            P[i, i], P[i, i + 1] = 1.0 - closeset_noise_ratio, closeset_noise_ratio
        P[closeset_nb_classes - 1, closeset_nb_classes - 1] = 1.0 - closeset_noise_ratio
        P[closeset_nb_classes - 1, 0] = closeset_noise_ratio
        for i in range(closeset_nb_classes, nb_classes):
            P[i, :] = 1.0 / closeset_nb_classes
        for i in range(closeset_nb_classes, nb_classes):
            P[:, i] = 0.0
    else:
        raise AssertionError("noise type must be either symmetric or asymmetric")
    return P


def noisify(y_train, noise_transition_matrix, random_state=None):
    y_train_noisy = multiclass_noisify(y_train, P=noise_transition_matrix, random_state=random_state)
    actual_noise = (y_train_noisy != y_train).mean()
    assert actual_noise > 0.0
    return y_train_noisy, actual_noise


def noisify_dataset(nb_classes=10, train_labels=None, noise_type=None,
                    closeset_noise_ratio=0.0, openset_noise_ratio=0.0, random_state=0, verbose=True):
    noise_transition_matrix = generate_noise_matrix(noise_type, closeset_noise_ratio, openset_noise_ratio, nb_classes)
    train_noisy_labels, actual_noise_rate = noisify(train_labels, noise_transition_matrix, random_state)
    if verbose:
        print(f'Noise Transition Matrix: \n {noise_transition_matrix}')
        print(f'Noise Type: {noise_type} (close set: {closeset_noise_ratio}, open set: {openset_noise_ratio})\n'
              f'Actual Total Noise Ratio: {actual_noise_rate:.3f}')
    return train_noisy_labels, actual_noise_rate


class NoisyCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 noise_type='clean', closeset_ratio=0.0, openset_ratio=0.2, random_state=0, verbose=True):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)
        if not train:
            assert noise_type == 'clean', f'In test mode, noise_type should be clean, but got {noise_type}!'

        self.noise_type = noise_type
        self.nb_classes = max(self.targets) + 1
        self.closeset_noise_rate = closeset_ratio
        self.openset_noise_ratio = openset_ratio
        num_samples = len(self.data)
        if self.train and (noise_type != 'clean'):
            train_labels = np.asarray([[self.targets[i]] for i in range(len(self.targets))])
            noisy_labels, self.actual_noise_rate = noisify_dataset(self.nb_classes, train_labels, noise_type, closeset_ratio,
                                                                   openset_ratio, random_state, verbose)
            self.noisy_labels = [i[0] for i in noisy_labels]
            train_labels = [i[0] for i in train_labels]
            self.noise_or_not = np.transpose(self.noisy_labels) == np.transpose(train_labels)
        else:
            new_data = []
            new_targets = []
            for i in range(num_samples):
                label = self.targets[i]
                if label < int(self.nb_classes * (1 - openset_ratio)):
                    new_data.append(self.data[i])
                    new_targets.append(label)
            self.data = np.stack(new_data, axis=0)
            self.targets = new_targets
        self.num_samples = len(self.data)
        self.samples = self.data

    def __getitem__(self, index):
        if self.noise_type != 'clean':
            img, target = self.data[index], self.noisy_labels[index]
        else:
            img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'index': index, 'data': img, 'label': target}

    def __len__(self):
        return len(self.data)

    def get_sets(self):
        if self.noise_type == 'clean':
            return None, None, None
        closed_set, open_set, clean_set = [], [], []
        closeset_nb_classes = int(self.nb_classes * (1 - self.openset_noise_ratio))
        openset_label_list = [i for i in range(closeset_nb_classes, self.nb_classes)]

        for idx in range(self.data.shape[0]):
            if self.targets[idx] in openset_label_list:
                assert self.targets[idx] != self.noisy_labels[idx]
                open_set.append(idx)
            elif self.targets[idx] != self.noisy_labels[idx]:
                assert self.targets[idx] not in openset_label_list
                closed_set.append(idx)
            else:
                clean_set.append(idx)

        return closed_set, open_set, clean_set


class NoisyCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 noise_type='clean', closeset_ratio=0.0, openset_ratio=0.2, random_state=0, verbose=True):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)
        if not train:
            assert noise_type == 'clean', f'In test mode, noise_type should be clean, but got {noise_type}!'

        self.noise_type = noise_type
        self.nb_classes = max(self.targets) + 1
        self.closeset_noise_rate = closeset_ratio
        self.openset_noise_ratio = openset_ratio
        num_samples = len(self.data)
        if self.train and (noise_type != 'clean'):
            train_labels = np.asarray([[self.targets[i]] for i in range(len(self.targets))])
            noisy_labels, self.actual_noise_rate = noisify_dataset(self.nb_classes, train_labels, noise_type, closeset_ratio,
                                                                   openset_ratio, random_state, verbose)
            self.noisy_labels = [i[0] for i in noisy_labels]
            train_labels = [i[0] for i in train_labels]
            self.noise_or_not = np.transpose(self.noisy_labels) == np.transpose(train_labels)
        else:
            new_data = []
            new_targets = []
            for i in range(num_samples):
                label = self.targets[i]
                if label < int(self.nb_classes * (1 - openset_ratio)):
                    new_data.append(self.data[i])
                    new_targets.append(label)
            self.data = np.stack(new_data, axis=0)
            self.targets = new_targets
        self.num_samples = len(self.data)
        self.samples = self.data

    def __getitem__(self, index):
        if self.noise_type != 'clean':
            img, target = self.data[index], self.noisy_labels[index]
        else:
            img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'index': index, 'data': img, 'label': target}

    def __len__(self):
        return len(self.data)

    def get_sets(self):
        if self.noise_type == 'clean':
            return None, None, None
        closed_set, open_set, clean_set = [], [], []
        closeset_nb_classes = int(self.nb_classes * (1 - self.openset_noise_ratio))
        openset_label_list = [i for i in range(closeset_nb_classes, self.nb_classes)]

        for idx in range(self.data.shape[0]):
            if self.targets[idx] in openset_label_list:
                assert self.targets[idx] != self.noisy_labels[idx]
                open_set.append(idx)
            elif self.targets[idx] != self.noisy_labels[idx]:
                assert self.targets[idx] not in openset_label_list
                closed_set.append(idx)
            else:
                clean_set.append(idx)

        return closed_set, open_set, clean_set
