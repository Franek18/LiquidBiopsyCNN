import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

class CancerDataset(Dataset):
    """Cancer Classification dataset.
        As our dataset we use samples stored as a 2D matrices in .txt file
        instead of images.
        The size of every raw sample is [267, 531].
        The size of vectorized sample is [24089,].
        The size of preprocessed matrix sample is [180, 134].
    """

    def __init__(self, data_file, labels_file, group=None, signals_permutation=None, columns_permutation=None, transform=None, target_transform=None):
        '''
            @param annotations_file: file with two columns: Path to the sample in
            directory with samples and Class of this sample.
            @param img_dir: directory in which samples from both classes: cancer
            and non-cancer are stored.
            @param group: parameter used when we want to select from a sample
            rows/signal paths only from a given group.
            @param signals_permutation: permutation of rows in every sample if
            we want to train on permutated rows.
            @param columns_permutation: permutation of columns in every sample if
            we want to train on permutated columns.
            @param transofrm: transoformation of a sample i.e. mean and standard deviation.
            @param target_transform: ransoformation of a label of a sample.
        '''


        self.data = torch.load(data_file)
        self.labels = np.load(labels_file)
        self.group = group
        self.signals_permutation = signals_permutation
        self.columns_permutation = columns_permutation
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        '''
            Return number of samples in a dataset.
        '''
        return len(self.labels)

    def __getitem__(self, idx):
        '''
            Return a sample from the given index.
        '''
        # Pytorch requires data in float32 not float64 (numpy) format
        sample = np.float32(self.data[idx])
        # if self.group is not None:
        #     sample = sample[self.group[0]:self.group[-1] + 1]
        #
        # if self.signals_permutation is not None:
        #     # randomly permutate signal's pathways in a sample
        #     image = image[self.signals_permutation]
        #
        # if self.columns_permutation is not None:
        #     image = image[:, self.columns_permutation]

        label = self.labels[idx]

        # if self.transform:
        #     image = self.transform(image.float())
        #
        # if self.target_transform:
        #     label = self.target_transform(label)

        return sample, label

    def getClassWeights(self):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.labels), y=self.labels)

        return class_weights


def save_data_as_array():
    data_dir = "data2/KEGG_Pathway_Image"
    annotation_file = "annotations/Cancer_annotations_mts2.csv"

    data = []
    labels = []
    i = 0

    with open(annotation_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            sample_path = row['Path']


            y = int(row['Class'])
            # split = row['Split']
            # if split == "Train":
            #     train_indices.append(i)
            # elif split == "Test":
            #     test_indices.append(i)

            sample = np.loadtxt(os.path.join(data_dir, sample_path))
            data.append(np.reshape(sample, (1, sample.shape[0], sample.shape[1])))
            labels.append(y)
            i += 1

    data = torch.tensor(np.array(data))
    torch.save(data, "biopsy_data2.pt")
    np.save("biopsy_labels.npy", labels)
    # np.save(train_indices_file, train_indices)
    # np.save(test_indices_file, test_indices)


def save_data_reduced():
    # Saving only data with nonzero standard deviation across dataset
    data = torch.load("biopsy/biopsy_data2.pt").numpy()

    data_nonzero_vectors = []
    data_nonzero_matrices = []
    data_std = np.std(data[:, 0], axis=0)

    # Get indices of pixels with nonzero std across all samples
    nonzero_std_idx = np.nonzero(data_std)

    for sample_matrix in data:
        # A vectorized form of a sample - 24089 pixels
        nonzero_std_vector = sample_matrix[0, nonzero_std_idx[0], nonzero_std_idx[1]]

        # A matrix form of a sample - 180x134, 24089 active and 31 nonactive pixels
        nonzero_vectors_padded = np.pad(nonzero_std_vector, (0, 31))
        new_sample_matrix = np.reshape(nonzero_vectors_padded, (180, 134))

        data_nonzero_vectors.append(np.expand_dims(nonzero_std_vector, axis=0))
        data_nonzero_matrices.append(np.expand_dims(new_sample_matrix, axis=0))


    torch.save(torch.tensor(np.array(data_nonzero_vectors)), "biopsy/biopsy_data_vectors.pt")
    torch.save(torch.tensor(np.array(data_nonzero_matrices)), "biopsy/new_biopsy_data_matrices.pt")
