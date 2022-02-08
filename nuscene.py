import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py
import faiss


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_whole_training_set(opt, onlyDB=False, forCluster=False):
    return WholeDatasetFromStruct(opt, join(opt.structDir, 'nuscenes_train.mat'), opt.imgDir, input_transform=input_transform(), onlyDB=onlyDB, forCluster=forCluster)


def get_whole_val_set(opt):
    return WholeDatasetFromStruct(opt, join(opt.structDir, 'nuscenes_val.mat'), opt.imgDir, input_transform=input_transform())


def get_whole_test_set(opt):
    return WholeDatasetFromStruct(opt, join(opt.structDir, 'nuscenes_test.mat'), opt.imgDir, input_transform=input_transform())


def get_training_query_set(opt, margin=0.1):
    return QueryDatasetFromStruct(opt, join(opt.structDir, 'nuscenes_train.mat'), opt.imgDir, input_transform=input_transform(), margin=margin)


def get_val_query_set(opt, margin=0.1):
    return QueryDatasetFromStruct(opt, join(opt.structDir, 'nuscenes_val.mat'), opt.imgDir, input_transform=input_transform(), margin=margin)


dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ', 'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    dataset = 'nuscenes'

    whichSet = matStruct[0].item()

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    # dbImage = [f[0].item() for f in matStruct[1]]
    dbImage = matStruct[1]
    utmDb = matStruct[2].T

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    # qImage = [f[0].item() for f in matStruct[3]]
    qImage = matStruct[3]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr, posDistSqThr, nonTrivPosDistSqThr)


class WholeDatasetFromStructForCluster(data.Dataset):
    def __init__(self, opt, structFile, img_dir, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)

        self.images = [join(img_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(img_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5)    # TODO: sort!!

        return self.positives


class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, opt, structFile, img_dir, input_transform=None, onlyDB=False, forCluster=False):
        super().__init__()
        self.opt = opt
        self.forCluster = forCluster

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)

        self.images = [join(img_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(img_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def load_images(self, index):
        filename = self.images[index]
        frame_index = int(filename[-9:-4])
        if self.opt.seqLen == 1:
            edge_indices = [0]
        elif self.opt.seqLen == 2:
            edge_indices = [-1, 0]
        elif self.opt.seqLen == 3:
            edge_indices = [-2, -1, 0]
        elif self.opt.seqLen == 4:
            edge_indices = [-3, -2, -1, 0]
        elif self.opt.seqLen == 5:
            edge_indices = [-4, -3, -2, -1, 0]
        imgs = []
        for offset in edge_indices:
            img = Image.open(filename[:-9] + '{:0>5d}.jpg'.format(int(frame_index + offset)))
            if self.input_transform:
                img = self.input_transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, 0)

        return imgs, index

    def __getitem__(self, index):
        if self.forCluster:
            img = Image.open(self.images[index])
            if self.input_transform:
                img = self.input_transform(img)

            return img, index
        else:
            imgs, index = self.load_images(index)
            return imgs, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5)    # TODO: sort!!

        return self.positives


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)      # ([8, 3, 200, 200]) = [(3, 200, 200), (3, 200, 200), ..  ]     ([8, 1, 3, 200, 200])
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)     # ([80, 3, 200, 200]) ([80, 1, 3, 200, 200])
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices


class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, opt, structFile, img_dir, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()
        self.opt = opt
        self.img_dir = img_dir
        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample  # number of negatives to randomly sample
        self.nNeg = nNeg  # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5, return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.posDistThr, return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True))

        self.cache = None  # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0, )) for _ in range(self.dbStruct.numQ)]

    def load_images(self, filename):
        # filename = self.images[index]
        frame_index = int(filename[-9:-4])
        if self.opt.seqLen == 1:
            edge_indices = [0]
        elif self.opt.seqLen == 2:
            edge_indices = [-1, 0]
        elif self.opt.seqLen == 3:
            edge_indices = [-2, -1, 0]
        elif self.opt.seqLen == 4:
            edge_indices = [-3, -2, -1, 0]
        elif self.opt.seqLen == 5:
            edge_indices = [-4, -3, -2, -1, 0]
        imgs = []
        for offset in edge_indices:
            img = Image.open(filename[:-9] + '{:0>5d}.jpg'.format(int(frame_index + offset)))
            if self.input_transform:
                img = self.input_transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, 0)

        return imgs

    def __getitem__(self, index):

        index = self.queries[index]  # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")
            qOffset = self.dbStruct.numDb

            qFeat = h5feat[index + qOffset]
            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            qFeat = torch.tensor(qFeat)
            posFeat = torch.tensor(posFeat)
            dist = torch.norm(qFeat - posFeat, dim=1, p=None)
            result = dist.topk(1, largest=False)
            dPos, posNN = result.values, result.indices
            posIndex = self.nontrivial_positives[index][posNN].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)  # randomly choose potential_negatives
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))        # remember negSamples history for each query

            negFeat = h5feat[negSample.tolist()]
            negFeat = torch.tensor(negFeat)
            dist = torch.norm(qFeat - negFeat, dim=1, p=None)
            result = dist.topk(self.nNeg * 10, largest=False)
            dNeg, negNN = result.values, result.indices

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg.numpy() < dPos.numpy() + self.margin**0.5

            if np.sum(violatingNeg) < 1:
                # if none are violating then skip this query
                return None

            negNN = negNN.numpy()
            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = self.load_images(join(self.img_dir, self.dbStruct.qImage[index]))
        positive = self.load_images(join(self.img_dir, self.dbStruct.dbImage[posIndex]))

        negatives = []
        for negIndex in negIndices:
            negative = self.load_images(join(self.img_dir, self.dbStruct.dbImage[negIndex]))
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)               # ([10, 3, 200, 200])
        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

    def __len__(self):
        return len(self.queries)
