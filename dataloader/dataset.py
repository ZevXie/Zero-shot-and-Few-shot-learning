import torch
from torch.utils.data.dataset import Dataset
import scipy.io
import numpy as np
from config import opt


class dataloader(Dataset):
    """docstring for CUB"""

    def __init__(self, param, transform,  split='train'):
        device = param.device
        path_features = './datasets/' + param.dataset_name + '/res101.mat'
        path_att_splits = './datasets/' + param.dataset_name + '/att_splits.mat'
        self.res101 = scipy.io.loadmat(path_features)
        att_splits = scipy.io.loadmat(path_att_splits)

        self.scaler = transform

        self.labels, self.feats, self.sig = self.get_data(att_splits, split)
        assert len(self.labels) == len(self.feats) == len(self.sig)
        if len(self.feats) == 0:
            raise (RuntimeError("Found zero feats in the directory: " + path_features))

        self.feats_ = torch.from_numpy(self.feats).float().to(device)
        self.labels_ = torch.from_numpy(self.labels).long().to(device)
        self.sig_ = torch.from_numpy(self.sig).float().to(device)

    def __getitem__(self, index):
        # index = np.random.randint(1,50)
        x = self.feats_[index, :]
        sig = self.sig_[index, :]
        y = self.labels_[index]
        return x, y, sig

    def __get_perclass_feats__(self, index):
        if index in torch.unique(self.labels_):
            idx = np.where(self.labels_.cpu().numpy() == index.cpu().numpy())
            return self.feats_[idx[0], :]

    def __NumClasses__(self):
        return torch.unique(self.labels_)

    def __get_attlen__(self):
        len_sig = self.sig.shape[1]
        return len_sig

    def __getlen__(self):
        len_feats = self.feats.shape[1]
        return len_feats

    def __totalClasses__(self):
        return len(np.unique(self.res101['labels']).tolist())

    def __attributeVector__(self):
        return self.signature[:, np.unique(self.labels_.cpu().numpy()) - 1].transpose(), np.unique(self.labels_.cpu().numpy())

    def __Test_Features_Labels__(self):
        return self.feats_.to(opt.device), self.labels_.to(opt.device)

    def check_unique_labels(self, labels, att_splits):
        trainval_loc = 'trainval_loc'
        train_loc = 'train_loc'
        val_loc = 'val_loc'
        test_loc = 'test_unseen_loc'

        self.labels_train = labels[np.squeeze(att_splits[train_loc] - 1)]
        self.labels_val = labels[np.squeeze(att_splits[val_loc] - 1)]
        self.labels_trainval = labels[np.squeeze(att_splits[trainval_loc] - 1)]
        self.labels_test = labels[np.squeeze(att_splits[test_loc] - 1)]

        self.train_labels_seen = np.unique(self.labels_train)
        self.val_labels_unseen = np.unique(self.labels_val)
        self.trainval_labels_seen = np.unique(self.labels_trainval)
        self.test_labels_unseen = np.unique(self.labels_test)

    def __len__(self):
        return self.feats.shape[0]

    def get_data(self, att_splits, split):
        labels = self.res101['labels']
        X_features = self.res101['features']
        self.signature = att_splits['att']  # 85*50

        self.check_unique_labels(labels, att_splits)
        if split == 'trainval':
            loc = 'trainval_loc'  # 23527  40类
        elif split == 'train':
            loc = 'train_loc'  # 20218
        elif split == 'val':
            loc = 'val_loc'  # 9191
        elif split == 'test_seen':
            loc = 'test_seen_loc'  # 5882
        else:
            loc = 'test_unseen_loc'  # 7913

        # aaa = np.squeeze(att_splits[loc] - 1)
        labels_loc = labels[np.squeeze(att_splits[loc] - 1)]  # labels：1-150 ndarray  labels的索引从0开始
        feat_vec = np.transpose(X_features[:, np.squeeze(att_splits[loc] - 1)])

        unique_labels = np.unique(labels_loc)
        sig_vec = np.zeros((labels_loc.shape[0], self.signature.shape[0]))  # 11727*312
        labels_list = np.squeeze(labels_loc).tolist()
        for i, idx in enumerate(labels_list):
            sig_vec[i, :] = self.signature[:, idx - 1]  # 属性大矩阵  7057*312

        self.scaler.fit_transform(feat_vec)

        labels_loc_ = np.int64(labels_loc)

        return labels_loc_, feat_vec, sig_vec


class classifier_dataloader(Dataset):
    """docstring for classifier_dataloader"""

    def __init__(self, features_img, labels, device):
        self.labels = labels.long().to(device)
        self.feats = features_img.float().to(device)

    def __getitem__(self, index):
        X = self.feats[index, :]
        y = self.labels[index] - 1  # for NLLL loss
        return X, y

    def __len__(self):
        return len(self.labels)

    def __targetClasses__(self):
        return torch.unique(self.labels)
