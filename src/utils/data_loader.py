import os.path as osp

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected, add_self_loops, degree
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array

import random
import numpy as np
import scipy.sparse as sp
import pickle
import json5
"""
	Functions to help load the graph data
"""

from tqdm import tqdm
import pickle
import json
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
import torch

def knn_impute_gpu_batched(features, k=5, batch_size=512):
    """
    GPU-based KNN imputation for the last column of `features` with batching.

    Args:
        features (torch.Tensor): [num_nodes, num_features] with NaN for missing values
        k (int): number of nearest neighbors
        batch_size (int): number of missing rows to process at a time

    Returns:
        features (torch.Tensor): same tensor with NaNs in last column imputed
    """
    device = features.device
    num_nodes, num_features = features.shape

    # Identify missing and known rows
    bot_scores = features[:, -1]
    missing_mask = torch.isnan(bot_scores)
    known_mask = ~missing_mask

    if missing_mask.sum() == 0:
        # Nothing to impute
        return features

    known_idx = known_mask.nonzero(as_tuple=True)[0]
    missing_idx = missing_mask.nonzero(as_tuple=True)[0]

    # Feature vectors without bot_score
    known_features = features[known_idx, :-1]  # [num_known, n_features-1]
    missing_features = features[missing_idx, :-1]  # [num_missing, n_features-1]

    # Batch processing to save memory
    for start in tqdm(range(0, len(missing_idx), batch_size), desc="GPU KNN imputation"):
        end = min(start + batch_size, len(missing_idx))
        batch_missing = missing_features[start:end]  # [batch_size, n_features-1]

        # Compute distances to all known nodes
        dists = torch.cdist(batch_missing, known_features, p=2)

        # Get top-k nearest neighbors
        _, nn_idx = torch.topk(dists, k, largest=False, dim=1)

        # Gather neighbor bot scores and compute mean
        neighbor_scores = bot_scores[known_idx][nn_idx]  # [batch_size, k]
        imputed_scores = neighbor_scores.mean(dim=1)

        # Fill missing values
        features[missing_idx[start:end], -1] = imputed_scores

        # Optional: free GPU memory for this batch
        del dists, nn_idx, neighbor_scores, imputed_scores
        torch.cuda.empty_cache()

    return features

def add_bot_score_feature(dataset, mapping_path, botscore_path, k=5):
    """
    Add bot score as a node feature and KNN-impute missing values, with progress bar.

    Parameters:
        dataset: FNNDataset object
        mapping_path: path to gos_id_twitter_mapping.pkl
        botscore_path: path to bot_score.json (with top-level "data" list)
        k: number of neighbors for KNN imputation
    """
    # Load mapping: node index -> Twitter user_id
    with open(mapping_path, 'rb') as f:
        idx_to_userid = pickle.load(f)  # dict: node_idx -> twitter_user_id (str)

    # Load bot scores JSON (with top-level "data")
    with open(botscore_path, 'r') as f:
        data = json.load(f)
    data_list = data["data"]

    # Build a mapping from user_id to bot_score
    bot_scores_dict = {str(entry['user_id']): entry['bot_score'] for entry in data_list}

    num_nodes = dataset.data.x.size(0)
    bot_scores = torch.full((num_nodes, 1), float('nan'), dtype=torch.float)

    # Fill known bot scores with a progress bar
    for idx in tqdm(range(num_nodes), desc="Assigning bot scores"):
        user_id = idx_to_userid.get(idx)
        if user_id is not None:
            score = bot_scores_dict.get(str(user_id))
            if score is not None:
                bot_scores[idx] = float(score)

    # Concatenate bot scores and move to GPU
    features = torch.cat([dataset.data.x, bot_scores], dim=1).cuda()
    
    # GPU KNN imputation with batching
    features_imputed = knn_impute_gpu_batched(features, k=5, batch_size=512)
    
    # Move back to CPU and update dataset
    dataset.data.x = features_imputed.cpu().float()

def preprocess_bot_score_edges(dataset, bot_score_idx=-1, diff_threshold=0.4):
    """
    Preprocess a dataset by removing edges whose endpoints differ too much
    in bot-score, and removing orphan nodes afterward.

    Args:
        dataset: FNNDataset or similar (list-like of Data objects)
        bot_score_idx: index of bot-score feature in data.x (default: last column)
        diff_threshold: max allowed absolute difference between bot scores
                        for an edge to be kept.

    Returns:
        A list of cleaned Data objects (same order as dataset).
    """

    processed_graphs = []

    for data in dataset:
        x = data.x
        edge_index = data.edge_index

        # -----------------------------------------
        # Step 1: Compute bot score per node
        # -----------------------------------------
        bot_score = x[:, bot_score_idx]  # shape [num_nodes]

        # -----------------------------------------
        # Step 2: Keep only edges with small bot-score difference
        # -----------------------------------------
        src, dst = edge_index

        diff = (bot_score[src] - bot_score[dst]).abs()
        keep_mask = diff <= diff_threshold

        # Apply mask
        new_edge_index = edge_index[:, keep_mask]

        # -----------------------------------------
        # Step 3: Remove orphan nodes
        # -----------------------------------------
        # compute degree
        deg = degree(new_edge_index[0], x.size(0)) + degree(new_edge_index[1], x.size(0))
        keep_nodes = deg > 0

        # map old -> new node indices
        new_index = torch.nonzero(keep_nodes, as_tuple=False).view(-1)
        # create a mapping array
        old_to_new = -torch.ones(x.size(0), dtype=torch.long)
        old_to_new[new_index] = torch.arange(new_index.size(0))

        # -----------------------------------------
        # Step 4: Reindex nodes and edges
        # -----------------------------------------
        x_clean = x[new_index]

        if new_edge_index.numel() > 0:
            src_new = old_to_new[new_edge_index[0]]
            dst_new = old_to_new[new_edge_index[1]]
            edge_index_clean = torch.stack([src_new, dst_new], dim=0)
        else:
            edge_index_clean = torch.empty((2, 0), dtype=torch.long)

        # -----------------------------------------
        # Step 5: Copy labels and other attributes
        # -----------------------------------------
        data_clean = Data(
            x=x_clean,
            edge_index=edge_index_clean,
            y=data.y
        )

        # preserve batch-level attributes if they exist
        for attr in ["graph_id", "num_nodes", "num_edges"]:
            if hasattr(data, attr):
                setattr(data_clean, attr, getattr(data, attr))

        processed_graphs.append(data_clean)

    return processed_graphs



def read_file(folder, name, dtype=None):
	path = osp.join(folder, '{}.txt'.format(name))
	return read_txt_array(path, sep=',', dtype=dtype)


def split(data, batch):
	"""
	PyG util code to create graph batches
	"""

	node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
	node_slice = torch.cat([torch.tensor([0]), node_slice])

	row, _ = data.edge_index
	edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
	edge_slice = torch.cat([torch.tensor([0]), edge_slice])

	# Edge indices should start at zero for every graph.
	data.edge_index -= node_slice[batch[row]].unsqueeze(0)
	data.__num_nodes__ = torch.bincount(batch).tolist()

	slices = {'edge_index': edge_slice}
	if data.x is not None:
		slices['x'] = node_slice
	if data.edge_attr is not None:
		slices['edge_attr'] = edge_slice
	if data.y is not None:
		if data.y.size(0) == batch.size(0):
			slices['y'] = node_slice
		else:
			slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

	return data, slices


def read_graph_data(folder, feature):
	"""
	PyG util code to create PyG data instance from raw graph data
	"""

	node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')
	edge_index = read_file(folder, 'A', torch.long).t()
	node_graph_id = np.load(folder + 'node_graph_id.npy')
	graph_labels = np.load(folder + 'graph_labels.npy')


	edge_attr = None
	x = torch.from_numpy(node_attributes.todense()).to(torch.float)
	node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
	y = torch.from_numpy(graph_labels).to(torch.long)
	_, y = y.unique(sorted=True, return_inverse=True)

	num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
	edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
	edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

	data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
	data, slices = split(data, node_graph_id)

	return data, slices


class ToUndirected:
	def __init__(self):
		"""
		PyG util code to transform the graph to the undirected graph
		"""
		pass

	def __call__(self, data):
		edge_attr = None
		edge_index = to_undirected(data.edge_index, data.x.size(0))
		num_nodes = edge_index.max().item() + 1 if data.x is None else data.x.size(0)
		# edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
		edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
		data.edge_index = edge_index
		data.edge_attr = edge_attr
		return data


class DropEdge:
	def __init__(self, tddroprate, budroprate):
		"""
		Drop edge operation from BiGCN (Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks)
		1) Generate TD and BU edge indices
		2) Drop out edges
		Code from https://github.com/TianBian95/BiGCN/blob/master/Process/dataset.py
		"""
		self.tddroprate = tddroprate
		self.budroprate = budroprate

	def __call__(self, data):
		edge_index = data.edge_index

		if self.tddroprate > 0:
			row = list(edge_index[0])
			col = list(edge_index[1])
			length = len(row)
			poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
			poslist = sorted(poslist)
			row = list(np.array(row)[poslist])
			col = list(np.array(col)[poslist])
			new_edgeindex = [row, col]
		else:
			new_edgeindex = edge_index

		burow = list(edge_index[1])
		bucol = list(edge_index[0])
		if self.budroprate > 0:
			length = len(burow)
			poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
			poslist = sorted(poslist)
			row = list(np.array(burow)[poslist])
			col = list(np.array(bucol)[poslist])
			bunew_edgeindex = [row, col]
		else:
			bunew_edgeindex = [burow, bucol]

		data.edge_index = torch.LongTensor(new_edgeindex)
		data.BU_edge_index = torch.LongTensor(bunew_edgeindex)
		data.root = torch.FloatTensor(data.x[0])
		data.root_index = torch.LongTensor([0])

		return data


class FNNDataset(InMemoryDataset):
	r"""
		The Graph datasets built upon FakeNewsNet data

	Args:
		root (string): Root directory where the dataset should be saved.
		name (string): The `name
			<https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
			dataset.
		transform (callable, optional): A function/transform that takes in an
			:obj:`torch_geometric.data.Data` object and returns a transformed
			version. The data object will be transformed before every access.
			(default: :obj:`None`)
		pre_transform (callable, optional): A function/transform that takes in
			an :obj:`torch_geometric.data.Data` object and returns a
			transformed version. The data object will be transformed before
			being saved to disk. (default: :obj:`None`)
		pre_filter (callable, optional): A function that takes in an
			:obj:`torch_geometric.data.Data` object and returns a boolean
			value, indicating whether the data object should be included in the
			final dataset. (default: :obj:`None`)
	"""

	def __init__(self, root, name, feature='spacy', empty=False, transform=None, pre_transform=None, pre_filter=None):
		self.name = name
		self.root = root
		self.feature = feature
		super(FNNDataset, self).__init__(root, transform, pre_transform, pre_filter)
		if not empty:
			self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = torch.load(self.processed_paths[0])

	@property
	def raw_dir(self):
		name = ''
		return osp.join(self.root, self.name, name)

	@property
	def processed_dir(self):
		name = 'processed/'
		return osp.join(self.root, self.name, name)

	@property
	def num_node_attributes(self):
		if self.data.x is None:
			return 0
		return self.data.x.size(1)

	@property
	def raw_file_names(self):
		names = ['node_graph_id', 'graph_labels']
		return ['{}.npy'.format(name) for name in names]

	@property
	def processed_file_names(self):
		if self.pre_filter is None:
			return f'{self.name[:3]}_data_{self.feature}.pt'
		else:
			return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

	def process(self):

		self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

		if self.pre_filter is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [data for data in data_list if self.pre_filter(data)]
			self.data, self.slices = self.collate(data_list)

		if self.pre_transform is not None:
			data_list = [self.get(idx) for idx in range(len(self))]
			data_list = [self.pre_transform(data) for data in data_list]
			self.data, self.slices = self.collate(data_list)

		# The fixed data split for benchmarking evaluation
		# train-val-test split is 70%-20%-10%
		self.train_idx = torch.from_numpy(np.load(self.raw_dir + 'test_idx.npy')).to(torch.long)
		self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'train_idx.npy')).to(torch.long)
		self.test_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)

		torch.save((self.data, self.slices, self.train_idx, self.val_idx, self.test_idx), self.processed_paths[0])

	def __repr__(self):
		return '{}({})'.format(self.name, len(self))