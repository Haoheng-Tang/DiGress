import json
import os
import pathlib
import re

import torch
from torch_geometric.data import Data, InMemoryDataset

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class InpatientGraphDataset(InMemoryDataset):
    FILE_PATTERNS = (
        r"^penn_inpatient_pavilion_subgraph_(\d+)\.json$",
        r"^subgraph_(\d+)\.json$",
    )
    TRAIN_SPLIT_RATIO = 0.8
    VAL_SPLIT_RATIO = 0.1
    SPLIT_SEED = 0

    def __init__(
        self,
        dataset_name,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        train_split_ratio=None,
        val_split_ratio=None,
        split_seed=None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.train_split_ratio = (
            self.TRAIN_SPLIT_RATIO if train_split_ratio is None else float(train_split_ratio)
        )
        self.val_split_ratio = (
            self.VAL_SPLIT_RATIO if val_split_ratio is None else float(val_split_ratio)
        )
        self.split_seed = self.SPLIT_SEED if split_seed is None else int(split_seed)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    @classmethod
    def _extract_file_index(cls, filename):
        for pattern in cls.FILE_PATTERNS:
            match = re.match(pattern, filename)
            if match:
                return int(match.group(1))

        # Generic fallback: accept any JSON file that ends with a numeric suffix.
        generic_match = re.search(r"(\d+)(?=\.json$)", filename)
        if filename.endswith(".json") and generic_match:
            return int(generic_match.group(1))

        return None

    def _compute_split_sizes(self, num_graphs):
        if num_graphs < 3:
            raise RuntimeError(
                f"Need at least 3 inpatient raw graphs for train/val/test split, found {num_graphs}."
            )

        train_size = max(1, int(num_graphs * self.train_split_ratio))
        val_size = max(1, int(num_graphs * self.val_split_ratio))

        # Keep at least one test sample.
        max_train_val = num_graphs - 1
        overflow = train_size + val_size - max_train_val
        if overflow > 0:
            reducible_train = max(0, train_size - 1)
            reduce_train = min(overflow, reducible_train)
            train_size -= reduce_train
            overflow -= reduce_train

        if overflow > 0:
            reducible_val = max(0, val_size - 1)
            reduce_val = min(overflow, reducible_val)
            val_size -= reduce_val
            overflow -= reduce_val

        test_size = num_graphs - train_size - val_size
        if test_size < 1:
            raise RuntimeError(
                f"Invalid inpatient split sizes for {num_graphs} graphs: "
                f"train={train_size}, val={val_size}, test={test_size}."
            )

        return train_size, val_size, test_size

    def download(self):
        json_paths = []
        for filename in os.listdir(self.raw_dir):
            index = self._extract_file_index(filename)
            if index is not None:
                json_paths.append((index, os.path.join(self.raw_dir, filename)))
        json_paths.sort(key=lambda x: x[0])

        if len(json_paths) < 3:
            raise RuntimeError(
                f"Expected at least 3 inpatient raw JSON files in {self.raw_dir}, "
                f"found {len(json_paths)}."
            )

        all_types = set()
        parsed_graphs = []
        for _, path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                graph_json = json.load(f)
            nodes = graph_json.get("nodes", [])
            links = graph_json.get("links", [])

            all_types.update(node.get("name", "unknown") for node in nodes)
            parsed_graphs.append({"nodes": nodes, "links": links})

        node_type_names = sorted(all_types)
        num_graphs = len(parsed_graphs)
        train_size, val_size, test_size = self._compute_split_sizes(num_graphs)

        generator = torch.Generator()
        generator.manual_seed(self.split_seed)
        shuffled_indices = torch.randperm(num_graphs, generator=generator).tolist()

        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]

        split_payloads = {
            "train": {"node_type_names": node_type_names, "graphs": [parsed_graphs[i] for i in train_indices]},
            "val": {"node_type_names": node_type_names, "graphs": [parsed_graphs[i] for i in val_indices]},
            "test": {"node_type_names": node_type_names, "graphs": [parsed_graphs[i] for i in test_indices]},
        }
        print(
            f"Inpatient split sizes: train={train_size}, val={val_size}, test={test_size} "
            f"(total={num_graphs}, seed={self.split_seed})"
        )

        torch.save(split_payloads["train"], self.raw_paths[0])
        torch.save(split_payloads["val"], self.raw_paths[1])
        torch.save(split_payloads["test"], self.raw_paths[2])

    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])
        node_type_names = raw_dataset["node_type_names"]
        node_type_to_idx = {name: idx for idx, name in enumerate(node_type_names)}
        num_types = len(node_type_names)

        data_list = []
        for graph_dict in raw_dataset["graphs"]:
            nodes = graph_dict["nodes"]
            links = graph_dict["links"]
            n = len(nodes)
            if n == 0:
                continue

            node_id_to_idx = {str(node["id"]): idx for idx, node in enumerate(nodes)}
            # Predict node's feature: 'name'
            node_type_ids = torch.tensor(
                [node_type_to_idx.get(node.get("name", "unknown"), 0) for node in nodes],
                dtype=torch.long
            )
            X = torch.nn.functional.one_hot(node_type_ids, num_classes=num_types).float()

            edges = []
            for edge in links:
                src = node_id_to_idx.get(str(edge.get("source")))
                dst = node_id_to_idx.get(str(edge.get("target")))
                if src is None or dst is None:
                    continue
                edges.append([src, dst])
                edges.append([dst, src])

            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            y = torch.zeros([1, 0]).float()
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            if edge_attr.shape[0] > 0:
                edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])



class InpatientGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        split_kwargs = {
            "train_split_ratio": getattr(cfg.dataset, "train_split_ratio", InpatientGraphDataset.TRAIN_SPLIT_RATIO),
            "val_split_ratio": getattr(cfg.dataset, "val_split_ratio", InpatientGraphDataset.VAL_SPLIT_RATIO),
            "split_seed": getattr(cfg.dataset, "split_seed", InpatientGraphDataset.SPLIT_SEED),
        }
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': InpatientGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path, **split_kwargs),
                    'val': InpatientGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path, **split_kwargs),
                    'test': InpatientGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path, **split_kwargs)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class InpatientDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        # Node features are one-hot encoded from room/category names in raw JSON files.
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

