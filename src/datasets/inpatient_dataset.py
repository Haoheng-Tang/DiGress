import json
import os
import pathlib
import re

import torch
from torch_geometric.data import Data, InMemoryDataset

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class InpatientGraphDataset(InMemoryDataset):
    FILE_PATTERN = r"^penn_inpatient_pavilion_subgraph_(\d+)\.json$"
    EXPECTED_FILE_COUNT = 6

    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        json_paths = []
        for filename in os.listdir(self.raw_dir):
            match = re.match(self.FILE_PATTERN, filename)
            if match:
                json_paths.append((int(match.group(1)), os.path.join(self.raw_dir, filename)))
        json_paths.sort(key=lambda x: x[0])

        if len(json_paths) != self.EXPECTED_FILE_COUNT:
            raise RuntimeError(
                f"Expected {self.EXPECTED_FILE_COUNT} inpatient raw files in {self.raw_dir}, "
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

        split_payloads = {
            "train": {"node_type_names": node_type_names, "graphs": parsed_graphs[:2]},
            "val": {"node_type_names": node_type_names, "graphs": parsed_graphs[2:4]},
            "test": {"node_type_names": node_type_names, "graphs": parsed_graphs[4:6]},
        }

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
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': InpatientGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': InpatientGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': InpatientGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}
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
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

