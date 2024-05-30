# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
import os
from itertools import permutations, product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from utils import TemporalData, calculate_rotation_matrix


class UAVDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        local_radius: float = 50,
    ) -> None:
        self._split = split
        self._local_radius = local_radius
        if split == "sample":
            self._directory = "forecasting_sample"
        elif split == "train":
            self._directory = "train"
        elif split == "val":
            self._directory = "val"
        elif split == "test":
            self._directory = "test_obs"
        else:
            raise ValueError(split + " is not valid")
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [
            os.path.splitext(f)[0] + ".pt" for f in self.raw_file_names
        ]
        self._processed_paths = [
            os.path.join(self.processed_dir, f) for f in self._processed_file_names
        ]
        super(UAVDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, "data")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, "processed")

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        for raw_path in tqdm(self.raw_paths):
            kwargs = uav_data(self._split, raw_path)
            if kwargs == {}:
                continue
            data = TemporalData(**kwargs)
            torch.save(
                data, os.path.join(self.processed_dir, str(kwargs["seq_id"]) + ".pt")
            )

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def uav_data(split: str, raw_path: str) -> Dict:
    df = pd.read_csv(raw_path)

    timestamps = list(np.sort(df["TIMESTAMP"].unique()))
    historical_timestamps = timestamps[:20]
    historical_df = df[df["TIMESTAMP"].isin(historical_timestamps)]
    actor_ids = list(historical_df["TRACK_ID"].unique())
    df = df[df["TRACK_ID"].isin(actor_ids)]
    num_nodes = len(actor_ids)

    av_df = df[df["OBJECT_TYPE"] == "AV"].iloc
    av_index = actor_ids.index(av_df[0]["TRACK_ID"])
    agent_df = df[df["OBJECT_TYPE"] == "AGENT"].iloc
    agent_index = actor_ids.index(agent_df[0]["TRACK_ID"])

    # Making scene centered here, now inlcuding Z indx
    origin = torch.tensor(
        [av_df[19]["X"], av_df[19]["Y"], av_df[19]["Z"]], dtype=torch.float
    )
    # Calculate the heading vector, now in 3d
    av_heading_vector = origin - torch.tensor(
        [av_df[18]["X"], av_df[18]["Y"], av_df[18]["Z"]], dtype=torch.float
    )

    try:
        rotate_mat = calculate_rotation_matrix(av_heading_vector)
    except:
        return {}

    # initialization
    x = torch.zeros(num_nodes, 50, 3, dtype=torch.float)
    edge_index = (
        torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    )
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, 3, dtype=torch.float)

    for actor_id, actor_df in df.groupby("TRACK_ID"):
        node_idx = actor_ids.index(actor_id)
        node_steps = [
            timestamps.index(timestamp) for timestamp in actor_df["TIMESTAMP"]
        ]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[
            node_idx, 19
        ]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True
        # Now working with xyz data
        xyz = torch.from_numpy(
            np.stack(
                [actor_df["X"].values, actor_df["Y"].values, actor_df["Z"].values],
                axis=-1,
            )
        ).float()
        x[node_idx, node_steps] = torch.matmul(xyz - origin, rotate_mat)
        node_historical_steps = list(
            filter(lambda node_step: node_step < 20, node_steps)
        )
        if (
            len(node_historical_steps) > 1
        ):  # calculate the heading of the actor (approximately)
            heading_vector = (
                x[node_idx, node_historical_steps[-1]]
                - x[node_idx, node_historical_steps[-2]]
            )
            rotate_angles[node_idx][0] = np.arctan2(
                heading_vector[2], heading_vector[1]
            )
            rotate_angles[node_idx][1] = np.arctan2(
                heading_vector[2], heading_vector[0]
            )
            rotate_angles[node_idx][2] = np.arctan2(
                heading_vector[1], heading_vector[0]
            )
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True

    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1:20] = padding_mask[:, :19] & ~padding_mask[:, 1:20]

    positions = x.clone()
    x[:, 20:] = torch.where(
        (padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
        torch.zeros(num_nodes, 30, 3),
        x[:, 20:] - x[:, 19].unsqueeze(-2),
    )
    x[:, 1:20] = torch.where(
        (padding_mask[:, :19] | padding_mask[:, 1:20]).unsqueeze(-1),
        torch.zeros(num_nodes, 19, 3),
        x[:, 1:20] - x[:, :19],
    )
    x[:, 0] = torch.zeros(num_nodes, 3)

    # get lane features at the current time step
    y = None if split == "test" else x[:, 20:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]

    return {
        "x": x[:, :20],  # [N, 20, 3]
        "positions": positions,  # [N, 50, 3]
        "edge_index": edge_index,  # [2, N x N - 1]
        "y": y,  # [N, 30, 3]
        "num_nodes": num_nodes,
        "padding_mask": padding_mask,  # [N, 50]
        "bos_mask": bos_mask,  # [N, 20]
        "rotate_angles": rotate_angles,  # [N, 3], Store theta and phi for all N
        "seq_id": int(seq_id),
        "av_index": av_index,
        "agent_index": agent_index,
        "origin": origin.unsqueeze(0),
    }
