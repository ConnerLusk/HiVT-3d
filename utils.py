# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data


class TemporalData(Data):
    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attrs: Optional[List[torch.Tensor]] = None,
        y: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        bos_mask: Optional[torch.Tensor] = None,
        rotate_angles: Optional[torch.Tensor] = None,
        lane_vectors: Optional[torch.Tensor] = None,
        is_intersections: Optional[torch.Tensor] = None,
        turn_directions: Optional[torch.Tensor] = None,
        traffic_controls: Optional[torch.Tensor] = None,
        lane_actor_index: Optional[torch.Tensor] = None,
        lane_actor_vectors: Optional[torch.Tensor] = None,
        seq_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(
            x=x,
            positions=positions,
            edge_index=edge_index,
            y=y,
            num_nodes=num_nodes,
            padding_mask=padding_mask,
            bos_mask=bos_mask,
            rotate_angles=rotate_angles,
            lane_vectors=lane_vectors,
            is_intersections=is_intersections,
            turn_directions=turn_directions,
            traffic_controls=traffic_controls,
            lane_actor_index=lane_actor_index,
            lane_actor_vectors=lane_actor_vectors,
            seq_id=seq_id,
            **kwargs,
        )
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f"edge_attr_{t}"] = edge_attrs[t]

    def __inc__(self, key, value):
        if key == "lane_actor_index":
            return torch.tensor([[self["lane_vectors"].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value)


class DistanceDropEdge(object):
    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif "weight_hr" in name:
                nn.init.xavier_uniform_(param)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)


def calculate_rotation_matrix(vector):
    """
    Calculates the 3D rotation matrix that aligns the z-axis with the given vector.

    Args:
        vector (torch.Tensor): A 3D vector of shape (3,).

    Returns:
        torch.Tensor: A 3x3 rotation matrix.
    """

    norm = torch.norm(vector)
    if norm == 0:
        raise ValueError("Input vector cannot be a zero vector.")
    unit_vector = vector / norm

    # Calculate rotation angle
    theta = torch.atan2(unit_vector[1], unit_vector[0])

    # Calculate rotation axis
    rot_axis = torch.cross(unit_vector, torch.tensor([0.0, 0.0, 1.0]))

    # Calculate rotation matrix using Rodrigues' rotation formula
    rot_matrix = (
        torch.eye(3)
        + torch.sin(theta)
        * torch.tensor(
            [
                [0, -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0, -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0],
            ]
        )
        + (1 - torch.cos(theta)) * torch.outer(rot_axis, rot_axis)
    )

    return rot_matrix


def rotation_matrix_from_angles(theta_x, theta_y, theta_z):
    """
    Calculate the 3D rotation matrix from the given rotation angles around the x, y, and z axes.

    Args:
        theta_x (float): Rotation angle around the x-axis in radians.
        theta_y (float): Rotation angle around the y-axis in radians.
        theta_z (float): Rotation angle around the z-axis in radians.

    Returns:
        torch.Tensor: A 3x3 rotation matrix.
    """
    cx = torch.cos(theta_x)
    sx = torch.sin(theta_x)
    cy = torch.cos(theta_y)
    sy = torch.sin(theta_y)
    cz = torch.cos(theta_z)
    sz = torch.sin(theta_z)

    rot_x = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    rot_y = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    rot_z = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    rot_matrix = rot_z @ rot_y @ rot_x

    return rot_matrix
