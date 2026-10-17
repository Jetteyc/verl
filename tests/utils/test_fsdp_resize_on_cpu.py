# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import pytest

from verl.utils.fsdp_resize import FSDPResizePlanner, ResizeUnit, compute_1d_shard_range, order_units


@pytest.mark.parametrize(
    ("dim_size", "world_size", "expected"),
    [
        (12, 4, [(0, 3), (3, 3), (6, 3), (9, 3)]),
        (12, 6, [(0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2)]),
        (3, 5, [(0, 1), (1, 1), (2, 1), (3, 0), (3, 0)]),
    ],
)
def test_compute_1d_shard_range_matches_dtensor_chunking(dim_size, world_size, expected):
    assert [compute_1d_shard_range(dim_size, world_size, rank) for rank in range(world_size)] == expected


@pytest.mark.parametrize(
    ("dim_size", "world_size", "rank"),
    [(-1, 4, 0), (8, 0, 0), (8, 4, -1), (8, 4, 4)],
)
def test_compute_1d_shard_range_rejects_invalid_inputs(dim_size, world_size, rank):
    with pytest.raises(ValueError):
        compute_1d_shard_range(dim_size, world_size, rank)


def test_expand_4_to_6_plan_is_exact_and_non_redundant():
    planner = FSDPResizePlanner(12, old_ranks=[0, 1, 2, 3], new_ranks=[0, 1, 2, 3, 4, 5])

    assert planner.group_ranks == [0, 1, 2, 3, 4, 5]
    assert planner.send_row_splits(0) == [2, 1, 0, 0, 0, 0]
    assert planner.send_row_splits(1) == [0, 1, 2, 0, 0, 0]
    assert planner.send_row_splits(2) == [0, 0, 0, 2, 1, 0]
    assert planner.send_row_splits(3) == [0, 0, 0, 0, 1, 2]

    assert planner.recv_row_splits(0) == [2, 0, 0, 0, 0, 0]
    assert planner.recv_row_splits(1) == [1, 1, 0, 0, 0, 0]
    assert planner.recv_row_splits(4) == [0, 0, 1, 1, 0, 0]
    assert planner.recv_row_splits(5) == [0, 0, 0, 2, 0, 0]

    # rank 0 keeps everything it needs; new members need their whole shard
    assert planner.network_recv_rows(0) == 0
    assert planner.network_recv_rows(1) == 1
    assert planner.network_recv_rows(4) == 2
    assert sum(planner.network_recv_rows(r) for r in planner.new_ranks) == 9  # 3/4 of 12 rows


def test_shrink_6_to_2_plan_uses_the_union_group():
    planner = FSDPResizePlanner(12, old_ranks=[0, 1, 2, 3, 4, 5], new_ranks=[0, 1])

    assert planner.group_ranks == [0, 1, 2, 3, 4, 5]
    # old rank 2 holds rows [4, 6) which belong to new rank 0
    assert planner.send_row_splits(2) == [2, 0, 0, 0, 0, 0]
    # old rank 5 holds rows [10, 12) which belong to new rank 1
    assert planner.send_row_splits(5) == [0, 2, 0, 0, 0, 0]
    # new rank 1 needs rows [6, 12) from old ranks 3, 4, 5
    assert planner.recv_row_splits(1) == [0, 0, 0, 2, 2, 2]
    assert planner.recv_row_splits(0) == [2, 2, 2, 0, 0, 0]
    assert planner.network_recv_rows(0) == 4  # keeps its own [0, 2)
    assert planner.network_recv_rows(1) == 6  # its old rows [2, 4) are not part of its new shard


def test_non_prefix_rank_sets_are_indexed_by_mesh_position():
    planner = FSDPResizePlanner(8, old_ranks=[2, 3], new_ranks=[1, 2, 3, 4])

    assert planner.group_ranks == [1, 2, 3, 4]
    # old rank 2 is mesh position 0 -> rows [0, 4); new rank 1 is position 0 -> [0, 2), new rank 2 -> [2, 4)
    assert planner.send_row_splits(2) == [2, 2, 0, 0]
    assert planner.send_row_splits(3) == [0, 0, 2, 2]
    assert planner.recv_row_splits(1) == [0, 2, 0, 0]
    assert planner.recv_row_splits(4) == [0, 0, 2, 0]
    # ranks 2 and 3 keep rows they already hold; ranks 1 and 4 are new members
    assert planner.network_recv_rows(2) == 0
    assert planner.network_recv_rows(3) == 0
    assert planner.network_recv_rows(1) == 2
    assert planner.network_recv_rows(4) == 2


@pytest.mark.parametrize(
    ("dim_size", "old_ranks", "new_ranks"),
    [
        (12, [0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),
        (13, [0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),
        (13, [0, 1, 2, 3, 4, 5], [0, 1]),
        (3, [0, 1], [0, 1, 2, 3, 4]),
        (0, [0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),
        (7, [4, 5], [0, 1, 2]),
    ],
)
def test_planner_conserves_rows(dim_size, old_ranks, new_ranks):
    planner = FSDPResizePlanner(dim_size, old_ranks, new_ranks)

    for index, rank in enumerate(old_ranks):
        assert sum(planner.send_row_splits(rank)) == planner.source_intervals[index].size
    for index, rank in enumerate(new_ranks):
        assert sum(planner.recv_row_splits(rank)) == planner.target_intervals[index].size
    assert sum(sum(planner.send_row_splits(r)) for r in old_ranks) == dim_size
    assert sum(sum(planner.recv_row_splits(r)) for r in new_ranks) == dim_size
    # what one side sends is exactly what the other side receives, pairwise
    for s, src in enumerate(planner.group_ranks):
        for t, dst in enumerate(planner.group_ranks):
            sent = planner.send_row_splits(src)[t] if src in old_ranks else 0
            received = planner.recv_row_splits(dst)[s] if dst in new_ranks else 0
            assert sent == received


def test_planner_handles_uneven_and_empty_target_shards():
    planner = FSDPResizePlanner(3, old_ranks=[0, 1], new_ranks=[0, 1, 2, 3, 4])

    assert planner.send_row_splits(0) == [1, 1, 0, 0, 0]
    assert planner.send_row_splits(1) == [0, 0, 1, 0, 0]
    assert planner.recv_row_splits(3) == [0, 0, 0, 0, 0]
    assert planner.recv_row_splits(4) == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("old_ranks,new_ranks", [([], [0]), ([0], []), ([1, 0], [0, 1]), ([0, 1], [0, 0, 1])])
def test_planner_rejects_empty_or_unsorted_rank_lists(old_ranks, new_ranks):
    with pytest.raises(ValueError):
        FSDPResizePlanner(8, old_ranks, new_ranks)


def test_low_memory_pipeline_orders_units_to_minimise_the_peak():
    units = [ResizeUnit("<root>", ["embed"], []), ResizeUnit("layers.0", ["a"], []), ResizeUnit("layers.1", ["b"], [])]
    sizes = {"<root>": 100, "layers.0": 10, "layers.1": 20}

    # expanding: memory falls per unit, so the smallest unit goes first (peak = old + smallest new shard)
    assert [u.name for u in order_units(units, sizes, expanding=True)] == ["layers.0", "layers.1", "<root>"]
    # shrinking: memory rises per unit, so the largest unit goes first (peak = new + smallest old shard)
    assert [u.name for u in order_units(units, sizes, expanding=False)] == ["<root>", "layers.1", "layers.0"]
