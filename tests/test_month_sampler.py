import torch

from src.data.sampling import TargetMonthBatchSampler


def test_month_batches_keep_every_target_and_do_not_mix_months():
    samples = [{"target_time": t} for t in [3, 1, 2, 1, 3, 3, 2]]
    sampler = TargetMonthBatchSampler(samples, shuffle=False)
    batches = list(sampler)
    assert batches == [[1, 3], [2, 6], [0, 4, 5]]
    assert sorted(i for batch in batches for i in batch) == list(range(7))


def test_month_shuffle_is_reproducible_and_advances():
    samples = [{"target_time": t} for t in range(20)]
    a = TargetMonthBatchSampler(samples, generator=torch.Generator().manual_seed(26))
    b = TargetMonthBatchSampler(samples, generator=torch.Generator().manual_seed(26))
    first = list(a)
    assert first == list(b)
    second = list(a)
    assert second == list(b)
    assert first != second


def test_pipeline_month_batches_and_partial_aggregate_batches():
    from tests.test_tabular_target_alignment import _pipe
    pipe = _pipe(True)
    pipe.config.batch_by_target_month = True
    train, _, _ = pipe.get_dataloaders(7, num_workers=0, shuffle_train=False)
    seen = 0
    for batch in train:
        assert len(set(batch["target_time"])) == 1
        seen += len(batch["target_time"])
    assert seen == len(train.dataset)
    pipe.config.batch_by_target_month = False
    pipe.config.train_drop_last = False
    train, _, _ = pipe.get_dataloaders(7, num_workers=0, shuffle_train=False)
    assert sum(len(batch["target_time"]) for batch in train) == len(train.dataset)
