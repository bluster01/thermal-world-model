"""Prepare the deterministic private input schedule, without constructing a model."""
import argparse
import json
from pathlib import Path

from .data import PlantDataset, text_sha256


def prepare(dataset_path, output, *, expected_identity_path, updates=250, batch_size=4,
            sample_seed=72001, anchors_per_side=16):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing private input plan')
    dataset = PlantDataset.load(dataset_path, expected_identity_path=expected_identity_path)
    plan = dataset.plan(updates=updates, batch_size=batch_size, sample_seed=sample_seed, anchors_per_side=anchors_per_side)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return dict(plan_sha256=text_sha256(output), requested_batches=len(plan['training']),
                diagnostic_windows_per_split={k: len(v) for k, v in plan['diagnostic_anchors'].items()},
                model_constructed=False, optimizer_run=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected', type=Path, default=Path(__file__).with_name('expected_data_identity_v1.json'))
    parser.add_argument('--updates', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--sample-seed', type=int, default=72001)
    parser.add_argument('--anchors-per-side', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output, expected_identity_path=args.expected,
                            updates=args.updates, batch_size=args.batch_size,
                            sample_seed=args.sample_seed, anchors_per_side=args.anchors_per_side), indent=2))
