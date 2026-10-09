"""Route mix validation groups to baseline, LoRA-Clean, or LoRA-Hazy.

The input directory is produced by ``split_mix_by_density.py``.  Each domain
and density combination is evaluated separately using this fixed routing:

    baseline:   clean-low, clean-medium
    LoRA-Clean: clean-high
    LoRA-Hazy:  hazy-low, hazy-medium, hazy-high

Example:

    python test_mix_density.py \
        --baseline-checkpoint path/to/baseline/best_model.pth \
        --lora-clean-checkpoint path/to/lora-clean/best_model.pth \
        --lora-hazy-checkpoint path/to/lora-hazy/best_model.pth
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.crowd import Crowd
from test import build_model, resolve_checkpoint, split_large_image


ROUTES = {
    'baseline': (
        ('clean', 'low'),
        ('clean', 'medium')
    ),
    'lora-clean': (
        ('clean', 'high'),
    ),
    'lora-hazy': (
        ('hazy', 'low'),
        ('hazy', 'high'),
        ('hazy', 'medium')
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Route density groups to baseline, LoRA-Clean, or '
                    'LoRA-Hazy and evaluate every group separately')
    parser.add_argument('--model-name', choices=['swin_l_trans', 'swin_t_trans'],
                        default='swin_t_trans')
    parser.add_argument(
        '--baseline-checkpoint',
        default='model/swin_t_trans/baseline/0922-200006/best_model.pth',
        help='baseline checkpoint; use an empty value to select the latest one')
    parser.add_argument(
        '--lora-clean-checkpoint', default='',
        help='LoRA-Clean checkpoint; defaults to the latest best_model*.pth')
    parser.add_argument(
        '--lora-hazy-checkpoint', default='',
        help='LoRA-Hazy checkpoint; defaults to the latest best_model*.pth')
    parser.add_argument('--checkpoint-root', default='model',
                        help='root used to find the latest checkpoints')
    parser.add_argument('--data-root', default='SHA/mix_split',
                        help='root containing the density-labelled split')
    parser.add_argument('--split', default='val',
                        help='split directory below --data-root (default: val)')
    parser.add_argument('--output-csv', default='test_mix_density_results.csv')
    parser.add_argument('--device', default='0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--downsample-ratio', type=int, default=16)
    parser.add_argument('--lora-rank', type=int, default=4,
                        help='required when the LoRA checkpoint is a .pth file')
    parser.add_argument('--lora-alpha', type=float, default=4.0,
                        help='must match training for a LoRA .pth checkpoint')
    return parser.parse_args()


def build_group_dataset(data_dir, domain, density, args):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError('Dataset directory not found: {}'.format(data_dir))

    dataset = Crowd(data_dir, args.crop_size, args.downsample_ratio,
                    is_gray=False, method='val')
    prefix = 'mix_{}_{}_'.format(density, domain)
    dataset.im_list = [
        path for path in dataset.im_list
        if os.path.basename(path).startswith(prefix)
    ]
    if not dataset.im_list:
        raise ValueError(
            'No {}*.jpg images found in {}'.format(prefix, data_dir))

    missing_gt = [
        path for path in dataset.im_list
        if not os.path.isfile(os.path.splitext(path)[0] + '.npy')
    ]
    if missing_gt:
        raise FileNotFoundError(
            'Missing matching .npy file for {}'.format(missing_gt[0]))
    return dataset


def evaluate_group(model, data_dir, domain, density, args, device, description):
    dataset = build_group_dataset(data_dir, domain, density, args)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    errors = []
    with torch.inference_mode():
        for inputs, count, _ in tqdm(
                loader, desc=description, dynamic_ncols=True):
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
            predicted = sum(
                model(patch)[0].sum().item()
                for patch in split_large_image(inputs)
            )
            errors.append(count.item() - predicted)

    errors = np.asarray(errors, dtype=np.float64)
    return {
        'images': len(errors),
        'absolute_error_sum': float(np.abs(errors).sum()),
        'squared_error_sum': float(np.square(errors).sum()),
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(np.square(errors)))),
    }


def combine_metrics(metrics_list):
    """Combine groups from their per-image error sums."""
    images = sum(metrics['images'] for metrics in metrics_list)
    if images == 0:
        raise ValueError('Cannot calculate metrics for an empty group')
    absolute_error_sum = sum(
        metrics['absolute_error_sum'] for metrics in metrics_list)
    squared_error_sum = sum(
        metrics['squared_error_sum'] for metrics in metrics_list)
    return {
        'images': images,
        'absolute_error_sum': absolute_error_sum,
        'squared_error_sum': squared_error_sum,
        'mae': absolute_error_sum / images,
        'rmse': float(np.sqrt(squared_error_sum / images)),
    }


def print_results(rows):
    print('\n{:<12} {:<8} {:<10} {:<12} {:>8} {:>12} {:>12}'.format(
        'Model', 'Domain', 'Density', 'Scope', 'Images', 'MAE', 'RMSE'))
    print('-' * 81)
    for row in rows:
        print('{model:<12} {domain:<8} {density:<10} {scope:<12} {images:>8d} '
              '{mae:>12.3f} {rmse:>12.3f}'.format(**row))


def write_results(rows, output_path):
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    fields = [
        'model', 'domain', 'density', 'scope', 'split', 'images', 'mae',
        'rmse', 'checkpoint',
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join(args.data_root, args.split)

    # Validate all six routed groups before loading the large model weights.
    routed_images = set()
    for model_label, groups in ROUTES.items():
        for domain, density in groups:
            dataset = build_group_dataset(data_dir, domain, density, args)
            overlap = routed_images.intersection(dataset.im_list)
            if overlap:
                raise ValueError('An image was assigned to multiple routes: {}'.format(
                    next(iter(overlap))))
            routed_images.update(dataset.im_list)
            print('{} -> {:<8} {:<6}: {} images'.format(
                model_label, domain, density, len(dataset)))

    all_images = set(Crowd(
        data_dir, args.crop_size, args.downsample_ratio,
        is_gray=False, method='val').im_list)
    if routed_images != all_images:
        missing = sorted(all_images - routed_images)
        raise ValueError(
            '{} images do not match the six routing groups; first: {}'.format(
                len(missing), missing[0] if missing else 'unknown'))

    baseline_checkpoint = resolve_checkpoint(
        args.baseline_checkpoint, args.checkpoint_root,
        args.model_name, 'baseline')
    lora_clean_checkpoint = resolve_checkpoint(
        args.lora_clean_checkpoint, args.checkpoint_root,
        args.model_name, 'lora-clean')
    lora_hazy_checkpoint = resolve_checkpoint(
        args.lora_hazy_checkpoint, args.checkpoint_root,
        args.model_name, 'lora-hazy')
    print('Baseline checkpoint: {}'.format(baseline_checkpoint))
    print('LoRA-Clean checkpoint: {}'.format(lora_clean_checkpoint))
    print('LoRA-Hazy checkpoint: {}'.format(lora_hazy_checkpoint))

    rows = []
    all_group_metrics = []
    checkpoints = (
        ('baseline', baseline_checkpoint, False, ROUTES['baseline']),
        ('lora-clean', lora_clean_checkpoint, True, ROUTES['lora-clean']),
        ('lora-hazy', lora_hazy_checkpoint, True, ROUTES['lora-hazy']),
    )
    for model_label, checkpoint_path, expect_lora, groups in checkpoints:
        model = build_model(args, checkpoint_path, expect_lora)
        model.to(device).eval()
        route_metrics = []
        for domain, density in groups:
            metrics = evaluate_group(
                model, data_dir, domain, density, args, device,
                '{} {} {}'.format(model_label.capitalize(), domain, density))
            route_metrics.append(metrics)
            all_group_metrics.append(metrics)
            rows.append({
                'model': model_label,
                'domain': domain,
                'density': density,
                'scope': 'category',
                'split': args.split,
                'images': metrics['images'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'checkpoint': checkpoint_path,
            })

        route_total = combine_metrics(route_metrics)
        rows.append({
            'model': model_label,
            'domain': 'all',
            'density': 'all',
            'scope': 'route_total',
            'split': args.split,
            'images': route_total['images'],
            'mae': route_total['mae'],
            'rmse': route_total['rmse'],
            'checkpoint': checkpoint_path,
        })
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    overall = combine_metrics(all_group_metrics)
    rows.append({
        'model': 'hybrid',
        'domain': 'all',
        'density': 'all',
        'scope': 'overall',
        'split': args.split,
        'images': overall['images'],
        'mae': overall['mae'],
        'rmse': overall['rmse'],
        'checkpoint': 'baseline={} | lora-clean={} | lora-hazy={}'.format(
            baseline_checkpoint, lora_clean_checkpoint, lora_hazy_checkpoint),
    })

    print_results(rows)
    write_results(rows, args.output_csv)
    print('\nCSV saved to {}'.format(os.path.abspath(args.output_csv)))


if __name__ == '__main__':
    main()
