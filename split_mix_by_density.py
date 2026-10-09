"""Copy mix/val into one directory and label filenames by GT count.

The original dataset is left untouched. Images and their matching NumPy GT
files are copied into ``SHA/mix_split/val`` with the same density prefix::

    clean_IMG_1.jpg -> mix_low_clean_IMG_1.jpg
    clean_IMG_1.npy -> mix_low_clean_IMG_1.npy

Density ranges are inclusive: low=0..211, medium=212..500, high=501+.
"""

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


DENSITY_RANGES = (
    ('low', 0, 211),
    ('medium', 212, 500),
    ('high', 501, None),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Label mix/val filenames according to GT crowd count')
    parser.add_argument(
        '--source-dir', type=Path, default=Path('SHA/mix/val'),
        help='directory containing source .jpg/.npy pairs '
             '(default: SHA/mix/val)')
    parser.add_argument(
        '--output-dir', type=Path, default=Path('SHA/mix_split/val'),
        help='single directory for renamed pairs '
             '(default: SHA/mix_split/val)')
    parser.add_argument(
        '--mode', choices=['copy', 'hardlink', 'symlink'], default='copy',
        help='how files are created; copy is safest (default: copy)')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='replace files that already exist in the output directory')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='validate and print counts without creating any files')
    parser.add_argument(
        '--manifest', type=Path, default=None,
        help='CSV path (default: SHA/mix_split/mix_density_manifest.csv)')
    return parser.parse_args()


def density_group(count):
    for name, lower, upper in DENSITY_RANGES:
        if count >= lower and (upper is None or count <= upper):
            return name
    raise ValueError('Invalid negative GT count: {}'.format(count))


def scan_dataset(source_dir, output_dir):
    """Validate all source pairs and destination names before writing."""
    if not source_dir.is_dir():
        raise FileNotFoundError(
            'Source directory not found: {}'.format(source_dir))

    images = sorted(
        path for path in source_dir.glob('*.jpg')
        if not path.name.startswith('._'))
    if not images:
        raise ValueError('No .jpg images found in: {}'.format(source_dir))

    records = []
    errors = []
    destination_names = set()
    for image_path in images:
        gt_path = image_path.with_suffix('.npy')
        if not gt_path.is_file():
            errors.append('Missing GT for {}: {}'.format(image_path, gt_path))
            continue
        try:
            gt_count = len(np.load(
                gt_path, mmap_mode='r', allow_pickle=False))
        except Exception as exc:
            errors.append('Cannot read {}: {}'.format(gt_path, exc))
            continue

        density = density_group(gt_count)
        output_stem = 'mix_{}_{}'.format(density, image_path.stem)
        target_image = output_dir / (output_stem + image_path.suffix.lower())
        target_gt = output_dir / (output_stem + '.npy')
        for destination in (target_image, target_gt):
            if destination.name in destination_names:
                errors.append(
                    'Duplicate output filename: {}'.format(destination.name))
            destination_names.add(destination.name)

        records.append({
            'source_image': image_path,
            'source_gt': gt_path,
            'target_image': target_image,
            'target_gt': target_gt,
            'gt_count': gt_count,
            'density': density,
        })

    if errors:
        raise RuntimeError(
            'Dataset validation failed; no files were created:\n  - ' +
            '\n  - '.join(errors))
    return records


def create_file(source, destination, mode, overwrite):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return 'skipped'
        destination.unlink()

    if mode == 'copy':
        shutil.copy2(source, destination)
    elif mode == 'hardlink':
        os.link(source, destination)
    else:
        destination.symlink_to(source.resolve())
    return 'created'


def write_manifest(records, manifest_path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'original_filename', 'renamed_filename', 'gt_count', 'density',
        'source_image', 'source_gt', 'target_image', 'target_gt', 'status',
    ]
    with manifest_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = (args.manifest or
                     output_dir.parent / 'mix_density_manifest.csv').resolve()

    records = scan_dataset(source_dir, output_dir)
    counts = Counter(record['density'] for record in records)
    print('Source: {}'.format(source_dir))
    print('Output: {}'.format(output_dir))
    print('Ranges: low=0-211, medium=212-500, high=501+')
    print('total={:>4}  low={:>4}  medium={:>4}  high={:>4}'.format(
        len(records), counts['low'], counts['medium'], counts['high']))

    if args.dry_run:
        print('Dry run complete; no files were created.')
        return

    manifest_records = []
    for record in records:
        image_status = create_file(
            record['source_image'], record['target_image'],
            args.mode, args.overwrite)
        gt_status = create_file(
            record['source_gt'], record['target_gt'],
            args.mode, args.overwrite)
        status = ('created' if 'created' in (image_status, gt_status)
                  else 'skipped')
        manifest_records.append({
            'original_filename': record['source_image'].name,
            'renamed_filename': record['target_image'].name,
            'gt_count': record['gt_count'],
            'density': record['density'],
            'source_image': str(record['source_image']),
            'source_gt': str(record['source_gt']),
            'target_image': str(record['target_image']),
            'target_gt': str(record['target_gt']),
            'status': status,
        })

    write_manifest(manifest_records, manifest_path)
    created = sum(row['status'] == 'created' for row in manifest_records)
    skipped = len(manifest_records) - created
    print('Finished: {} image/GT pairs created, {} already existed.'.format(
        created, skipped))
    print('Manifest: {}'.format(manifest_path))


if __name__ == '__main__':
    main()
