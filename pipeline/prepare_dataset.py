# prepare_dataset.py
"""
Preprocess ARW bracket triplets into aligned 16-bit PNGs for training.

Usage:
    # 50-pair test run
    python prepare_dataset.py --arw-root /path/to/raw --output-root dataset/ \
        --classifier labeling/classifier.pth --limit 50

    # Full run (all CPU cores)
    python prepare_dataset.py --arw-root /path/to/raw --output-root dataset/ \
        --classifier labeling/classifier.pth --workers 16
"""
import argparse
import rawpy
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import cv2
import exifread
from PIL import Image
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count, set_start_method
import json
import shutil
import time
import sys

from alignment import compute_homography, apply_homography

# -- Scene classifier loading --

def load_classifier(checkpoint_path, device):
    """Load trained ResNet-18 scene classifier."""
    import torchvision.models as models
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine number of classes — checkpoint may use different key names
    num_classes = checkpoint.get('num_classes', 3)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Try common state dict key names
    for key in ['model_state', 'model_state_dict', 'state_dict']:
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            break
    else:
        # Assume checkpoint IS the state dict
        model.load_state_dict(checkpoint)

    model.to(device).eval()

    label_names = checkpoint.get('label_names', ['interior', 'exterior', 'drone'][:num_classes])
    return model, label_names


def classify_scene(image_path, model, label_names, device):
    """Classify a single image. Returns (label, confidence)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, idx = probs.max(dim=1)

    return label_names[idx.item()], conf.item()


# -- ARW conversion --

def convert_arw_to_16bit(arw_path):
    """Convert ARW to 16-bit numpy array. Returns (H, W, 3) uint16 RGB array."""
    with rawpy.imread(str(arw_path)) as raw:
        arr = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=16,
            user_flip=0
        )
    return arr


def convert_arw_to_8bit(arw_path):
    """Convert ARW to 8-bit numpy array for classification/alignment."""
    with rawpy.imread(str(arw_path)) as raw:
        arr = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=8,
            user_flip=0
        )
    return arr


# -- Bracket matching --

MIN_BOUNDARY_GAP = 5.0  # seconds between bracket groups

def get_exif_timestamp(arw_path):
    """Extract DateTimeOriginal from EXIF."""
    with open(arw_path, 'rb') as f:
        tags = exifread.process_file(f, stop_tag='EXIF DateTimeOriginal')
    dt_tag = tags.get('EXIF DateTimeOriginal')
    if dt_tag is None:
        return None
    return datetime.strptime(str(dt_tag), '%Y:%m:%d %H:%M:%S')


def find_bracket_triplets(arw_dir):
    """
    Find bracket triplets in a directory of ARW files.
    Returns list of (under_path, mid_path, over_path) tuples.
    """
    arw_files = sorted(Path(arw_dir).glob('DSC*.ARW'))
    if not arw_files:
        arw_files = sorted(Path(arw_dir).glob('DSC*.arw'))
    if len(arw_files) < 3:
        return []

    # Cache timestamps
    timestamps = {}
    for f in arw_files:
        ts = get_exif_timestamp(str(f))
        if ts:
            timestamps[f] = ts

    if not timestamps:
        return []

    # Cluster by timestamp gaps
    sorted_files = sorted(timestamps.keys(), key=lambda f: timestamps[f])
    triplets = []
    cluster = [sorted_files[0]]

    for i in range(1, len(sorted_files)):
        gap = (timestamps[sorted_files[i]] - timestamps[sorted_files[i-1]]).total_seconds()
        if gap > MIN_BOUNDARY_GAP:
            if len(cluster) == 3:
                triplets.append(tuple(cluster))
            cluster = []
        cluster.append(sorted_files[i])

    if len(cluster) == 3:
        triplets.append(tuple(cluster))

    return triplets


# -- Alignment and conversion (parallelizable) --

def process_single_pair(args):
    """
    Process a single pair: demosaic, align, convert to output format.
    Designed to be called from multiprocessing Pool.

    Args is a dict with keys:
        under_arw, mid_arw, over_arw, target_jpg, input_dir, output_dir,
        pair_id, label, confidence, shoot_folder, output_format
    Returns a dict with status and pair_entry (or error info).
    """
    under_arw = args['under_arw']
    mid_arw = args['mid_arw']
    over_arw = args['over_arw']
    target_jpg = args['target_jpg']
    input_dir = Path(args['input_dir'])
    output_dir = Path(args['output_dir'])
    pair_id = args['pair_id']
    label = args['label']
    confidence = args['confidence']
    shoot_folder = args['shoot_folder']
    fmt = args.get('output_format', 'png16')
    resize_mp = args.get('resize_mp', None)  # Target megapixels (e.g., 12)
    output_root = Path(args.get('output_root', '')) if args.get('output_root') else None

    ext = '.jpg' if fmt == 'jpg' else '.png'

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed (resumable)
    under_out = input_dir / f"{pair_id}_under{ext}"
    mid_out = input_dir / f"{pair_id}_mid{ext}"
    over_out = input_dir / f"{pair_id}_over{ext}"
    target_out = output_dir / f"{pair_id}.jpg"

    def _rel(p):
        """Return path relative to output_root if set, otherwise absolute."""
        if output_root:
            return str(Path(p).relative_to(output_root))
        return str(p)

    if under_out.exists() and under_out.stat().st_size > 0 and mid_out.exists() and mid_out.stat().st_size > 0 and over_out.exists() and over_out.stat().st_size > 0:
        if not target_out.exists():
            shutil.copy2(target_jpg, target_out)
        return {
            'status': 'success',
            'pair_entry': {
                'pair_id': pair_id,
                'shoot_folder': shoot_folder,
                'input_under': _rel(under_out),
                'input_mid': _rel(mid_out),
                'input_over': _rel(over_out),
                'output': _rel(target_out),
                'scene_type': label,
                'scene_confidence': round(confidence, 4),
            },
            'skipped': True,
        }

    try:
        # Load target for alignment reference
        target_img = np.array(Image.open(target_jpg).convert('RGB'), dtype=np.uint8)
        target_h, target_w = target_img.shape[:2]

        # Demosaic mid bracket to 8-bit for feature matching
        mid_8bit = convert_arw_to_8bit(mid_arw)

        # Compute homography
        H, h_status = compute_homography(mid_8bit, target_img)

        if h_status != 'success':
            return {'status': 'alignment_failed', 'pair_id': pair_id, 'error': h_status}

        # Compute resize dimensions if requested
        resize_dim = None
        if resize_mp:
            current_mp = (target_h * target_w) / 1e6
            if current_mp > resize_mp:
                scale = (resize_mp / current_mp) ** 0.5
                resize_dim = (int(target_w * scale), int(target_h * scale))
                # Ensure divisible by 16
                resize_dim = ((resize_dim[0] // 16) * 16, (resize_dim[1] // 16) * 16)

        if fmt == 'jpg':
            # JPG mode: demosaic to 8-bit, align, optionally resize, save as JPG
            for arw_path, out_path, existing_8bit in [
                (under_arw, under_out, None),
                (mid_arw, mid_out, mid_8bit),
                (over_arw, over_out, None),
            ]:
                arr = existing_8bit if existing_8bit is not None else convert_arw_to_8bit(arw_path)
                aligned = apply_homography(arr, H, (target_w, target_h))
                if resize_dim:
                    aligned = cv2.resize(aligned, resize_dim, interpolation=cv2.INTER_LANCZOS4)
                aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), aligned_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            # PNG16 mode: demosaic to 16-bit, align, optionally resize, save as 16-bit PNG
            for arw_path, out_path in [
                (under_arw, under_out),
                (mid_arw, mid_out),
                (over_arw, over_out),
            ]:
                arr_16 = convert_arw_to_16bit(arw_path)
                aligned = apply_homography(arr_16, H, (target_w, target_h))
                if resize_dim:
                    aligned = cv2.resize(aligned, resize_dim, interpolation=cv2.INTER_LANCZOS4)
                aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), aligned_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])

        # Copy (and optionally resize) target
        if not target_out.exists():
            if resize_dim:
                tgt_resized = cv2.resize(target_img, resize_dim, interpolation=cv2.INTER_LANCZOS4)
                tgt_bgr = cv2.cvtColor(tgt_resized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(target_out), tgt_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                shutil.copy2(target_jpg, target_out)

        return {
            'status': 'success',
            'pair_entry': {
                'pair_id': pair_id,
                'shoot_folder': shoot_folder,
                'input_under': _rel(under_out),
                'input_mid': _rel(mid_out),
                'input_over': _rel(over_out),
                'output': _rel(target_out),
                'scene_type': label,
                'scene_confidence': round(confidence, 4),
            },
            'skipped': False,
        }
    except Exception as e:
        return {'status': 'error', 'pair_id': pair_id, 'error': str(e)}


# -- Pairs CSV generation --

def generate_pairs_csv(pairs_list, csv_path):
    """Write pairs list to CSV."""
    df = pd.DataFrame(pairs_list)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} pairs to {csv_path}")


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="Preprocess ARW brackets to 16-bit PNG")
    parser.add_argument('--arw-root', required=True,
                        help='Root directory containing shoot folders with ARW files')
    parser.add_argument('--finished-folder', default='05-Finished-Photos',
                        help='Name of finished photos subfolder')
    parser.add_argument('--raw-folder', default='01-RAW-Photos',
                        help='Name of raw photos subfolder')
    parser.add_argument('--output-root', default='dataset/',
                        help='Output directory for processed files')
    parser.add_argument('--classifier', required=True,
                        help='Path to scene classifier checkpoint')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of pairs to process (for testing)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: all CPU cores)')
    parser.add_argument('--format', default='png16', choices=['png16', 'jpg'],
                        help='Output format: png16 (16-bit PNG) or jpg (8-bit JPEG, faster)')
    parser.add_argument('--resize', default=None,
                        help='Resize to target megapixels, e.g. "12" for 12MP')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip Phase 1, load work items from cached phase1_work_items.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    n_workers = args.workers or cpu_count()
    device = torch.device(args.device)
    output_root = Path(args.output_root)

    # Load classifier
    print("Loading scene classifier...")
    classifier, label_names = load_classifier(args.classifier, device)
    print(f"Labels: {label_names}")

    # Scan for shoot folders
    arw_root = Path(args.arw_root)
    shoot_folders = sorted([
        d for d in arw_root.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    print(f"Found {len(shoot_folders)} shoot folders")

    # =====================================================================
    # PHASE 1: Scan, match brackets, classify scenes (serial — uses GPU)
    # =====================================================================
    phase1_cache = output_root / 'phase1_work_items.json'

    if args.skip_phase1 and phase1_cache.exists():
        print(f"\n--- Phase 1: SKIPPED (loading from {phase1_cache}) ---")
        with open(phase1_cache, 'r') as f:
            work_items = json.load(f)
        skipped = {'exterior': 0, 'drone': 0, 'twilight': 0,
                   'alignment_failed': 0, 'no_triplets': 0, 'no_finished': 0, 'error': 0}
        print(f"Loaded {len(work_items)} work items from cache")
        # Evict source files from RAM cache if vmtouch is available
        import shutil
        if shutil.which('vmtouch'):
            import subprocess
            subprocess.run(['vmtouch', '-e', str(Path(args.arw_root))], capture_output=True)
    else:
        print(f"\n--- Phase 1: Scanning and classifying ---")
        work_items = []
        skipped = {'exterior': 0, 'drone': 0, 'twilight': 0,
               'alignment_failed': 0, 'no_triplets': 0, 'no_finished': 0, 'error': 0}

        for shoot_dir in shoot_folders:
            if args.limit and len(work_items) >= args.limit:
                break

            raw_dir = shoot_dir / args.raw_folder
            finished_dir = shoot_dir / args.finished_folder

            if not raw_dir.exists():
                continue
            if not finished_dir.exists():
                skipped['no_finished'] += 1
                continue

            # Find bracket triplets
            triplets = find_bracket_triplets(raw_dir)
            if not triplets:
                skipped['no_triplets'] += 1
                continue

            # Find finished JPGs
            finished_jpgs = {
                f.stem.split('_')[-1] if '_' in f.stem else f.stem: f
                for f in finished_dir.glob('*.jpg')
            }
            finished_jpgs.update({
                f.stem.split('_')[-1] if '_' in f.stem else f.stem: f
                for f in finished_dir.glob('*.JPG')
            })

            for under, mid, over in triplets:
                if args.limit and len(work_items) >= args.limit:
                    break

                # Match to finished JPG
                target_jpg = (finished_jpgs.get(under.stem) or
                              finished_jpgs.get(mid.stem) or
                              finished_jpgs.get(over.stem))
                if target_jpg is None:
                    continue

                # Classify scene using mid bracket (quick 8-bit demosaic)
                temp_path = output_root / 'temp_classify.jpg'
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    mid_8bit = convert_arw_to_8bit(mid)
                    Image.fromarray(mid_8bit).save(str(temp_path), quality=85)
                    label, confidence = classify_scene(
                        str(temp_path), classifier, label_names, device
                    )
                except Exception as e:
                    print(f"  Classification failed for {under.stem}: {e}")
                    skipped['error'] = skipped.get('error', 0) + 1
                    continue

                if label != 'interior':
                    skipped[label] = skipped.get(label, 0) + 1

                # Route by scene type
                scene_dir = 'interior' if label == 'interior' else 'exterior'
                input_dir = output_root / scene_dir / 'inputs'
                output_dir = output_root / scene_dir / 'outputs'

                shoot_name = shoot_dir.name[:15].replace(' ', '_')
                pair_id = f"{shoot_name}_{under.stem}"

                work_items.append({
                    'under_arw': str(under),
                    'mid_arw': str(mid),
                    'over_arw': str(over),
                    'target_jpg': str(target_jpg),
                    'input_dir': str(input_dir),
                    'output_dir': str(output_dir),
                    'pair_id': pair_id,
                    'label': label,
                    'confidence': confidence,
                    'shoot_folder': shoot_dir.name,
                    'output_format': args.format,
                    'resize_mp': float(args.resize) if args.resize else None,
                    'output_root': str(output_root),
                })
                print(f"  [{len(work_items)}] {pair_id} — {label} ({confidence:.4f})", flush=True)

                # Incremental save every 1000 pairs + evict RAM cache
                if len(work_items) % 1000 == 0:
                    with open(phase1_cache, 'w') as f:
                        json.dump(work_items, f)
                    # Evict processed files from RAM cache
                    import shutil as _shutil
                    if _shutil.which('vmtouch'):
                        import subprocess as _sp
                        _sp.run(['vmtouch', '-e', str(raw_dir)], capture_output=True)
                    print(f"  [checkpoint] Saved {len(work_items)} work items", flush=True)

        # Clean up temp file
        temp_path = output_root / 'temp_classify.jpg'
        if temp_path.exists():
            temp_path.unlink()

        # Save Phase 1 results so we can skip it on restart
        with open(phase1_cache, 'w') as f:
            json.dump(work_items, f)
        print(f"Phase 1 results saved to {phase1_cache}")

    print(f"\nFound {len(work_items)} pairs to process")

    # =====================================================================
    # PHASE 2: Convert and align in parallel (CPU-bound, no GPU needed)
    # =====================================================================
    print(f"\n--- Phase 2: Converting and aligning ({n_workers} workers) ---")
    start_time = time.time()

    pairs = []
    completed = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_single_pair, work_items):
            completed += 1
            if result['status'] == 'success':
                pairs.append(result['pair_entry'])
                tag = "cached" if result.get('skipped') else "done"
                print(f"  [{completed}/{len(work_items)}] {result['pair_entry']['pair_id']} — {tag}", flush=True)
            elif result['status'] == 'alignment_failed':
                skipped['alignment_failed'] += 1
                print(f"  [{completed}/{len(work_items)}] {result['pair_id']} — alignment failed: {result['error']}", flush=True)
            else:
                skipped['error'] += 1
                print(f"  [{completed}/{len(work_items)}] {result['pair_id']} — error: {result['error']}", flush=True)

    elapsed = time.time() - start_time

    # Generate CSV (interior only for training)
    interior_pairs = [p for p in pairs if p['scene_type'] == 'interior']
    generate_pairs_csv(interior_pairs, output_root / 'pairs.csv')

    # Summary
    print(f"\n{'='*50}")
    print(f"PREPROCESSING COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*50}")
    print(f"Interior pairs: {len(interior_pairs)}")
    print(f"Exterior pairs: {len(pairs) - len(interior_pairs)}")
    print(f"Workers used: {n_workers}")
    print(f"Skipped: {json.dumps(skipped, indent=2)}")


if __name__ == '__main__':
    set_start_method('spawn', force=True)
    main()
