from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from depth_anything_3.api import DepthAnything3


BASE_DIR = Path(r"C:\Users\hmbar\Downloads\HCI_CV_1_outputs_gpu")
FOLLOWUP_DIR = BASE_DIR / "phd_followup"
RUN_DIR = BASE_DIR / "da3_runs"
RESTAURANTS_DIR = RUN_DIR / "restaurants"

CANDIDATES_CSV = FOLLOWUP_DIR / "restaurant_interior_40plus_candidates.csv"
MANIFEST_CSV = FOLLOWUP_DIR / "da3_candidate_image_manifest.csv"

DEFAULT_MODEL_DIR = "depth-anything/DA3-SMALL"
DEFAULT_PROCESS_RES = 336
DEFAULT_EXPORT_FORMAT = "mini_npz-depth_vis"


def slugify(value: str, max_len: int = 70) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return value[:max_len] or "restaurant"


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESTAURANTS_DIR.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_restaurant_inputs(row: pd.Series, manifest: pd.DataFrame) -> Path:
    restaurant_dir = restaurant_output_dir(row)
    input_dir = restaurant_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    subset = manifest[manifest["business_id"].eq(row["business_id"])].sort_values("photo_id")
    for _, image_row in subset.iterrows():
        src = Path(image_row["image_path"])
        dst = input_dir / f"{image_row['photo_id']}.jpg"
        if not src.exists():
            raise FileNotFoundError(src)
        copy_or_link(src, dst)
    subset.to_csv(restaurant_dir / "restaurant_input_manifest.csv", index=False)
    return input_dir


def restaurant_output_dir(row: pd.Series) -> Path:
    return RESTAURANTS_DIR / f"{int(row['rank']):03d}_{slugify(row['name'])}_{row['business_id']}"


def count_outputs(output_dir: Path) -> dict[str, int]:
    output_files = [p for p in output_dir.rglob("*") if p.is_file()]
    npz_files = [p for p in output_files if p.suffix.lower() == ".npz"]
    depth_vis_files = [
        p
        for p in output_files
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and ("depth" in p.name.lower() or "vis" in str(p.parent).lower())
    ]
    glb_files = [p for p in output_files if p.suffix.lower() == ".glb"]
    return {
        "output_file_count": len(output_files),
        "npz_count": len(npz_files),
        "depth_visual_count": len(depth_vis_files),
        "glb_count": len(glb_files),
    }


def run_da3_for_restaurant(
    model: DepthAnything3,
    model_dir: str,
    process_res: int,
    export_format: str,
    row: pd.Series,
    input_dir: Path,
) -> dict[str, object]:
    restaurant_dir = restaurant_output_dir(row)
    output_dir = restaurant_dir / "outputs"
    logs_dir = restaurant_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "da3_stdout.log"
    stderr_path = logs_dir / "da3_stderr.log"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [str(path) for path in sorted(input_dir.glob("*.jpg"))]

    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr
        return_code = 0
        effective_res = process_res
        try:
            try:
                prediction = model.inference(
                    image=images,
                    export_dir=str(output_dir),
                    export_format=export_format,
                    process_res=process_res,
                    process_res_method="upper_bound_resize",
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                effective_res = min(224, process_res)
                print(f"CUDA OOM at process_res={process_res}; retrying full image set at {effective_res}")
                prediction = model.inference(
                    image=images,
                    export_dir=str(output_dir),
                    export_format=export_format,
                    process_res=effective_res,
                    process_res_method="upper_bound_resize",
                )
            print("processed_images", getattr(prediction, "processed_images", None).shape)
            print("depth", getattr(prediction, "depth", None).shape)
            print("conf", getattr(prediction, "conf", None).shape)
            print("extrinsics", None if prediction.extrinsics is None else prediction.extrinsics.shape)
            print("intrinsics", None if prediction.intrinsics is None else prediction.intrinsics.shape)
        except Exception as exc:
            return_code = 1
            print(repr(exc), file=stderr)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            torch.cuda.empty_cache()
    elapsed = time.time() - start
    counts = count_outputs(output_dir)

    return {
        "rank": int(row["rank"]),
        "business_id": row["business_id"],
        "name": row["name"],
        "city": row["city"],
        "state": row["state"],
        "inside_photo_count": int(row["inside_photo_count"]),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "return_code": return_code,
        "status": "success" if return_code == 0 else "failed",
        "elapsed_seconds": round(elapsed, 2),
        "process_res": effective_res,
        **counts,
    }


def safe_open(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except OSError:
        return Image.new("RGB", (180, 180), "white")


def find_depth_visuals(output_dir: Path) -> list[Path]:
    return [
        p
        for p in output_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and ("depth" in p.name.lower() or "vis" in str(p.parent).lower())
    ]


def make_contact_sheet(row: pd.Series, n: int = 18) -> None:
    restaurant_dir = restaurant_output_dir(row)
    input_images = sorted((restaurant_dir / "inputs").glob("*.jpg"))[:n]
    depth_images = find_depth_visuals(restaurant_dir / "outputs")[:n]
    if not input_images:
        return

    thumb_w, thumb_h = 160, 120
    title_h = 40
    rows = len(input_images)
    canvas = Image.new("RGB", (thumb_w * 2, title_h + rows * thumb_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 10), f"{row['name']} - original/depth samples", fill=(20, 20, 20), font=font)
    for idx, original_path in enumerate(input_images):
        y = title_h + idx * thumb_h
        original = ImageOps.fit(safe_open(original_path), (thumb_w, thumb_h))
        canvas.paste(original, (0, y))
        if idx < len(depth_images):
            depth = ImageOps.fit(safe_open(depth_images[idx]), (thumb_w, thumb_h))
            canvas.paste(depth, (thumb_w, y))
    canvas.save(restaurant_dir / "original_depth_contact_sheet.jpg", quality=92)


def write_restaurant_summary(row: pd.Series, result: dict[str, object]) -> None:
    restaurant_dir = restaurant_output_dir(row)
    with (restaurant_dir / "restaurant_da3_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)


def write_meeting_notes(results: list[dict[str, object]], args: argparse.Namespace) -> None:
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] != "success"]
    top_successes = "\n".join(
        f"- {r['rank']:03d} {r['name']} ({r['city']}, {r['state']}): "
        f"{r['inside_photo_count']} images, {r['npz_count']} npz, {r['depth_visual_count']} depth visuals"
        for r in successes[:12]
    ) or "- None yet"
    failure_lines = "\n".join(
        f"- {r['rank']:03d} {r['name']}: return code {r['return_code']}, stderr log {r['stderr_log']}"
        for r in failures[:12]
    ) or "- None"

    notes = f"""# DA3 Full Restaurant Interior Run

## Run Setup

- Candidate threshold: Yelp `inside` photos >= 40
- Restaurants scheduled: {len(results)}
- Successful DA3 runs: {len(successes)}
- Failed DA3 runs: {len(failures)}
- Model: `{args.model_dir}`
- Process resolution: `{args.process_res}`
- Export format: `{args.export_format}`

## Why These Restaurants

These restaurants were selected because they have many Yelp photos labeled `inside`. The `inside` label is Yelp-provided photo metadata, and the restaurant grouping comes from the earlier broad business-category mapping.

## What DA3 Adds

DA3 estimates depth/geometry and, when available, camera-related outputs from the repeated interior images. This supports manual inspection of viewpoint spread, layout visibility, repeated furniture/seating, people/crowd occlusion, table clutter, and other environment-state variation.

## Successful Runs

{top_successes}

## Failed Runs

{failure_lines}

## Interpretation Reminder

DA3 outputs are estimates, not ground-truth camera geometry. Furniture changes, objects on furniture, and occluders still need manual inspection or a follow-up labeling pass using the DA3 contact sheets and original images.
"""
    (RUN_DIR / "da3_meeting_notes.md").write_text(notes, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES)
    parser.add_argument("--export-format", default=DEFAULT_EXPORT_FORMAT)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; DA3 full run requires the GPU environment.")
    print(f"Loading DA3 model {args.model_dir} on {torch.cuda.get_device_name(0)}")
    model = DepthAnything3.from_pretrained(args.model_dir).to("cuda")

    candidates = pd.read_csv(CANDIDATES_CSV).sort_values("rank")
    manifest = pd.read_csv(MANIFEST_CSV)
    if len(candidates) != 36:
        print(f"WARNING: expected 36 candidates, found {len(candidates)}")
    missing = manifest[~manifest["image_exists"].astype(bool)]
    if not missing.empty:
        raise FileNotFoundError(f"{len(missing)} manifest image paths are missing")

    if args.smoke_test:
        candidates = candidates.head(1).copy()
        first_id = candidates.iloc[0]["business_id"]
        manifest = manifest[manifest["business_id"].eq(first_id)].head(5).copy()

    results: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        restaurant_dir = restaurant_output_dir(row)
        summary_path = restaurant_dir / "restaurant_da3_summary.csv"
        if summary_path.exists() and not args.force:
            existing = pd.read_csv(summary_path).iloc[0].to_dict()
            results.append(existing)
            print(f"Skipping existing {int(row['rank']):03d} {row['name']}")
            continue

        print(f"Preparing {int(row['rank']):03d} {row['name']} ({int(row['inside_photo_count'])} images)")
        try:
            input_dir = prepare_restaurant_inputs(row, manifest)
            result = run_da3_for_restaurant(
                model=model,
                model_dir=args.model_dir,
                process_res=args.process_res,
                export_format=args.export_format,
                row=row,
                input_dir=input_dir,
            )
            make_contact_sheet(row)
            write_restaurant_summary(row, result)
            results.append(result)
            print(f"  {result['status']} rc={result['return_code']} outputs={result['output_file_count']}")
        except Exception as exc:
            restaurant_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = restaurant_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            error_path = logs_dir / "runner_error.log"
            error_path.write_text(repr(exc), encoding="utf-8")
            result = {
                "rank": int(row["rank"]),
                "business_id": row["business_id"],
                "name": row["name"],
                "city": row["city"],
                "state": row["state"],
                "inside_photo_count": int(row["inside_photo_count"]),
                "input_dir": str(restaurant_dir / "inputs"),
                "output_dir": str(restaurant_dir / "outputs"),
                "stdout_log": "",
                "stderr_log": str(error_path),
                "return_code": -1,
                "status": "runner_failed",
                "elapsed_seconds": 0,
                "output_file_count": 0,
                "npz_count": 0,
                "depth_visual_count": 0,
                "glb_count": 0,
            }
            write_restaurant_summary(row, result)
            results.append(result)
            print(f"  runner_failed {exc!r}")

        pd.DataFrame(results).to_csv(RUN_DIR / "da3_run_summary.csv", index=False)
        write_meeting_notes(results, args)

    pd.DataFrame(results).to_csv(RUN_DIR / "da3_run_summary.csv", index=False)
    write_meeting_notes(results, args)
    print(f"Wrote DA3 summary to {RUN_DIR / 'da3_run_summary.csv'}")


if __name__ == "__main__":
    main()
