from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


BASE_DIR = Path(r"C:\Users\hmbar\Downloads\HCI_CV_1_outputs_gpu")
CACHE_DIR = BASE_DIR / "cache"
OUT_DIR = BASE_DIR / "phd_followup"
SHEETS_DIR = OUT_DIR / "taxonomy_example_sheets"

PHOTOS_JOINED_PATH = CACHE_DIR / "photos_joined.pkl"
EMBEDDINGS_PATH = CACHE_DIR / "clip_image_embeddings_float32.npy"
PHOTO_IDS_PATH = CACHE_DIR / "clip_photo_ids.npy"

MIN_DA3_INTERIOR_PHOTOS = 40

OCCLUDER_ENVIRONMENT_PROMPTS = {
    "people_crowd": "a photo with people or a crowd blocking parts of the scene",
    "furniture_seating": "a photo showing furniture seating chairs tables booths or sofas",
    "tabletop_objects_clutter": "a photo with objects clutter plates cups menus or items on a table",
    "counter_bar_kitchen": "a photo of a counter bar kitchen service area or preparation area",
    "signage_menu_text": "a photo centered on signs menus text or printed information",
    "exterior_entrance": "a photo of a storefront exterior entrance doorway or threshold",
    "low_light_blur": "a blurry low light noisy or low quality photo",
    "partial_blocked_view": "a partial view where objects people or framing block the full scene",
}

TAXONOMY_PROMPTS = {
    "close_up_food_object": {
        "prompt": "a close-up photo of food drink or a small object",
        "prefer_labels": {"food", "drink"},
    },
    "table_level_seated": {
        "prompt": "a table-level photo from a seated customer position",
        "prefer_labels": {"food", "drink", "inside"},
    },
    "wide_interior": {
        "prompt": "a wide interior scene showing a room seating layout counter or business interior",
        "prefer_labels": {"inside"},
    },
    "exterior_storefront_threshold": {
        "prompt": "an exterior storefront entrance doorway patio or threshold photo",
        "prefer_labels": {"outside"},
    },
    "menu_signage": {
        "prompt": "a photo of a menu sign wall board or text",
        "prefer_labels": {"menu"},
    },
    "people_occlusion": {
        "prompt": "a photo with people crowd staff performers or people occluding the scene",
        "prefer_labels": {"inside", "outside"},
    },
    "low_context_blurred_partial": {
        "prompt": "a blurry low context partial cropped or incomplete evidence photo",
        "prefer_labels": {"inside", "outside", "food", "drink"},
    },
    "room_edge_corner_pathway": {
        "prompt": "a room corner edge of room aisle hallway or pathway interior view",
        "prefer_labels": {"inside"},
    },
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    photos = pd.read_pickle(PHOTOS_JOINED_PATH)
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    photo_ids = np.load(PHOTO_IDS_PATH, allow_pickle=True).astype(str)
    current_ids = photos["photo_id"].astype(str).to_numpy()
    if len(photos) != len(embeddings):
        raise ValueError(f"Photo rows ({len(photos)}) do not match embeddings ({len(embeddings)}).")
    if not np.array_equal(photo_ids, current_ids):
        raise ValueError("Cached CLIP photo IDs do not match photos_joined.pkl order.")
    return photos, embeddings


def load_clip_text_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    print(f"Using device for text prompts: {device}")
    return model, tokenizer, device


def encode_text_prompts(prompts: list[str]) -> np.ndarray:
    model, tokenizer, device = load_clip_text_model()
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu().numpy().astype("float32")


def score_prompts(embeddings: np.ndarray, prompt_map: dict[str, str], cache_name: str) -> pd.DataFrame:
    cache_path = OUT_DIR / cache_name
    names = list(prompt_map.keys())
    prompts = [prompt_map[name] for name in names]
    if cache_path.exists():
        scores = np.load(cache_path)
    else:
        text_embeddings = encode_text_prompts(prompts)
        scores = np.asarray(embeddings @ text_embeddings.T, dtype="float32")
        np.save(cache_path, scores)
    return pd.DataFrame(scores, columns=names)


def export_restaurant_interior_candidates(photos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    restaurants = photos[photos["business_group"].eq("restaurants")].copy()
    inside = restaurants[restaurants["label"].eq("inside")].copy()

    counts = (
        inside.groupby(
            ["business_id", "name", "city", "state", "categories", "stars", "review_count"],
            dropna=False,
        )
        .size()
        .reset_index(name="inside_photo_count")
        .sort_values("inside_photo_count", ascending=False)
        .reset_index(drop=True)
    )
    counts.insert(0, "rank", np.arange(1, len(counts) + 1))
    counts["has_40plus_inside"] = counts["inside_photo_count"] >= 40
    counts["has_45plus_inside"] = counts["inside_photo_count"] >= 45
    counts["has_50plus_inside"] = counts["inside_photo_count"] >= 50
    counts.to_csv(OUT_DIR / "restaurant_interior_counts.csv", index=False)

    candidates = counts[counts["inside_photo_count"] >= MIN_DA3_INTERIOR_PHOTOS].copy()
    candidates.to_csv(OUT_DIR / "restaurant_interior_40plus_candidates.csv", index=False)

    manifest = inside.merge(
        candidates[["rank", "business_id", "inside_photo_count"]],
        on="business_id",
        how="inner",
        validate="many_to_one",
    )
    manifest = manifest[
        [
            "rank",
            "business_id",
            "name",
            "city",
            "state",
            "inside_photo_count",
            "photo_id",
            "image_path",
            "caption",
            "label",
            "categories",
            "stars",
            "review_count",
        ]
    ].sort_values(["rank", "photo_id"])
    manifest["image_exists"] = manifest["image_path"].map(lambda path: Path(path).exists())
    manifest.to_csv(OUT_DIR / "da3_candidate_image_manifest.csv", index=False)
    return counts, manifest


def safe_open_image(path: str | Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return Image.new("RGB", (180, 180), color=(230, 230, 230))


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def make_contact_sheet(frame: pd.DataFrame, output_path: Path, title: str, n: int = 24) -> None:
    selected = frame.head(n).copy()
    if selected.empty:
        return
    thumb = 180
    label_h = 42
    cols = 6
    rows = math.ceil(len(selected) / cols)
    title_h = 44
    canvas = Image.new("RGB", (cols * thumb, title_h + rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (8, 12), title)
    for idx, (_, row) in enumerate(selected.iterrows()):
        col = idx % cols
        line = idx // cols
        x = col * thumb
        y = title_h + line * (thumb + label_h)
        image = safe_open_image(row["image_path"])
        image = ImageOps.fit(image, (thumb, thumb), method=Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y + label_h))
        label = f"{row.get('label', '')} | {str(row.get('name', ''))[:22]}"
        draw_label(draw, (x + 4, y + 6), label)
    canvas.save(output_path, quality=92)


def export_taxonomy_examples(photos: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    prompt_map = {name: item["prompt"] for name, item in TAXONOMY_PROMPTS.items()}
    taxonomy_scores = score_prompts(embeddings, prompt_map, "taxonomy_prompt_scores.npy")
    rows = []
    for category, config in TAXONOMY_PROMPTS.items():
        work = photos.copy()
        work["taxonomy_score"] = taxonomy_scores[category].to_numpy()
        preferred = work[work["label"].isin(config["prefer_labels"])].copy()
        if len(preferred) < 24:
            preferred = work
        selected = (
            preferred[preferred["image_exists"]]
            .sort_values("taxonomy_score", ascending=False)
            .drop_duplicates("photo_id")
            .head(24)
            .copy()
        )
        sheet_name = f"taxonomy_examples_{category}.jpg"
        make_contact_sheet(selected, SHEETS_DIR / sheet_name, category.replace("_", " ").title())
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            rows.append(
                {
                    "taxonomy_category": category,
                    "example_rank": rank,
                    "photo_id": row["photo_id"],
                    "business_id": row["business_id"],
                    "name": row["name"],
                    "city": row["city"],
                    "state": row["state"],
                    "label": row["label"],
                    "taxonomy_score": row["taxonomy_score"],
                    "image_path": row["image_path"],
                    "sheet_path": str(SHEETS_DIR / sheet_name),
                }
            )
    examples = pd.DataFrame(rows)
    examples.to_csv(OUT_DIR / "taxonomy_example_manifest.csv", index=False)
    return examples


def export_occluder_environment_scores(photos: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_scores = score_prompts(embeddings, OCCLUDER_ENVIRONMENT_PROMPTS, "occluder_environment_prompt_scores.npy")
    metadata_cols = ["photo_id", "business_id", "name", "city", "state", "label", "caption", "image_path", "business_group"]
    scored = pd.concat([photos[metadata_cols].reset_index(drop=True), prompt_scores], axis=1)
    scored.to_csv(OUT_DIR / "occluder_environment_prompt_scores.csv", index=False)

    inside_restaurant = scored[
        photos["business_group"].eq("restaurants").to_numpy() & photos["label"].eq("inside").to_numpy()
    ].copy()
    agg_spec = {col: ["mean", "max"] for col in OCCLUDER_ENVIRONMENT_PROMPTS}
    summary = inside_restaurant.groupby(["business_id", "name", "city", "state"], dropna=False).agg(agg_spec)
    summary.columns = [f"{prompt}_{stat}" for prompt, stat in summary.columns]
    summary = summary.reset_index()
    count_df = inside_restaurant.groupby("business_id").size().reset_index(name="inside_photo_count")
    summary = summary.merge(count_df, on="business_id", how="left")
    summary = summary.sort_values(["inside_photo_count", "people_crowd_mean"], ascending=[False, False])
    summary.to_csv(OUT_DIR / "occluder_environment_summary_by_business.csv", index=False)
    return scored, summary


def write_meeting_notes(
    counts: pd.DataFrame,
    manifest: pd.DataFrame,
    examples: pd.DataFrame,
    occluder_summary: pd.DataFrame,
) -> None:
    n_40 = int((counts["inside_photo_count"] >= 40).sum())
    n_45 = int((counts["inside_photo_count"] >= 45).sum())
    n_50 = int((counts["inside_photo_count"] >= 50).sum())
    top_candidates = counts[counts["inside_photo_count"] >= 40].head(10)
    top_lines = "\n".join(
        f"- {row['name']} ({row['city']}, {row['state']}): {int(row['inside_photo_count'])} inside photos"
        for _, row in top_candidates.iterrows()
    )
    category_counts = examples["taxonomy_category"].value_counts().sort_index()
    category_lines = "\n".join(f"- {category}: {count} examples" for category, count in category_counts.items())
    top_occluder = occluder_summary.head(10)
    occluder_lines = "\n".join(
        f"- {row['name']} ({row['city']}, {row['state']}): {int(row['inside_photo_count'])} inside photos"
        for _, row in top_occluder.iterrows()
    )

    notes = f"""# PhD Follow-Up: Interior Candidates, Taxonomy Examples, And Occluder Signals

## What Was Added

This follow-up uses the existing Yelp outputs and cached CLIP image embeddings. It does not recompute CLIP image embeddings. It adds:

- DA3-ready restaurant interior candidate lists.
- A per-image manifest for restaurants with at least 40 Yelp `inside` photos.
- Taxonomy example sheets for camera/framing categories.
- Weak CLIP prompt scores for occluders and environment-state cues.
- A business-level summary of likely occluder/environment signals.

## Important Metadata Clarification

`inside` is Yelp-provided photo metadata from `photos.json`; it was not manually created. The `business_group` field, such as `restaurants`, is our broad grouping derived from Yelp business categories.

## DA3 Candidate Counts

- Restaurants with >=40 Yelp `inside` photos: **{n_40}**
- Restaurants with >=45 Yelp `inside` photos: **{n_45}**
- Restaurants with >=50 Yelp `inside` photos: **{n_50}**

The main DA3 candidate threshold is >=40 interior shots, giving enough repeated images per place while preserving a larger candidate pool.

## Top DA3 Restaurant Candidates

{top_lines}

## Taxonomy Example Sheets

The follow-up generated example sheets for each taxonomy category:

{category_lines}

These sheets are meant for manual inspection. They help verify whether the current taxonomy categories correspond to visually meaningful capture behavior.

## Occluder / Environment-State Signals

The occluder/environment scores are weak CLIP prompt scores, not ground-truth labels. They help prioritize which businesses/images to inspect for:

- people or crowds
- furniture and seating
- tabletop objects or clutter
- counters, bars, or kitchens
- signage, menus, or text
- exterior entrances or thresholds
- low light, blur, or camera noise
- partial or blocked views

Top restaurants by interior-photo availability for follow-up occluder/environment inspection:

{occluder_lines}

## How This Connects To The Simulator

This step moves the analysis from broad image clusters toward simulator-relevant visual conditions. For DA3/manual inspection, the key question is whether repeated interior photos of the same restaurant reveal stable layout, furniture arrangement, changing tabletop objects, people/crowd occlusion, lighting changes, and different human camera viewpoints.

## Files Produced

- `restaurant_interior_counts.csv`
- `restaurant_interior_40plus_candidates.csv`
- `da3_candidate_image_manifest.csv`
- `taxonomy_example_manifest.csv`
- `occluder_environment_prompt_scores.csv`
- `occluder_environment_summary_by_business.csv`
- `taxonomy_example_sheets/taxonomy_examples_*.jpg`
"""
    (OUT_DIR / "phd_followup_meeting_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    photos, embeddings = load_inputs()
    counts, manifest = export_restaurant_interior_candidates(photos)
    examples = export_taxonomy_examples(photos, embeddings)
    _, occluder_summary = export_occluder_environment_scores(photos, embeddings)
    write_meeting_notes(counts, manifest, examples, occluder_summary)

    n_40 = int((counts["inside_photo_count"] >= 40).sum())
    n_45 = int((counts["inside_photo_count"] >= 45).sum())
    n_50 = int((counts["inside_photo_count"] >= 50).sum())
    missing_images = int((~manifest["image_exists"]).sum())
    print(f"Wrote follow-up outputs to: {OUT_DIR}")
    print(f"Restaurants with >=40 inside photos: {n_40}")
    print(f"Restaurants with >=45 inside photos: {n_45}")
    print(f"Restaurants with >=50 inside photos: {n_50}")
    print(f"DA3 manifest rows: {len(manifest)}")
    print(f"Missing image paths in DA3 manifest: {missing_images}")
    print(f"Taxonomy examples: {len(examples)}")


if __name__ == "__main__":
    main()
