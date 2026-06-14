from PIL import Image
import json
import os
import glob
import argparse
from remove_evo_overlay import remove_pixels


def crop_card_art(image_path, height_offset=435):
    image = Image.open(image_path)
    _, height = image.size
    left = 48
    right = left + 504
    bottom = height - height_offset
    top = bottom - 308
    cropped_image = image.crop((left, top, right, bottom))

    output_path = f"{image_path.rsplit('.', 1)[0]}_cropped_art.png"
    cropped_image.save(output_path, "PNG")
    return output_path


def process_card(card_id, apply_evo_removal=False):
    set_name = card_id.split("-")[0]

    with open(f"./{set_name}/{set_name}_filtered_cards.json", "r") as f:
        cards_data = json.load(f)

    card_data = next((card for card in cards_data if card["id"] == card_id), None)
    if not card_data:
        return

    image_path = f"./{set_name}/{card_id}.png"
    if card_data.get("category") == "Pokemon":
        cropped = crop_card_art(image_path)
        if apply_evo_removal and card_data.get("stage") != "Basic":
            remove_pixels(cropped, "pixels.json")

    if card_data.get("category") == "Trainer":
        crop_card_art(image_path, height_offset=400)


def main(set_dir, apply_evo_removal=False):
    png_files = glob.glob(f"./{set_dir}/{set_dir}-*.png")

    for png_file in png_files:
        card_id = os.path.basename(png_file).rsplit(".", 1)[0]
        process_card(card_id, apply_evo_removal=apply_evo_removal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Pokémon/Trainer art from full card images")
    parser.add_argument(
        "--sets",
        nargs="+",
        default=[f"sv{i:02d}" for i in range(3, 11)],
        help="Sets to process (default: sv03–sv10)",
    )
    parser.add_argument(
        "--evo-removal",
        action="store_true",
        help="Remove evolution-line pixels on non-Basic Pokémon (requires pixels.json)",
    )
    args = parser.parse_args()

    for set_dir in args.sets:
        print(f"Processing {set_dir}")
        main(set_dir, apply_evo_removal=args.evo_removal)
