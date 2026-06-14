import argparse
import asyncio
import json
from pathlib import Path

from tcgdexsdk import TCGdex
from tcgdexsdk.enums import Extension, Quality


async def download_high_quality_images(set_name):
    sdk = TCGdex("en")

    metadata_path = Path(f"{set_name}/{set_name}_filtered_cards.json")
    if not metadata_path.exists():
        print(f"Skipping {set_name}: missing {metadata_path}")
        return

    with open(metadata_path, "r") as f:
        cards_data = json.load(f)

    Path(set_name).mkdir(exist_ok=True)

    for i, card_data in enumerate(cards_data):
        card_id = card_data["id"]
        card_name = card_data["name"]
        print(f"Processing {i + 1}/{len(cards_data)}: {card_name} ({card_id})")

        try:
            card = await sdk.card.get(card_id)
            image_response = card.get_image(Quality.HIGH, Extension.PNG)
            filepath = Path(f"{set_name}/{card_id}.png")
            filepath.write_bytes(image_response.read())
            print(f"Downloaded: {filepath}")
        except Exception as e:
            print(f"Failed: {card_id} - {e}")


async def main(sets):
    for set_name in sets:
        print(f"\n{'=' * 50}\nProcessing set: {set_name}\n{'=' * 50}")
        await download_high_quality_images(set_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download high-res card images from TCGdex")
    parser.add_argument(
        "--sets",
        nargs="+",
        default=[f"sv{i:02d}" for i in range(3, 11)],
        help="Sets to download (default: sv03–sv10)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.sets))
