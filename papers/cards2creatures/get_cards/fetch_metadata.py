import argparse
import asyncio
import json
import os

from tcgdexsdk import TCGdex, Query


async def fetch_filtered_cards(set_id, rarities):
    tcgdx = TCGdex("en")
    card_resumes = await tcgdx.card.list(Query().equal("set.id", set_id))

    filtered_cards = []
    for resume in card_resumes:
        full_card = await tcgdx.card.get(resume.id)
        if full_card.rarity in rarities:
            filtered_cards.append(full_card.__dict__)

    os.makedirs(set_id, exist_ok=True)
    output_path = f"{set_id}/{set_id}_filtered_cards.json"
    with open(output_path, "w") as f:
        json.dump(filtered_cards, f, indent=2, default=str)

    print(f"Saved {len(filtered_cards)} cards to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch filtered card metadata from TCGdex")
    parser.add_argument("--set", default="sv10", help="Set id (default: sv10)")
    parser.add_argument(
        "--rarities",
        nargs="+",
        default=["Common", "Uncommon", "Rare"],
        help="Rarities to include",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(fetch_filtered_cards(args.set, args.rarities))
