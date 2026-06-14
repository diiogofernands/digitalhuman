from PIL import Image
import os
import glob
import re


def get_yolo_annotation_from_crop(left, top, right, bottom, image_width, image_height, class_id=0):
    x_center = (left + right) / 2 / image_width
    y_center = (top + bottom) / 2 / image_height
    width = (right - left) / image_width
    height = (bottom - top) / image_height
    return f"{class_id} {x_center} {y_center} {width} {height}"


def create_card_number_bbox(image_width=600, image_height=825):
    left, right, top, bottom = 98, 165, 781, 799
    return get_yolo_annotation_from_crop(left, top, right, bottom, image_width, image_height, class_id=0)


def create_card_set_bbox(image_width=600, image_height=825):
    left, right, top, bottom = 54, 105, 772, 797
    return get_yolo_annotation_from_crop(left, top, right, bottom, image_width, image_height, class_id=1)


def process_card(card_id):
    set_name = card_id.split("-")[0]
    image_path = f"./{set_name}/{card_id}.png"

    if not os.path.exists(image_path):
        return

    bbox_dir = f"./bbox_{set_name}"
    os.makedirs(bbox_dir, exist_ok=True)

    yolo_content = f"{create_card_number_bbox()}\n{create_card_set_bbox()}\n"
    with open(f"{bbox_dir}/{card_id}.txt", "w") as f:
        f.write(yolo_content)


def main(set_dir):
    png_files = glob.glob(f"./{set_dir}/{set_dir}-*.png")

    for png_file in png_files:
        card_id = os.path.basename(png_file).rsplit(".", 1)[0]
        if re.match(r"^sv\d+-\d+$", card_id):
            process_card(card_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create YOLO labels for card_num and card_set regions")
    parser.add_argument(
        "--sets",
        nargs="+",
        default=[f"sv{i:02d}" for i in range(1, 11)],
        help="Sets to process (default: sv01–sv10)",
    )
    args = parser.parse_args()

    for set_dir in args.sets:
        print(f"Processing {set_dir}")
        main(set_dir)
