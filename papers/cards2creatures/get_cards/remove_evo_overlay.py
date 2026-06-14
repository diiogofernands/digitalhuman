from PIL import Image
import json


def make_map(image_path, color="cf00ff"):
    target_rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        pixels = []
        for y in range(height):
            for x in range(width):
                if img.getpixel((x, y)) == target_rgb:
                    pixels.append([x, y])

        with open("pixels.json", "w") as f:
            json.dump(pixels, f)
        return pixels


def remove_pixels(image_path, json_path):
    with open(json_path, "r") as f:
        pixels = json.load(f)

    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        for x, y in pixels:
            current = img.getpixel((x, y))
            img.putpixel((x, y), (current[0], current[1], current[2], 0))
        img.save(f"{image_path.rsplit('.', 1)[0]}_transparent.png", "PNG")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit("Usage: python remove_evo_overlay.py <image.png> <pixels.json>")
    remove_pixels(sys.argv[1], sys.argv[2])
