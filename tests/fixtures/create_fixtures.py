import io

from PIL import Image, ImageDraw


def generate_test_image(width: int = 224, height: int = 224, color: tuple = (120, 150, 180)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, width - 20, height - 20], fill=(245, 200, 20), outline=(200, 160, 10))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def generate_corrupt_image() -> bytes:
    return b"NOT_A_VALID_IMAGE_BYTE_STREAM_CORRUPT_HEADER_0xDEADBEEF"


def generate_empty_image() -> bytes:
    return b""
