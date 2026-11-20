"""Cover image: one frame from the finished video + a big title (and an optional second line)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .common import ffmpeg, font_path, load


def make(d, t, title, sub=""):
    d = Path(d)
    proj = load(d, "project.json")
    # t is a time in the finished video; grab the clean picture (no subtitles / cards) from the cut
    t_cut = float(t) * float(proj.get("speed", 1.0))
    frame = d / "cover_frame.png"
    ffmpeg("-ss", f"{t_cut:.3f}", "-i", d / "cut_video.mp4", "-frames:v", 1, frame)
    img = Image.open(frame).convert("RGB")
    vertical = img.height > img.width
    W, H = (1080, 1920) if vertical else (1280, 720)
    img = img.resize((W, int(img.height * W / img.width)), Image.LANCZOS)
    if img.height < H:
        img = img.resize((int(img.width * H / img.height), H), Image.LANCZOS)
    x0, y0 = (img.width - W) // 2, (img.height - H) // 2
    img = img.crop((x0, y0, x0 + W, y0 + H)).convert("RGBA")

    # darken the bottom so the text reads on any picture
    shade = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shade)
    for y in range(H):
        f = max(0.0, (y / H - 0.45) / 0.55)
        sd.line([(0, y), (W, y)], fill=int(190 * f))
    black = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    img = Image.composite(black, img, shade)

    fp = font_path(proj.get("font"))
    base = W if vertical else H
    size = int(base * (0.16 if vertical else 0.17))
    f_main = ImageFont.truetype(fp, size)
    dr = ImageDraw.Draw(img)
    while dr.textlength(title, font=f_main) > W * 0.9 and size > 20:
        size -= 4
        f_main = ImageFont.truetype(fp, size)
    f_sub = ImageFont.truetype(fp, int(size * 0.42))
    y_main = H * (0.70 if vertical else 0.60)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    tw = dr.textlength(title, font=f_main)
    gd.text(((W - tw) / 2, y_main), title, font=f_main, fill=(255, 159, 28, 200))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(size * 0.12)))
    dr = ImageDraw.Draw(img)
    dr.text(((W - tw) / 2, y_main), title, font=f_main, fill=(255, 255, 255, 255),
            stroke_width=max(3, size // 14), stroke_fill=(0, 0, 0, 255))
    if sub:
        sw = dr.textlength(sub, font=f_sub)
        dr.text(((W - sw) / 2, y_main + size * 1.18), sub, font=f_sub, fill=(255, 159, 28, 255),
                stroke_width=max(2, size // 30), stroke_fill=(0, 0, 0, 255))
    out = Path(proj["output_dir"]) / f"{proj['name']}_封面.jpg"
    img.convert("RGB").save(out, quality=92)
    print(f"封面：{out}")
