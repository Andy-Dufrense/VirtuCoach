"""生成 VirtuCoach PWA 图标（幂等，可重复运行）。

输出: frontend/icons/icon-192.png, icon-512.png, apple-touch-icon.png
需要 Pillow，无网络依赖。
"""
import os
from PIL import Image, ImageDraw


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "icons")


def _gradient(size, top=(108, 92, 231), bottom=(72, 52, 212)):
    img = Image.new("RGB", size)
    px = img.load()
    r0, g0, b0 = top
    r1, g1, b1 = bottom
    for y in range(size[1]):
        t = y / max(1, size[1] - 1)
        color = (
            int(r0 + (r1 - r0) * t),
            int(g0 + (g1 - g0) * t),
            int(b0 + (b1 - b0) * t),
        )
        for x in range(size[0]):
            px[x, y] = color
    return img


def _draw_notes(draw, w, h):
    """在中心绘制两个相连的八分音符（白描）。"""
    s = w / 512.0

    def S(v):
        return int(v * s)

    # 音符头（圆点）
    head_r = S(38)
    heads = [(S(168), S(352)), (S(318), S(352))]
    for cx, cy in heads:
        draw.ellipse(
            [cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill="white"
        )

    # 符干
    stem_w = S(22)
    for cx, cy in heads:
        draw.rectangle(
            [cx + S(20), S(132), cx + S(20) + stem_w, cy], fill="white"
        )

    # 符梁（连接两条符干顶部）
    left = S(160)
    right = S(400)
    top = S(132)
    beam_h = S(24)
    draw.polygon(
        [
            (left, top),
            (left + S(34), top),
            (right - S(14), top + beam_h),
            (right - S(48), top + beam_h),
        ],
        fill="white",
    )

    # 音符尾部小钩（第二个音符）
    tail_top = (S(420), S(144))
    tail_bottom = (S(388), S(224))
    draw.line(
        [tail_top, tail_bottom], fill="white", width=S(16), joint="curve"
    )


def _render(size, radius_scale=0.22):
    img = _gradient((size, size))
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    radius = int(size * radius_scale)
    md.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)

    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)

    # 深色底部叠一层，突出音符
    shade = Image.new("RGBA", (size, size), (26, 26, 46, 120))
    out = Image.alpha_composite(out, shade)

    draw = ImageDraw.Draw(out)
    _draw_notes(draw, size, size)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    icon_512 = _render(512)
    icon_512.save(os.path.join(OUT_DIR, "icon-512.png"))
    icon_512.resize((192, 192), Image.LANCZOS).save(
        os.path.join(OUT_DIR, "icon-192.png")
    )
    # Apple touch icon：去圆角，交给系统裁切
    apple = _gradient((180, 180))
    ad = ImageDraw.Draw(apple)
    _draw_notes(ad, 180, 180)
    apple.save(os.path.join(OUT_DIR, "apple-touch-icon.png"))
    print("icons written ->", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
