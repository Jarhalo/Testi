"""Luo Northern Exposure -jaksokuvake: pieni jaksokyltti logokyltin alle.

Käyttö:  python3 make_episode_icon.py <pohjakuva> <jakso> <tulos.png>

E- ja P-kirjaimet kopioidaan suoraan logon sanasta "EXPOSURE", jotta
kirjaintyyli täsmää. Numerot piirretään ja niiden reunat karhennetaan
logon siveltimenjäljen tyyliin.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Logokyltin mitat ja sävyt alkuperäisestä kuvasta
PAPER = 231.0        # kyltin pohjan harmaa
INK = 45.0           # tekstin ja reunuksen tumma sävy
LOGO_ROW = (424, 464)                     # "EXPOSURE"-rivin y-alue
GLYPHS = {"E": (351, 374), "P": (410, 436)}  # kirjainten x-alueet
POLE_X = 465         # oikean tolpan keskikohta

# Uuden kyltin paikka (peittää vanhan laatikon 393-557 x 511-575)
SX0, SY0, SX1, SY1 = 380, 509, 551, 580
OLD_BOX = (388, 506, 562, 580)


def glyph_alpha(src_l, name):
    x0, x1 = GLYPHS[name]
    y0, y1 = LOGO_ROW
    g = src_l[y0:y1, x0:x1].astype(float)
    return np.clip((PAPER - 8 - g) / (PAPER - 8 - INK), 0, 1)


def digit_alpha(text, height, stroke_h):
    """Numero, jonka korkeus vastaa logon kirjaimia ja reunat ovat röpelöiset."""
    scale = 4
    font = ImageFont.truetype(FONT, int(height * scale * 1.38))
    l, t, r, b = font.getbbox(text)
    img = Image.new("L", (r - l + 40, b - t + 40), 0)
    ImageDraw.Draw(img).text((20 - l, 20 - t), text, font=font, fill=255)
    # hieman kapeampi, kuten logon kirjaimet
    img = img.resize((int(img.width * 0.86), img.height), Image.LANCZOS)
    m = np.asarray(img, float) / 255
    rng = np.random.default_rng(len(text) * 7 + ord(text[0]))
    noise = rng.normal(0, 1, m.shape)
    noise = np.asarray(Image.fromarray(((noise + 4) * 30).clip(0, 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(6)), float) / 30 - 4
    blurred = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                         .filter(ImageFilter.GaussianBlur(4)), float) / 255
    rough = ((blurred + noise * 0.5) > 0.6).astype(np.uint8) * 255
    out = Image.fromarray(rough).filter(ImageFilter.GaussianBlur(1.5))
    out = out.crop(out.getbbox())
    h = stroke_h
    out = out.resize((max(1, round(out.width * h / out.height)), h), Image.LANCZOS)
    return np.asarray(out, float) / 255


def build(src_path, episode, out_path):
    src = Image.open(src_path).convert("RGB")
    rgb = np.asarray(src, float).copy()
    L = np.asarray(src.convert("L"))

    # 1) Poista vanha laatikko: täytä peilaamalla ympäröivää taustaa.
    bx0, by0, bx1, by1 = OLD_BOX
    for y in range(by0, by1):
        for x in range(bx0, bx1):
            if x >= SX1 - 2:  # oikea reuna jää näkyviin -> peilaa oikealta
                rgb[y, x] = rgb[y, bx1 + (bx1 - x)]

    # 2) Kyltin pohja: harmahtava valkoinen, kevyt liukuma ja kohina.
    W, H = SX1 - SX0, SY1 - SY0
    rng = np.random.default_rng(1)
    yy = np.linspace(0, 1, H)[:, None]
    base = PAPER + 3 - 6 * yy + rng.normal(0, 2.5, (H, W))
    sign = np.repeat(base[:, :, None], 3, 2)
    sign[..., 2] += 1.0  # logokyltti on aavistuksen sinertävä

    # Reunus: ohut tumma, pyöristetyt kulmat kuten logokyltissä
    scale = 4
    mask = Image.new("L", (W * scale, H * scale), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, W * scale - 1, H * scale - 1), radius=9 * scale, fill=255)
    border = Image.new("L", mask.size, 0)
    bd = ImageDraw.Draw(border)
    bd.rounded_rectangle((3 * scale, 3 * scale, W * scale - 1 - 3 * scale, H * scale - 1 - 3 * scale),
                         radius=7 * scale, outline=255, width=int(2.6 * scale))
    mask = np.asarray(mask.resize((W, H), Image.LANCZOS), float) / 255
    border = np.asarray(border.resize((W, H), Image.LANCZOS), float) / 255

    ink = np.zeros((H, W))
    ink = np.maximum(ink, border * 0.95)

    # Pultit tolpan kohdalla (haaleat harmaat, kuten logokyltissä)
    px = POLE_X - SX0
    for cy in (8, H - 9):
        yy2, xx2 = np.ogrid[:H, :W]
        dot = np.clip(2.2 - np.hypot(xx2 - px, yy2 - cy), 0, 1)
        sign -= dot[..., None] * 70

    # 3) Teksti: E ja P logosta + numero
    parts = [glyph_alpha(L, "E"), glyph_alpha(L, "P")]
    cap_h = parts[0].shape[0]
    r = (parts[0].sum(1) > 0.3).nonzero()[0]
    letter_h = int(r[-1] - r[0]) + 1
    num = digit_alpha(str(episode), letter_h, letter_h)
    gap_letters, gap_word = -2, 9
    total = parts[0].shape[1] + gap_letters + parts[1].shape[1] + gap_word + num.shape[1]
    x = (W - total) // 2
    top = (H - cap_h) // 2 + 1
    for i, g in enumerate(parts):
        h, w = g.shape
        ink[top:top + h, x:x + w] = np.maximum(ink[top:top + h, x:x + w], g)
        x += w + (gap_letters if i == 0 else gap_word)
    # numero samalle perusviivalle kuin logon kirjaimet
    rows = (parts[0].sum(1) > 0.3).nonzero()[0]
    ny = top + rows[0]
    h, w = num.shape
    ink[ny:ny + h, x:x + w] = np.maximum(ink[ny:ny + h, x:x + w], num)

    sign = sign * (1 - ink[..., None]) + INK * ink[..., None]
    sign = np.clip(sign, 0, 255)

    # Pieni skannauspehmeys kuten muussa kuvassa
    sign_img = Image.fromarray(sign.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.45))
    sign = np.asarray(sign_img, float)

    region = rgb[SY0:SY1, SX0:SX1]
    rgb[SY0:SY1, SX0:SX1] = region * (1 - mask[..., None]) + sign * mask[..., None]

    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]), sys.argv[3])
