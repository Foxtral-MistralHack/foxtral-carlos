#!/usr/bin/env python3
"""
Tiles grass blocks from blocks.png into a background image.
Run from project root:
  tools/.venv/bin/python3 tools/make_background.py
"""

from PIL import Image
import os

src = Image.open("assets/backgrounds/blocks.png")
TILE_W, TILE_H = 36, 30

# Use first grass tile (top-left)
grass = src.crop((0, 0, TILE_W, TILE_H))

# Isometric tiling: offset every other row by half a tile width
COLS = 20
ROWS = 16
OUT_W = COLS * TILE_W
OUT_H = ROWS * TILE_H + TILE_H // 2

bg = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))  # transparent

for row in range(ROWS):
    for col in range(COLS):
        x = col * TILE_W + (TILE_W // 2 if row % 2 == 1 else 0)
        y = row * 14
        bg.paste(grass, (x, y), grass)

os.makedirs("assets/backgrounds", exist_ok=True)
bg.save("assets/backgrounds/background.png")
print(f"Saved background.png ({bg.size})")
