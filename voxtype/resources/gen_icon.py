import struct, zlib, math, os

W, H = 64, 64

def rgba(r, g, b, a=255):
    return bytes([r, g, b, a])

TRANSPARENT = rgba(0, 0, 0, 0)
WHITE = rgba(255, 255, 255, 255)
WHITE60 = rgba(255, 255, 255, 153)
WHITE30 = rgba(255, 255, 255, 77)

def lerp(a, b, t):
    return int(a * (1 - t) + b * t)

def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def in_rounded_rect(x, y, x0, y0, x1, y1, r):
    if x < x0 or x > x1 or y < y0 or y > y1:
        return False
    for cx, cy in [(x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)]:
        dx, dy = x - cx, y - cy
        in_zone = (dx < 0 if cx < (x0 + x1) / 2 else dx > 0) and (dy < 0 if cy < (y0 + y1) / 2 else dy > 0)
        if in_zone and dist(x, y, cx, cy) > r:
            return False
    return True

raw = b''
for y in range(H):
    raw += b'\x00'
    for x in range(W):
        bg = TRANSPARENT

        # Microphone capsule
        mc_x, mc_y = W / 2, H / 2 - 6
        mc_rx, mc_ry = 7, 10
        if ((x - mc_x) / mc_rx) ** 2 + ((y - mc_y) / mc_ry) ** 2 <= 1.0:
            raw += WHITE
            continue

        # Mic arc
        arc_cy = H / 2
        d = dist(x, y, W / 2, arc_cy)
        if 11 < d < 15 and y > arc_cy:
            raw += WHITE
            continue

        # Mic stand
        if abs(x - W / 2) < 2.0 and H / 2 + 13 < y < H / 2 + 19:
            raw += WHITE
            continue

        # Base line
        if abs(y - (H / 2 + 19)) < 2.0 and abs(x - W / 2) < 7:
            raw += WHITE
            continue

        # Sound wave small (right side)
        dw1 = dist(x, y, W / 2 + 9, H / 2 - 4)
        if abs(dw1 - 10) < 2.0 and x > W / 2 + 9:
            raw += WHITE60
            continue

        # Sound wave large (right side)
        dw2 = dist(x, y, W / 2 + 9, H / 2 - 4)
        if abs(dw2 - 16) < 2.0 and x > W / 2 + 9:
            raw += WHITE30
            continue

        raw += bg

def chunk(ctype, data):
    c = ctype + data
    return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

out = os.path.join(os.path.dirname(__file__), 'icon.png')
with open(out, 'wb') as f:
    f.write(b'\x89PNG\r\n\x1a\n')
    f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 6, 0, 0, 0)))
    f.write(chunk(b'IDAT', zlib.compress(raw, 9)))
    f.write(chunk(b'IEND', b''))
print(f'Created {out}')
