"""Perceptual-hash near-duplicate detection for the aesthetic pass.

An optional dedupe step: bursts, bracketed frames, and minor re-crops of the
same shot are visually near-identical. When enabled, images are clustered by a
difference hash (dHash) and only the highest aesthetic score in each cluster is
kept — the rest are set aside as duplicates.

Uses only core dependencies (Pillow + numpy); no extra install is needed, so
dedupe works even without the optional 'aesthetic' extra installed.
"""

import numpy as np
from PIL import Image

from .config import DEFAULT_HAMMING_THRESHOLD, DEFAULT_HASH_SIZE


def dhash(image, hash_size=DEFAULT_HASH_SIZE):
    """Return the difference-hash of a PIL image as an int (``hash_size**2`` bits).

    Resizes to ``(hash_size + 1) x hash_size`` grayscale, then encodes, per row,
    whether each pixel is brighter than its right-hand neighbour. Encoding the
    *gradient* rather than absolute brightness makes it robust to scaling, mild
    JPEG compression, and small exposure/tonal shifts — while still separating
    genuinely different compositions."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)  # shape (hash_size, hash_size + 1)
    diff = pixels[:, 1:] > pixels[:, :-1]      # each pixel vs its right neighbour
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def format_hash(bits, hash_size=DEFAULT_HASH_SIZE):
    """Render an integer hash as a stable ``0x``-prefixed, zero-padded hex string
    for the CSV log. The prefix keeps pandas from ever coercing an all-digit hex
    value to a number (a 64-bit hash overflows int64 and would round-trip as a
    lossy float otherwise)."""
    hex_digits = (hash_size * hash_size + 3) // 4
    return f"0x{bits:0{hex_digits}x}"


def parse_hash(text):
    """Inverse of :func:`format_hash`: hex string -> int (accepts a bare or
    ``0x``-prefixed value)."""
    return int(text, 16)


def hamming_distance(a, b):
    """Number of differing bits between two integer hashes."""
    return bin(a ^ b).count("1")


def cluster_by_hash(items, threshold=DEFAULT_HAMMING_THRESHOLD):
    """Group ``items`` whose hashes fall within ``threshold`` Hamming distance.

    Each item must be a mapping carrying a ``"hash"`` key. Uses union-find over
    all pairs (O(n^2) in the hash comparison, which is comfortably fine for the
    hundreds-to-few-thousand frames a single shoot produces). Returns a list of
    clusters, each a list of the original items; every item belongs to exactly
    one cluster (singletons included), and input order is preserved within a
    cluster."""
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(items[i]["hash"], items[j]["hash"]) <= threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(items[i])
    return list(clusters.values())
