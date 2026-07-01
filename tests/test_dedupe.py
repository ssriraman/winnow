"""Perceptual-hash dedupe: stable hashing, tolerant matching, correct clustering."""

from PIL import Image

from winnow.dedupe import cluster_by_hash, dhash, format_hash, hamming_distance, parse_hash


def _scene(width=64, height=64, shift=0, seed=1):
    """A textured, non-monotonic image so the hash carries real bits. ``shift``
    applies a small brightness nudge (near-duplicate); ``seed`` changes the
    composition (different scene)."""
    img = Image.new("RGB", (width, height))
    px = img.load()
    for x in range(width):
        for y in range(height):
            v = (x * 7 + y * 13 + (x % (seed + 2)) * 40 + (y % 3) * 30) % 256
            v = min(255, max(0, v + shift))
            px[x, y] = (v, v, v)
    return img


def test_dhash_is_deterministic_and_64_bit():
    img = _scene()
    h = dhash(img)
    assert h == dhash(img)  # stable
    assert 0 <= h < (1 << 64)  # default hash_size=8 -> 64 bits


def test_identical_images_have_zero_distance():
    assert hamming_distance(dhash(_scene()), dhash(_scene())) == 0


def test_near_duplicate_is_close_and_different_scene_is_far():
    base = dhash(_scene(seed=1))
    nudged = dhash(_scene(seed=1, shift=8))  # slight exposure change, same composition
    different = dhash(_scene(seed=5))         # different composition

    assert hamming_distance(base, nudged) <= 5
    assert hamming_distance(base, different) > 5


def test_format_hash_is_prefixed_zero_padded_and_round_trips():
    # 64-bit hash -> 16 hex digits, '0x' prefixed, low value stays zero-padded.
    assert format_hash(0) == "0x0000000000000000"
    assert format_hash(0xABC) == "0x0000000000000abc"
    for value in (0, 1, 0xDEADBEEF, (1 << 64) - 1):  # incl. > int64 max
        assert parse_hash(format_hash(value)) == value


def test_format_hash_survives_csv_round_trip_as_string():
    import pandas as pd

    big = (1 << 64) - 1  # would corrupt if stored as an int64/float
    df = pd.DataFrame({"hash": [format_hash(big)]})
    text = df.to_csv(index=False)
    restored = pd.read_csv(pd.io.common.StringIO(text))
    assert parse_hash(restored["hash"].iloc[0]) == big


def test_cluster_groups_near_duplicates_and_isolates_distinct():
    items = [
        {"id": "a", "hash": 0b0000},
        {"id": "b", "hash": 0b0001},  # 1 bit from a
        {"id": "c", "hash": 0b1111},  # far from a/b
    ]
    clusters = cluster_by_hash(items, threshold=1)
    ids = sorted(sorted(item["id"] for item in c) for c in clusters)
    assert ids == [["a", "b"], ["c"]]


def test_cluster_is_transitive_via_union_find():
    # a~b and b~c (each 1 bit apart) but a and c are 2 bits apart: still one cluster.
    items = [
        {"id": "a", "hash": 0b000},
        {"id": "b", "hash": 0b001},
        {"id": "c", "hash": 0b011},
    ]
    clusters = cluster_by_hash(items, threshold=1)
    assert len(clusters) == 1
    assert sorted(item["id"] for item in clusters[0]) == ["a", "b", "c"]


def test_every_item_appears_in_exactly_one_cluster():
    items = [{"id": i, "hash": i} for i in range(5)]
    clusters = cluster_by_hash(items, threshold=0)  # nothing merges
    assert len(clusters) == 5
    assert sorted(item["id"] for c in clusters for item in c) == [0, 1, 2, 3, 4]
