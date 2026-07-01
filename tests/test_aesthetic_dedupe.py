"""Dedupe selection: the highest-scoring frame of a near-duplicate cluster wins,
and the losers are moved aside. Exercises _resolve_duplicates without torch."""

from winnow.aesthetic import _resolve_duplicates


def _touch(directory, name):
    path = directory / name
    path.write_bytes(b"stub")
    return path


def test_keeps_highest_score_and_moves_duplicates(tmp_path):
    dup_dir = tmp_path / "duplicates"
    # Three near-duplicate frames (same hash) + one distinct image.
    results = [
        {"path": _touch(tmp_path, "burst1.jpg"), "score": 5.0, "hash": 0b0000},
        {"path": _touch(tmp_path, "burst2.jpg"), "score": 7.5, "hash": 0b0000},  # best
        {"path": _touch(tmp_path, "burst3.jpg"), "score": 6.0, "hash": 0b0001},  # 1 bit
        {"path": _touch(tmp_path, "solo.jpg"), "score": 4.0, "hash": 0b1111},
    ]

    survivors = _resolve_duplicates(results, dup_dir, hash_threshold=2)

    survivor_names = {r["path"].name for r in survivors}
    assert survivor_names == {"burst2.jpg", "solo.jpg"}  # cluster winner + the distinct one

    # Losers were physically moved into the duplicates dir, winners were not.
    assert {p.name for p in dup_dir.iterdir()} == {"burst1.jpg", "burst3.jpg"}
    assert (tmp_path / "burst2.jpg").exists()
    assert (tmp_path / "solo.jpg").exists()


def test_no_duplicates_leaves_everything_in_place(tmp_path):
    dup_dir = tmp_path / "duplicates"
    results = [
        {"path": _touch(tmp_path, "a.jpg"), "score": 5.0, "hash": 0b0000},
        {"path": _touch(tmp_path, "b.jpg"), "score": 6.0, "hash": 0b1111},
    ]

    survivors = _resolve_duplicates(results, dup_dir, hash_threshold=1)

    assert {r["path"].name for r in survivors} == {"a.jpg", "b.jpg"}
    assert not dup_dir.exists()  # nothing moved -> dir never created
