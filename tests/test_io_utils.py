"""File discovery: case-insensitive, covers RAW + JPEG/PNG, non-recursive."""

from winnow.io_utils import find_images, find_raws


def _touch(directory, *names):
    for name in names:
        (directory / name).write_bytes(b"")


def test_find_images_covers_raw_and_standard_case_insensitively(tmp_path):
    _touch(
        tmp_path,
        "a.CR3", "b.cr3", "c.NEF",       # RAW, mixed case
        "d.jpg", "e.JPEG", "f.png", "g.PNG",  # standard, mixed case
        "notes.txt", "clip.mp4",         # ignored
    )
    # keepers/ subdir must not be descended into
    sub = tmp_path / "keepers"
    sub.mkdir()
    _touch(sub, "already.CR3")

    names = {p.name for p in find_images(tmp_path)}
    assert names == {"a.CR3", "b.cr3", "c.NEF", "d.jpg", "e.JPEG", "f.png", "g.PNG"}


def test_find_raws_matches_only_raw_extensions(tmp_path):
    _touch(tmp_path, "a.CR3", "b.cr3", "c.dng", "d.jpg", "e.png")

    names = {p.name for p in find_raws(tmp_path)}
    assert names == {"a.CR3", "b.cr3", "c.dng"}
