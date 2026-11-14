import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


def _tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


class CloudTrailDownloadTest(unittest.TestCase):
    """CloudTrail comes from the flAWS archive, not Hugging Face; a file:// URL
    stands in for the real one."""

    def setUp(self):
        from scripts import download_dataset

        self.dl = download_dataset
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.files = {"flaws_cloudtrail00.json.gz": b"zero", "flaws_cloudtrail01.json.gz": b"one"}
        self.expected = {n: hashlib.sha256(d).hexdigest() for n, d in self.files.items()}

    def test_extracts_only_expected_files_by_base_name(self):
        archive = self.tmp / "flaws.tar"
        _tar(archive, {f"flaws_cloudtrail_logs/{n}": d for n, d in self.files.items()} | {"../evil.txt": b"x"})
        dest = self.tmp / "raw"
        self.dl.download_cloudtrail(dest=str(dest), url=archive.as_uri(), expected=self.expected)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), sorted(self.files))
        self.assertFalse((self.tmp / "evil.txt").exists())

    def test_a_wrong_checksum_stops_and_writes_nothing(self):
        archive = self.tmp / "flaws.tar"
        _tar(archive, {"flaws_cloudtrail_logs/flaws_cloudtrail00.json.gz": b"tampered",
                       "flaws_cloudtrail_logs/flaws_cloudtrail01.json.gz": b"one"})
        dest = self.tmp / "raw"
        with self.assertRaises(SystemExit):
            self.dl.download_cloudtrail(dest=str(dest), url=archive.as_uri(), expected=self.expected)
        self.assertEqual(list(dest.iterdir()), [])

    def test_skips_the_download_when_every_file_is_already_verified(self):
        dest = self.tmp / "raw"
        dest.mkdir()
        for name, data in self.files.items():
            (dest / name).write_bytes(data)
        self.dl.download_cloudtrail(dest=str(dest), url="file:///does/not/exist.tar", expected=self.expected)

    def test_cloudtrail_is_not_fetched_from_hugging_face(self):
        self.assertNotIn("cloudtrail", self.dl.FOLDER_SOURCES)
        self.assertEqual(len(self.dl.FLAWS_SHA256), 20)
