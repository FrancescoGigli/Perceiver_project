import errno
import tempfile
import threading
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from strumenti import make_consegna as package


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "repo"
        self.source = self.root / "progetto"
        self.source.mkdir(parents=True)
        self.script = self.source / "experiments.py"
        self.script.write_text("# commento\nEXPERIMENTS = ['e01']\n", encoding="utf-8")
        self.dest = self.base / "consegna"
        self.archive = self.dest.with_suffix(".zip")
        for item in (
            patch.object(package, "ROOT", self.root),
            patch.object(package, "SOURCE", self.source),
            patch.object(package, "check_nothing_is_left_behind"),
            patch.object(package, "tracked_files", return_value=["progetto/experiments.py"]),
        ):
            item.start()
            self.addCleanup(item.stop)

    def assert_no_outputs(self):
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.archive.exists())
        self.assertEqual(list(self.base.glob(".consegna-*")), [])

    def test_contents_and_no_zip_mode(self):
        package.build(self.dest)
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(archive.namelist(), ["experiments.py"])
            actual = (self.dest / "experiments.py").read_bytes()
            self.assertEqual(archive.read("experiments.py"), actual)
            self.assertNotIn(b"# commento", actual)
        other = self.base / "altro"
        other.with_suffix(".zip").write_bytes(b"archivio esistente")
        package.build(other, keep_comments=True, no_zip=True)
        self.assertEqual((other / "experiments.py").read_bytes(), self.script.read_bytes())
        self.assertEqual(other.with_suffix(".zip").read_bytes(), b"archivio esistente")

    def test_existing_paths_and_source(self):
        self.dest.mkdir()
        (self.dest / "appunti").write_bytes(b"preziosi")
        with self.assertRaises(FileExistsError):
            package.build(self.dest)
        self.assertEqual((self.dest / "appunti").read_bytes(), b"preziosi")
        self.archive.write_bytes(b"originale")
        with self.assertRaises(FileExistsError):
            package.build(self.base / "consegna.v2")
        self.assertEqual(self.archive.read_bytes(), b"originale")
        existing_file = self.base / "file"
        existing_file.write_bytes(b"originale")
        with self.assertRaises(FileExistsError):
            package.build(existing_file)
        self.assertEqual(existing_file.read_bytes(), b"originale")
        for target in (self.source, self.root, self.source / "copia", self.root / ".git" / "x",
                       self.base / "solo.zip"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                package.build(target)
        self.assertTrue(self.script.exists())

    def test_failures_before_publication(self):
        with patch.object(package.shutil, "copy2", side_effect=OSError("copia")):
            with self.assertRaisesRegex(OSError, "copia"):
                package.build(self.dest, keep_comments=True)
        self.assert_no_outputs()
        self.script.write_text("def !!!", encoding="utf-8")
        with self.assertRaises(SyntaxError):
            package.build(self.dest, keep_comments=True)
        self.assert_no_outputs()
        self.script.write_text("EXPERIMENTS = []", encoding="utf-8")
        with patch.object(zipfile.ZipFile, "write", side_effect=OSError("zip pieno")):
            with self.assertRaisesRegex(OSError, "zip pieno"):
                package.build(self.dest)
        self.assert_no_outputs()

    def test_late_directory_collision_preserves_even_empty_directory(self):
        publish = package.publish_new

        def compete(source, target):
            if target == self.dest:
                target.mkdir()
            publish(source, target)

        with patch.object(package, "publish_new", side_effect=compete):
            with self.assertRaises(OSError):
                package.build(self.dest)
        self.assertEqual(list(self.dest.iterdir()), [])
        self.assertFalse(self.archive.exists())
        self.assertEqual(list(self.base.glob(".consegna-*")), [])

    def test_late_zip_collision_preserves_the_other_archive(self):
        publish = package.publish_new

        def compete(source, target):
            if target == self.archive:
                target.write_bytes(b"originale")
            publish(source, target)

        with patch.object(package, "publish_new", side_effect=compete):
            with self.assertRaises(OSError):
                package.build(self.dest)
        self.assertFalse(self.dest.exists())
        self.assertEqual(self.archive.read_bytes(), b"originale")

    def test_concurrent_builds_publish_one_complete_pair(self):
        publish = package.publish_new
        both_ready = threading.Barrier(2)

        def start_together(source, target):
            if target == self.archive:
                both_ready.wait(timeout=10)
            publish(source, target)

        def attempt():
            try:
                package.build(self.dest)
            except FileExistsError as exc:
                self.assertEqual(exc.errno, errno.EEXIST)
                return False
            return True

        with patch.object(package, "publish_new", side_effect=start_together):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [pool.submit(attempt) for _ in range(2)]
                self.assertEqual(sorted(job.result() for job in results), [False, True])
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(archive.read("experiments.py"), (self.dest / "experiments.py").read_bytes())
        self.assertEqual(list(self.base.glob(".consegna-*")), [])
