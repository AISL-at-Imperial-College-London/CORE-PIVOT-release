"""License and packaging checks only; no learner or evaluation is executed."""
import importlib.util
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LicensePackagingTests(unittest.TestCase):
    def test_original_license_names_aisl(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("MIT License\n"))
        self.assertIn(
            "Copyright (c) 2026 Autonomous Industrial Systems Laboratory", text
        )
        self.assertIn('THE SOFTWARE IS PROVIDED "AS IS"', text)

    def test_corl_scope_is_explicit(self):
        notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
        self.assertIn("algorithms/corl_vendor/", notice)
        self.assertIn("ablations/cql_backup.py", notice)
        self.assertIn("remain under Apache-2.0, not the", notice)
        vendor = (ROOT / "algorithms/corl_vendor/LICENSE").read_text(
            encoding="utf-8"
        )
        self.assertIn("Apache License", vendor)
        self.assertIn("Version 2.0, January 2004", vendor)

    def test_distribution_declares_both_licenses(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["project"]["license"], "MIT AND Apache-2.0")
        self.assertEqual(
            set(config["project"]["license-files"]),
            {"LICENSE", "NOTICE.md", "algorithms/corl_vendor/LICENSE",
             "algorithms/corl_vendor/third_party/*_LICENSE.txt"},
        )
        for relative in config["project"]["license-files"]:
            self.assertTrue(list(ROOT.glob(relative)), relative)
        for name, holder in [('TD3_BC', 'Scott Fujimoto'),
                             ('IQL_PyTorch', 'Garrett Thomas'), ('CQL', 'Xinyang Geng')]:
            text = (ROOT/'algorithms/corl_vendor/third_party'/f'{name}_LICENSE.txt').read_text()
            self.assertIn(holder, text)
            self.assertIn('Permission is hereby granted', text)

    def test_authorship_and_private_package_guard_are_preserved(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["project"]["authors"], [{"name": "Christos Evripidou"}])
        self.assertIn("Private :: Do Not Upload", config["project"]["classifiers"])

    def test_source_archive_includes_license_and_notice(self):
        spec = importlib.util.spec_from_file_location(
            "package_release", ROOT / "scripts/package_release.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue({"LICENSE", "NOTICE.md"}.issubset(module.ROOT_FILES))
        self.assertIn("algorithms", module.DIRECTORIES)

    def test_current_docs_do_not_claim_license_is_pending(self):
        for relative in ("README.md", "NOTICE.md", "CONTRIBUTING.md"):
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
            self.assertNotIn("license is pending", text, relative)
            self.assertNotIn("license has not yet", text, relative)
            self.assertIn("mit", text, relative)


if __name__ == "__main__":
    unittest.main()
