"""Discovery-only Windows regressions: no installation, imports or training."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == 'nt', 'Windows batch discovery')
class UVDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='uv discovery ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = os.environ.copy()
        self.env.pop('CORE_PIVOT_UV', None)
        self.env.update(LOCALAPPDATA=str(self.root/'local'),
                        USERPROFILE=str(self.root/'user'),
                        PROGRAMFILES=str(self.root/'programs'),
                        PATH=str(Path(os.environ['SystemRoot'])/'System32'))
        shutil.copyfile(Path(__file__).resolve().parents[1]/'scripts/find_uv.cmd',
                        self.root/'find_uv.cmd')
        # CMD can restore the machine ProgramFiles variable at startup.
        (self.root/'probe.cmd').write_text(
            '@echo off\nset "ProgramFiles='+str(self.root/'programs')+'"\n'
            'call find_uv.cmd\nset "RC=%ERRORLEVEL%"\n'
            'echo FOUND=%UV_EXE%\nexit /b %RC%\n')

    def discover(self, relative=None):
        target = None
        if relative:
            target = self.root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()  # Discovery never executes this dummy file.
        result = subprocess.run([os.environ['ComSpec'], '/d', '/c', 'probe.cmd'],
                                cwd=self.root, env=self.env, capture_output=True,
                                text=True, timeout=15)
        return target, result

    def check_location(self, relative):
        target, result = self.discover(relative)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('FOUND='+str(target), result.stdout)

    def test_winget_link(self):
        self.check_location('local/Microsoft/WinGet/Links/uv.exe')

    def test_winget_package_root_without_link(self):
        self.check_location('local/Microsoft/WinGet/Packages/astral-sh.uv_test/uv.exe')

    def test_winget_nested_package_without_link(self):
        self.check_location('local/Microsoft/WinGet/Packages/astral-sh.uv_test/uv-x86_64-pc-windows-msvc/uv.exe')

    def test_machine_package_without_link(self):
        self.check_location('programs/WinGet/Packages/astral-sh.uv_test/bin/uv.exe')

    def test_standalone_user_install(self):
        self.check_location('user/.local/bin/uv.exe')

    def test_missing_installation(self):
        _, result = self.discover()
        self.assertEqual(result.returncode, 1)

    def test_invalid_override_is_not_silently_ignored(self):
        self.env['CORE_PIVOT_UV'] = str(self.root/'missing.exe')
        _, result = self.discover('local/Microsoft/WinGet/Links/uv.exe')
        self.assertEqual(result.returncode, 2)
        self.assertIn('CORE_PIVOT_UV', result.stderr)


if __name__ == '__main__':
    unittest.main()
