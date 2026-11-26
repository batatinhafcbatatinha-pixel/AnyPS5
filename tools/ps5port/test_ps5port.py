"""Checks of tools/ps5port/ps5port.py that need no game and no network (run: python -m unittest tools/ps5port/test_ps5port.py)."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('ps5port', HERE / 'ps5port.py')
ps5port = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps5port)


def elf64_with_load(vaddr, data):
    """A minimal ELF64 file: one PT_LOAD segment of `data` at virtual address `vaddr`."""
    header = bytearray(0x40)
    header[:4] = b'\x7fELF'
    struct.pack_into('<Q', header, 0x20, 0x40)        # e_phoff
    struct.pack_into('<HH', header, 0x36, 56, 1)      # e_phentsize, e_phnum
    phdr = struct.pack('<IIQQQQQQ', 1, 5, 0x78, vaddr, vaddr, len(data), len(data), 0x1000)
    return bytes(header) + phdr + bytes(0x78 - 0x40 - 56) + data


class Ps5PortTests(unittest.TestCase):
    def test_pinned_runtime_assets(self):
        for system in ('windows', 'linux'):
            for kind in ('relinker', 'libs'):
                name, digest = ps5port.RUNTIME[system][kind]
                self.assertEqual(len(digest), 64, (system, kind))
                int(digest, 16)
                self.assertIn(ps5port.RUNTIME['version'], name)
        self.assertTrue(ps5port.RUNTIME['linux']['libs'][0].endswith('.tar.gz'))
        self.assertFalse(ps5port.RUNTIME['linux']['relinker'][0].endswith('.exe'))

    def test_settings_env_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            port = Path(tmp)
            (port / 'pc-settings.ini').write_text('modo=1\nvsync=1\nvolume=50\nlanguage=17\n')
            env = ps5port.settings_env(port, None)
        self.assertEqual(env['ANYPS5_FULLSCREEN'], '1')
        self.assertEqual(env['ANYPS5_PRESENT_MODE'], 'mailbox')
        self.assertEqual(env['ANYPS5_VOLUME'], '50')
        self.assertEqual(env['ANYPS5_LANGUAGE'], '17')

    def test_system_language_from_environment(self):
        with mock.patch.object(ps5port, 'WINDOWS', False):
            for value, expected in (('pt_BR.UTF-8', 17), ('en_GB.UTF-8', 18), ('de_DE@euro', 4), ('ja_JP', 0), ('C', None)):
                with mock.patch.dict(os.environ, {'LC_ALL': value, 'LC_MESSAGES': '', 'LANGUAGE': '', 'LANG': ''}):
                    self.assertEqual(ps5port.system_language(), expected, value)

    def test_target_flags(self):
        with mock.patch.object(ps5port, 'WINDOWS', True):
            self.assertEqual(ps5port.target_flags(), ['--windows'])
        with mock.patch.object(ps5port, 'WINDOWS', False), mock.patch.object(ps5port, 'MACOS', False):
            with mock.patch.object(ps5port, 'cpu_vendor', return_value='GenuineIntel'):
                self.assertEqual(ps5port.target_flags(), ['--to-intel'])
            with mock.patch.object(ps5port, 'cpu_vendor', return_value='AuthenticAMD'):
                self.assertEqual(ps5port.target_flags(), [])

    def test_elf_segments_and_image_write(self):
        data = bytes(range(16)) * 4
        with tempfile.TemporaryDirectory() as tmp:
            port = Path(tmp)
            out = port / 'build' / 'out'
            out.mkdir(parents=True)
            exe = out / ps5port.PROGRAM
            image = elf64_with_load(ps5port.LINUX_IMAGE_BASE + 0x1000, data)
            exe.write_bytes(image)
            segments = ps5port.elf_segments(exe)
            self.assertEqual(segments, [(ps5port.LINUX_IMAGE_BASE + 0x1000, len(data), 0x78, len(data))])
            offset = 0x1000 + 8 + ps5port.GOLDHEN_MAIN_BASE
            write = {'module': 'eboot.bin', 'offset': hex(offset), 'find': data[8:12].hex(), 'write': '00000000'}
            with mock.patch.object(ps5port, 'WINDOWS', False), mock.patch.object(ps5port, 'MACOS', False):
                target, where, new = ps5port.image_write(port, write)
                self.assertEqual((target, where, new), (exe, 0x78 + 8, bytes(4)))
                write['find'] = 'ffffffff'
                with self.assertRaises(ValueError):
                    ps5port.image_write(port, write)
                self.assertEqual(ps5port.memory_write(port, {**write, 'find': data[8:12].hex()}).split(':')[0], '1008')

    def test_remove_tree_keeps_link_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dump = tmp / 'dump'
            dump.mkdir()
            (dump / 'file.bin').write_bytes(b'x')
            link = tmp / 'link'
            try:
                ps5port.junction(dump, link)
            except OSError:
                self.skipTest('links are not allowed here')
            ps5port.remove_tree(link)
            self.assertFalse(link.exists())
            self.assertTrue((dump / 'file.bin').is_file())

    def test_launcher_json_blocks(self):
        config = json.loads((ROOT / 'launcher.json').read_text(encoding='utf-8'))
        for system, exe in (('linux', 'build/out/app.elf'), ('macos', 'build/out/app')):
            block = config[system]
            self.assertEqual(block['exe'], exe)
            self.assertEqual(block['generic']['exe'], exe)
            self.assertEqual(block['generic']['steps'][0]['done_if'], exe)
            self.assertTrue(block['generic']['play_sh'].startswith('#!/usr/bin/env bash\n'))
            self.assertNotIn('\r', block['generic']['play_sh'])
            self.assertIn('ps5port.py', ' '.join(block['generic']['steps'][0]['run']))
        self.assertEqual(config['exe'], 'build/out/app.exe')
        self.assertTrue(config['linux']['packages']['apt-get'])


if __name__ == '__main__':
    unittest.main()
