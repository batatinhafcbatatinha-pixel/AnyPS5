"""PS5 dumps to Windows programs with the AnyPS5 relinker, for the port launcher (tools/port-installer).

A port folder (<repo>/.local/ports/<title id>/, made by the launcher) holds setup.json, installer.json, Play.cmd and
game_files/ (a junction to the player's dump: nothing of the game is copied). Commands:

  build --port <dir>   unwrap the decrypted SELF containers (eboot.bin and every .prx/.sprx module), relink them for
                       Windows into <port>/build/out (app.exe, app0/ with the converted modules, libs/ = the AnyPS5
                       system libraries) and fill app0/ with links to the dump.
  play --port <dir>    start build/out/app.exe with the PC settings of <port>/pc-settings.ini as AnyPS5 variables.
  env --port <dir>     print the variables play would set (a check of the PC settings without starting the game).

Saves, trophies and screenshots stay in build/out (the game's working folder); a new build keeps them.
The AnyPS5 relinker and system libraries come from an AnyPS5 release, pinned by SHA-256, into <repo>/.local/tools.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / '.local' / 'tools'
sys.path.insert(0, str(REPO / 'tools'))

# The AnyPS5 runtime for Windows: relinker + system libraries of one release.
RUNTIME = {
    'version': 'v0.1.4',
    'base': 'https://github.com/boykopovar/AnyPS5/releases/download/v0.1.4/',
    'relinker': ('relinker-v0.1.4.exe', '6f6d49e3613ea7fce859621265068bfd64161a729a8051c83150f2d45ec193e0'),
    'libs': ('prx-windows-v0.1.4.zip', 'cbd61c521d555614dbcddc62cdabf45acfe409c520307cdb58732c8486dd0d7a'),
}
ELF = b'\x7fELF'
SELF = (b'\x4f\x15\x3d\x1d', b'\x54\x14\xf5\xee')   # fSELF / SELF containers (PS4-style and PS5)
MODULE_EXTS = ('.prx', '.sprx')
STANDARD_MODULE_DIRS = ('sce_module', 'sce_modules', 'prx')
SKIP_DIRS = {'sce_sys'}


def log(text):
    print(text, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target, expected):
    if target.is_file() and sha256(target) == expected:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    curl = shutil.which('curl.exe')
    log(f'Downloading {url}')
    if not (curl and subprocess.run([curl, '-L', '--fail', '--retry', '3', '-sS', '-o', str(part), url]).returncode == 0):
        request = urllib.request.Request(url, headers={'User-Agent': 'ps5port'})
        with urllib.request.urlopen(request, timeout=120) as response, part.open('wb') as out:
            shutil.copyfileobj(response, out, 1 << 20)
    if sha256(part) != expected:
        part.unlink()
        raise SystemExit(f'{url}: SHA-256 mismatch (download damaged or changed); run the build again')
    part.replace(target)


def runtime():
    """(relinker.exe, libs folder) of the pinned AnyPS5 release; ANYPS5_RELINKER / ANYPS5_LIBS use a local build instead."""
    if os.environ.get('ANYPS5_RELINKER') and os.environ.get('ANYPS5_LIBS'):
        return Path(os.environ['ANYPS5_RELINKER']), Path(os.environ['ANYPS5_LIBS'])
    home = TOOLS / f'anyps5-{RUNTIME["version"]}'
    relinker, libs = home / 'relinker.exe', home / 'libs'
    if relinker.is_file() and (libs / 'libkernel.prx').is_file():
        return relinker, libs
    name, digest = RUNTIME['relinker']
    download(RUNTIME['base'] + name, relinker, digest)
    name, digest = RUNTIME['libs']
    archive = home / name
    download(RUNTIME['base'] + name, archive, digest)
    log('Unpacking the AnyPS5 system libraries ...')
    temp = home / 'libs.extracting'
    remove_tree(temp)
    with zipfile.ZipFile(archive) as z:
        z.extractall(temp)
    found = next((p for p in temp.rglob('libkernel.prx')), None)
    if found is None:
        raise SystemExit(f'{name}: libkernel.prx missing')
    remove_tree(libs)
    found.parent.replace(libs)
    remove_tree(temp)
    archive.unlink()
    return relinker, libs


# ---------------------------------------------------------------- links (never touch what a link points to)

def is_link(path):
    p = Path(path)
    return p.is_symlink() or (hasattr(os.path, 'isjunction') and os.path.isjunction(p))


def remove_tree(path):
    """Deletes a folder made by this script. Junctions and symbolic links are removed as links only, so the
    player's dump they point to is never touched."""
    path = Path(path)
    if is_link(path):
        try:
            os.rmdir(path)
        except OSError:
            os.unlink(path)
        return
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            remove_tree(child)
        path.rmdir()
    else:
        path.unlink()


def junction(target, link):
    import _winapi
    _winapi.CreateJunction(str(target), str(link))


def link_file(target, link):
    """Hard link on the same drive, else a copy (small files) or a symbolic link (large ones; needs Developer
    Mode), else a copy."""
    try:
        os.link(target, link)
        return
    except OSError:
        pass
    if os.path.getsize(target) > (256 << 20):
        try:
            os.symlink(target, link)
            return
        except OSError:
            pass
    shutil.copy2(target, link)


def overlay(source, dest):
    """dest gets every entry of source that it does not have yet: folders as junctions (or, when dest already
    has that folder because it holds converted modules, the same overlay inside it), files as links."""
    dest.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        target = dest / entry.name
        if entry.is_dir():
            if target.exists() and not is_link(target):
                overlay(entry, target)
            elif not target.exists():
                junction(entry.resolve(), target)
        elif not target.exists():
            if entry.suffix.lower() in MODULE_EXTS and (dest / (entry.name + '.guest.prx')).exists():
                continue   # the converted module replaces it; the runtime never reads the original
            if entry.name.endswith('.esbak'):
                continue
            link_file(entry.resolve(), target)


# ---------------------------------------------------------------- the dump

def magic(path):
    with open(path, 'rb') as f:
        return f.read(4)


def clean_elf(path, output):
    """Writes the ELF of a module or executable: an ELF as is, a decrypted SELF unwrapped. A dumper's backup
    (<name>.esbak, the decrypted ELF of a fake-signed file) is preferred when it is an ELF."""
    backup = path.with_name(path.name + '.esbak')
    if backup.is_file() and magic(backup) == ELF:
        path = backup
    kind = magic(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if kind == ELF:
        shutil.copyfile(path, output)
        return True
    if kind in SELF:
        from unwrap_self import Refused, unwrap
        try:
            unwrap(str(path), str(output))
        except Refused as error:
            raise SystemExit(f'{path.name}: {error}\nThis dump is still encrypted: AnyPS5 needs a decrypted dump '
                             '(eboot.bin and the .prx modules dumped from a console with a dumper).')
        return True
    return False


def title_info(game):
    info = {'titleId': game.name, 'name': game.name}
    try:
        param = json.loads((game / 'sce_sys' / 'param.json').read_text(encoding='utf-8'))
        info['titleId'] = param.get('titleId') or info['titleId']
        loc = param.get('localizedParameters') or {}
        lang = loc.get(loc.get('defaultLanguage', '')) or loc.get('en-US') or next((v for v in loc.values() if isinstance(v, dict)), {})
        info['name'] = lang.get('titleName') or info['name']
    except (OSError, ValueError, AttributeError):
        pass
    return info


def build(port):
    game = (port / 'game_files').resolve()
    if not (game / 'eboot.bin').is_file():
        raise SystemExit(f'{game}: eboot.bin not found (game_files must be the dump folder that holds eboot.bin and sce_sys)')
    relinker, libs = runtime()
    src, out = port / 'build' / 'src', port / 'build' / 'out'
    remove_tree(src)
    log('Reading the dump: unwrapping eboot.bin and the modules ...')
    if not clean_elf(game / 'eboot.bin', src / 'eboot.elf'):
        raise SystemExit('eboot.bin is neither an ELF nor a SELF container')
    module_dirs = set()
    for folder, dirs, files in os.walk(game):
        rel = Path(folder).relative_to(game)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not is_link(Path(folder) / d)]
        if len(rel.parts) > 4:
            dirs[:] = []
        for name in files:
            if Path(name).suffix.lower() not in MODULE_EXTS:
                continue
            if clean_elf(Path(folder) / name, src / rel / name):
                module_dirs.add(rel.as_posix())
    extra = sorted(d for d in module_dirs if d not in STANDARD_MODULE_DIRS)
    if 'sce_module' in module_dirs and 'sce_modules' in module_dirs:
        raise SystemExit('The dump has both sce_module and sce_modules')
    log(f'{len(list(src.rglob("*.prx")) + list(src.rglob("*.sprx")))} modules; extra module folders: {", ".join(extra) or "none"}')

    # The previous conversion goes; saves (_sd), trophies, data/ and screenshots in out/ stay.
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob('app*.json')) + [out / 'app.exe', out / 'app0', out / 'libs']:
        remove_tree(old)
    cmd = [str(relinker), '--windows']
    for d in extra:
        cmd += ['--module-dir', d if d != '' else '.']
    cmd += [str((src / 'eboot.elf').resolve()), 'app.exe']
    log('Relinking for Windows (AnyPS5 relinker) ...')
    result = subprocess.run(cmd, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    lines = [l for l in result.stdout.replace('\r', '\n').splitlines() if l.strip() and '%' not in l]
    print('\n'.join(lines[-40:]), flush=True)
    if result.returncode != 0 or not (out / 'app.exe').is_file():
        raise SystemExit(f'The relinker failed (code {result.returncode}).')

    log('Linking the game files into app0 (nothing is copied but small files from another drive) ...')
    overlay(game, out / 'app0')
    junction(libs.resolve(), out / 'libs')
    info = title_info(game)
    (out / 'ps5port.json').write_text(json.dumps({**info, 'anyps5': RUNTIME['version'], 'module_dirs': sorted(module_dirs)}, indent=2))
    log(f'Done: {info["name"]} ({info["titleId"]}) -> {out / "app.exe"}')


# ---------------------------------------------------------------- PC settings -> AnyPS5 variables

# Windows locale -> console language number (ANYPS5_LANGUAGE).
LANGUAGES = {
    'ja': 0, 'en-us': 1, 'en': 1, 'fr': 2, 'fr-ca': 22, 'es': 3, 'es-mx': 20, 'es-419': 20, 'de': 4, 'it': 5, 'nl': 6,
    'pt-pt': 7, 'pt': 17, 'pt-br': 17, 'ru': 8, 'ko': 9, 'zh-tw': 10, 'zh-hk': 10, 'zh-cn': 11, 'zh': 11, 'fi': 12,
    'sv': 13, 'da': 14, 'nb': 15, 'no': 15, 'pl': 16, 'en-gb': 18, 'tr': 19, 'ar': 21, 'cs': 23,
}


def windows_language():
    try:
        buf = ctypes.create_unicode_buffer(85)
        ctypes.windll.kernel32.GetUserDefaultLocaleName(buf, 85)
        name = buf.value.lower()
    except (AttributeError, OSError):
        return None
    parts = name.split('-')
    for key in (name, '-'.join(parts[:2]), parts[0]):
        if key in LANGUAGES:
            return LANGUAGES[key]
    return None


def read_settings(port):
    values = {}
    ini = port / 'pc-settings.ini'
    if ini.is_file():
        for line in ini.read_text(encoding='utf-8', errors='replace').splitlines():
            line = line.strip()
            if not line or line[0] in '#;[' or '=' not in line:
                continue
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values


def number(values, key, default):
    try:
        return int(values.get(key, default))
    except ValueError:
        return default


def settings_env(port, game):
    """The AnyPS5 variables for the player's PC settings (launcher.json "groups"); a setting at its default sets
    nothing, so AnyPS5's own default applies."""
    v = read_settings(port)
    env = {}
    if number(v, 'modo', 0) == 1:
        env['ANYPS5_FULLSCREEN'] = '1'
    env_present = {1: 'mailbox', 2: 'immediate'}.get(number(v, 'vsync', 0))
    if env_present:
        env['ANYPS5_PRESENT_MODE'] = env_present
    res = {1: '1920x1080', 2: '3840x2160'}.get(number(v, 'res', 1))
    if res:
        env['APS5_OUTPUT_RESOLUTION'] = res
    overlay_page = number(v, 'show_fps', 0)
    if overlay_page in (1, 2):
        env['ANYPS5_OVERLAY'] = str(overlay_page)
    pos = {1: 'tr', 2: 'bl', 3: 'br'}.get(number(v, 'overlay_pos', 0))
    if pos:
        env['ANYPS5_OVERLAY_POS'] = pos
    volume = max(0, min(100, number(v, 'volume', 100)))
    if volume != 100:
        env['ANYPS5_VOLUME'] = str(volume)
    rumble = max(0, min(100, number(v, 'vibration', 100)))
    if rumble != 100:
        env['ANYPS5_RUMBLE'] = str(rumble)
    if number(v, 'mute_unfocused', 0) == 1:
        env['ANYPS5_MUTE_ON_FOCUS_LOSS'] = '1'
    language = number(v, 'language', -1)
    if language < 0:
        language = windows_language()
    if language is not None and language != 1:
        env['ANYPS5_LANGUAGE'] = str(language)
    # A dump without PlayGo data (sce_sys/playgo-chunk.dat) is complete: every chunk is installed.
    if game is not None and not (game / 'sce_sys' / 'playgo-chunk.dat').is_file():
        env['APS5_PLAYGO_UNLISTED_CHUNKS'] = 'installed'
    return env


def play(port):
    out = port / 'build' / 'out'
    exe = out / 'app.exe'
    if not exe.is_file():
        raise SystemExit('Not built yet: install the game in the launcher first.')
    game = (port / 'game_files').resolve() if (port / 'game_files').exists() else None
    env = dict(os.environ)
    env.update(settings_env(port, game))
    log_file = out / 'game-log.txt'
    with open(log_file, 'w', encoding='utf-8', errors='replace') as f:
        code = subprocess.run([str(exe)], cwd=out, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if code != 0:
        # The end of the log for the launcher's error box (the whole log is in game-log.txt).
        tail = log_file.read_text(encoding='utf-8', errors='replace').splitlines()
        keep = [l for l in tail if l.startswith(('FATAL', '  what()', 'terminate', '[libc] unhandled'))] or tail[-15:]
        print('\n'.join(keep[-15:]))
        print(f'(log: {log_file})')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build', 'play', 'env'])
    parser.add_argument('--port', required=True, help='the game folder made by the launcher')
    args = parser.parse_args()
    port = Path(args.port).resolve()
    if args.command == 'build':
        build(port)
        return 0
    if args.command == 'env':
        game = (port / 'game_files').resolve() if (port / 'game_files').exists() else None
        for key, value in sorted(settings_env(port, game).items()):
            print(f'{key}={value}')
        return 0
    return play(port)


if __name__ == '__main__':
    sys.exit(main())
