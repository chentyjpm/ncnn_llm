"""Locate reference probes built against the real runtime utility objects.
Set NCNN_LLM_BUILD_DIR for an out-of-tree CMake build (default: ./build).
"""
import os
import pathlib
import sys


def probe_path(name, kind):
    build = pathlib.Path(os.environ.get('NCNN_LLM_BUILD_DIR',
                         str(pathlib.Path(__file__).resolve().parents[1] / 'build')))
    if kind == 'library':
        filename = (name + '.dll' if sys.platform == 'win32' else
                    'lib' + name + ('.dylib' if sys.platform == 'darwin' else '.so'))
    else:
        filename = name + ('.exe' if sys.platform == 'win32' else '')
    for directory in (build, build / 'Release', build / 'Debug'):
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Build the CMake target {name} and set NCNN_LLM_BUILD_DIR; '
                            f'{filename} was not found in {build}')
