#!/usr/bin/env python3
"""Reproduce the combined synthetic language tests in one Python process.

Run with the repository's Transformers/ncnn-enabled Python environment:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/run_minicpm_language_smoke.py
No real model checkpoints are loaded. Keep this separate from converter runs.
"""
import unittest
from minicpm_streamed_reference import self_test
import test_minicpm_v47_language_export


def main():
    self_test()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_minicpm_v47_language_export)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print('PASS: streamed reference + all language export tests in one Python process', flush=True)


if __name__ == '__main__':
    main()
