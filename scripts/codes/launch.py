"""Scheduler of the code-variation fleets: scripts/launch_matrix.py with a choice of final tests.

scripts/launch_matrix.py is left untouched, because the records of earlier studies in this repository pin its
exact text by checksum. Its list of final tests (FULL_PROBES) is replaced here before its main() runs.

  python scripts/codes/launch.py --probes intact,no_code,no_goal [every option of scripts/launch_matrix.py]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import launch_matrix as lm  # noqa: E402


def main():
  argv = sys.argv[1:]
  if '--probes' in argv:
    i = argv.index('--probes')
    lm.FULL_PROBES = argv[i + 1]
    del argv[i:i + 2]
  sys.argv = [sys.argv[0]] + argv
  lm.main()


if __name__ == '__main__':
  main()
