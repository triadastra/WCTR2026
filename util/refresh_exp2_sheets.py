#!/usr/bin/env python
"""Retired in-place workbook updater: stale sheets cannot be certified."""
def main():
    print("In-place updates are retired. Rebuild the complete validated workbook "
          "with scripts/build_paper_results.py; generate statistical outputs "
          "from the validated reporting scripts.")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
