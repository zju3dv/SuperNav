"""Support ``python -m supernav`` with the same command tree as ``supernav``."""

from supernav.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
