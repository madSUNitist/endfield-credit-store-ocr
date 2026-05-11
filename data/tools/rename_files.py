#!/usr/bin/env python3
"""
Rename all files in a folder to sequential numeric names (e.g., 00000.jpg, 00001.png).
Features:
- Pad numbers to 5 digits (configurable)
- Keep original file extensions
- Optional sorting: by name (default) or by modification time
- Retry on failure (e.g., file locked by another process)
- Dry-run preview
- Output mapping.json: original filename -> new filename
- Warn if pad width is insufficient for the number of files

Usage:
    python rename_files.py /path/to/folder
    python rename_files.py /path/to/folder --sort time
    python rename_files.py /path/to/folder --dry-run
    python rename_files.py /path/to/folder --pad 4
"""

import os
import sys
import time
import json
import argparse
from typing import List

def get_file_list(folder: str, sort_key: str) -> List[str]:
    """Return sorted list of file paths (excluding directories)."""
    files = []
    for entry in os.listdir(folder):
        full = os.path.join(folder, entry)
        if os.path.isfile(full):
            files.append(full)
    # Sort according to chosen key
    if sort_key == "name":
        files.sort(key=lambda p: os.path.basename(p))
    elif sort_key == "time":
        files.sort(key=lambda p: os.path.getmtime(p))
    else:
        raise ValueError(f"Unknown sort key: {sort_key}")
    return files

def rename_with_retry(src: str, dst: str, retries: int = 5, delay: float = 1.0) -> bool:
    """
    Attempt to rename src to dst, retrying on failure.
    Returns True if successful, False otherwise.
    """
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED after {retries} retries: {e}")
                return False
            wait = delay * (2 ** attempt)  # exponential backoff
            print(f"  Retry {attempt+1}/{retries} after {wait:.2f}s: {e}")
            time.sleep(wait)
    return False

def rename_files(
    folder: str,
    sort_key: str = "name",
    pad_width: int = 5,
    dry_run: bool = False,
    retries: int = 5,
    delay: float = 1.0,
) -> None:
    """
    Rename all files in 'folder' to sequential numbers with fixed digit padding.
    Outputs mapping.json (original basename -> new basename) after renaming.
    """
    files = get_file_list(folder, sort_key)
    if not files:
        print(f"No files found in {folder}")
        return

    # Check pad width overflow
    max_index = len(files) - 1
    max_representable = 10 ** pad_width - 1
    if max_index > max_representable:
        print(f"WARNING: Number of files ({len(files)}) exceeds the maximum representable "
              f"with pad width {pad_width} (max index = {max_representable}). "
              f"Filenames will have more than {pad_width} digits. "
              f"Consider increasing --pad value (e.g., --pad {pad_width + 1}).")

    print(f"Found {len(files)} files. Sorting by {sort_key}.")
    if dry_run:
        print("DRY RUN - no changes will be made.")

    # Prepare mapping: original basename -> new basename
    mapping = {}
    for idx, src in enumerate(files):
        ext = os.path.splitext(src)[1]
        new_name = f"{idx:0{pad_width}d}{ext}"
        mapping[os.path.basename(src)] = new_name

    success_count = 0
    for idx, src in enumerate(files):
        new_name = mapping[os.path.basename(src)]
        dst = os.path.join(folder, new_name)

        if dry_run:
            print(f"[DRY] {os.path.basename(src)} -> {new_name}")
            success_count += 1
        else:
            print(f"Renaming {os.path.basename(src)} -> {new_name}")
            if rename_with_retry(src, dst, retries, delay):
                success_count += 1
            else:
                print(f"  ERROR: Could not rename {src}")

    # Write mapping.json (only when actually renaming, not in dry-run)
    if not dry_run:
        mapping_path = os.path.join(folder, "mapping.json")
        try:
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
            print(f"Mapping saved to {mapping_path}")
        except Exception as e:
            print(f"WARNING: Could not write mapping.json: {e}")

    print(f"Done. {success_count} of {len(files)} files renamed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Rename files to sequential numbers (00000.ext).")
    parser.add_argument("folder", help="Target folder path")
    parser.add_argument("--sort", choices=["name", "time"], default="name",
                        help="Sort order for renaming (default: name)")
    parser.add_argument("--pad", type=int, default=5,
                        help="Number of digits to pad numbers (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without actually renaming")
    parser.add_argument("--retries", type=int, default=5,
                        help="Number of retries on failure (default: 5)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Initial retry delay in seconds (default: 1.0)")

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    rename_files(
        folder=args.folder,
        sort_key=args.sort,
        pad_width=args.pad,
        dry_run=args.dry_run,
        retries=args.retries,
        delay=args.delay,
    )

if __name__ == "__main__":
    main()