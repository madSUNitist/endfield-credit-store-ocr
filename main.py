# main.py
"""
Mod entry point for batch processing large datasets of shop screenshots.
Uses streaming generation, JSONL intermediate storage, and explicit resource management.
Designed for ~10GB datasets: constant memory footprint, crash-resilient, and fully typed.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from tqdm import tqdm

from endfield_ocr import ShopOCRProcessor
from endfield_ocr.config import PipelineConfig, OCRConfig
from endfield_ocr.models import ShopResult


# ==========================================================
# Configuration Constants (Adjust for your mod environment)
# ==========================================================
INPUT_DIR = Path("data/")
OUTPUT_DIR = Path("output")
REFS_DIR = Path("assests/refs")  # Optional: enables icon matching for namebar-less cards

MAX_INPUT_SIDE = 2400                 # Downscale high-res images to prevent VRAM/CPU spikes
OCR_MODE = "smart"                    # "fast" (single pass), "smart" (targeted fallback), "full" (debug)
ENABLE_RECURSIVE_REFS = False         # Scan refs directory recursively for sub-folders

# Output file paths
JSONL_PATH = OUTPUT_DIR / "results_stream.jsonl"
FAILED_PATH = OUTPUT_DIR / "failed_paths.txt"
FINAL_JSON_PATH = OUTPUT_DIR / "results_final.json"

ITEM_NAMES: List[str] = [
    # "高级作战记录", 
    "武器检查装置",  
    # "武器检查套组", 
    "武器检查单元", 
    "武库配额", 
    "强固模具", 
    "初级认知载体", 
    "初级作战记录", 
    "重型强固模具", 
    "中级作战记录", 
    "嵌晶玉", 
    "协议圆盘", 
    "协议圆盘组", 
    "协议棱柱", 
    "协议棱柱组", 
    "折金票", 
]

TOKEN = os.environ.get('PADDLE_TOKEN', None)
USE_REMOTE_BACKEND = TOKEN is not None

def setup_logging() -> None:
    """Configure application-wide logging to console and file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "processing.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


def collect_image_paths(input_dir: Path, recursive: bool = False) -> List[Path]:
    """Recursively collect all supported image files from the input directory."""
    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not input_dir.exists():
        return []
        
    if recursive:
        iterator = input_dir.rglob("*")
    else:
        iterator = input_dir.iterdir()
        
    results: List[Path] = []
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() not in supported_exts:
            continue
        if p.name.startswith("._"):
            continue
        results.append(p)
        
    results.sort()
    return results


def process_dataset() -> None:
    """Main pipeline execution: streams images, processes them, and saves results safely."""
    setup_logging()
    logging.info("Starting batch OCR processing...")
    logging.info("Input directory: %s", INPUT_DIR)
    logging.info("Output directory: %s", OUTPUT_DIR)

    if not INPUT_DIR.exists():
        logging.error("Input directory does not exist: %s", INPUT_DIR)
        return

    # 1. Initialize processor configuration explicitly
    ocr_cfg = OCRConfig(
        mode=OCR_MODE, 
        use_remote_backend=USE_REMOTE_BACKEND, 
        remote_api_token=TOKEN
    )
    config = PipelineConfig(
        ocr=ocr_cfg,
        max_input_side=MAX_INPUT_SIDE,
        recursive_refs=ENABLE_RECURSIVE_REFS,
        # debug_save_dir=Path("output/debug"),   # activate DEBUG
    )

    # 2. Prepare paths and collect images
    image_paths = collect_image_paths(INPUT_DIR, recursive=False)
    if not image_paths:
        logging.warning("No supported images found in %s", INPUT_DIR)
        return

    total_images = len(image_paths)
    logging.info("Found %d images to process.", total_images)

    # 3. Clear previous output files to prevent stale data from old runs
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()
    if FAILED_PATH.exists():
        FAILED_PATH.unlink()

    refs_path = REFS_DIR if REFS_DIR.exists() else None

    # 4. Initialize processor with context manager for safe resource cleanup
    with ShopOCRProcessor(config=config, refs_dir=refs_path, item_names=ITEM_NAMES) as processor:

        processed_count = 0
        error_count = 0

        # Progress callback: updates tqdm and flushes result to JSONL immediately
        def on_progress(index: int, total: Optional[int], result: ShopResult) -> None:
            nonlocal processed_count
            processed_count += 1
            
            # Write result to JSONL line-by-line. 
            # This prevents data loss if the script crashes mid-batch.
            with open(JSONL_PATH, "a", encoding="utf-8") as f:
                line = json.dumps(result.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

        # Error callback: logs failure, records path, and instructs pipeline to continue
        def on_error(path: Path, exception: Exception) -> bool:
            nonlocal error_count
            error_count += 1
            logging.error("Failed to process %s: %s", path.name, exception)
            
            with open(FAILED_PATH, "a", encoding="utf-8") as f:
                f.write(f"{path}\t{exception}\n")
                
            return True  # Return True to skip the failed image and continue iteration

        # 5. Execute batch processing using the streaming generator
        progress_bar = tqdm(
            processor.process_batch(image_paths, on_progress=on_progress, on_error=on_error),
            total=total_images,
            desc="Processing Images",
            unit="img",
            smoothing=0.05,  # Smoother ETA for long-running batches
        )

        # Consume the generator. All work happens inside process_batch callbacks.
        for _ in progress_bar:
            pass

        progress_bar.close()

    # 6. Final summary
    logging.info("Processing complete.")
    logging.info("Successfully processed: %d images", processed_count)
    logging.info("Failed: %d images", error_count)
    logging.info("Streamed results saved to: %s", JSONL_PATH)
    logging.info("Failed paths logged to: %s", FAILED_PATH)

    # 7. Optional: Consolidate JSONL to a single JSON array
    # Note: For extremely large datasets, consider keeping JSONL format.
    # This step loads all lines into memory for standard JSON compatibility.
    logging.info("Consolidating JSONL to final JSON...")
    final_results: List[Dict[str, Any]] = []
    try:
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    final_results.append(json.loads(stripped))
                    
        with open(FINAL_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
            
        logging.info("Final JSON array saved to: %s", FINAL_JSON_PATH)
    except Exception as e:
        logging.error("Failed to consolidate JSONL to JSON: %s", e)


if __name__ == "__main__":
    process_dataset()