import argparse
import json
import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import ollama
from pydantic import BaseModel

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_THRESHOLD = 7.0
DEFAULT_REPORT_NAME = "realism_audit_report.json"
MAX_WORKERS = 16
DEFAULT_PIPELINE = 4  # ponytail: Ollama's usual auto default; server throttles actual GPU work


class RealismAnalysis(BaseModel):
    realism_score: float
    is_realistic: bool
    detected_artifacts: list[str]
    reasoning: str


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and sort AI-generated images based on photorealism using Ollama.")
    parser.add_argument("--dir", default="./photos", help="Input directory containing images (.png, .jpg, .jpeg, .webp)")
    parser.add_argument("--filter-dir", default=None, help="Directory to move filtered-out/rejected files into (default: <input_dir>/rejects)")
    parser.add_argument("--model", default="llava", help="Local vision model to query via Ollama")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum realism score (1.0 to 10.0) to keep image in-place; required unless --dry-run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate the JSON report without physically moving files into filter_dir")
    parser.add_argument(
        "--apply-report",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help="Apply file moves from an audit report without re-analyzing (default: <input_dir>/realism_audit_report.json)",
    )
    parser.add_argument("--report-path-display", default=None, help="Custom path string to display in the final report message")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override concurrent Ollama requests (default: auto; use 1 for sequential)",
    )
    args = parser.parse_args()

    if args.apply_report is not None and args.dry_run:
        parser.error("--apply-report cannot be used with --dry-run")

    if args.dry_run:
        if args.threshold is None:
            args.threshold = DEFAULT_THRESHOLD
    elif args.threshold is None:
        parser.error("--threshold is required when not using --dry-run")

    if not (1.0 <= args.threshold <= 10.0) or math.isnan(args.threshold):
        parser.error(f"--threshold must be between 1.0 and 10.0, got {args.threshold}")

    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")

    return args


def load_report(path: Path) -> tuple[dict | None, list]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return None, data
    if isinstance(data, dict) and "results" in data:
        return data.get("meta"), data["results"]
    raise SystemExit(f"Error: Unrecognized report format in '{path}'")


def write_report(path: Path, meta: dict, results: list):
    with open(path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)


def score_sort_key(entry: dict) -> tuple[float, str]:
    analysis = entry.get("analysis")
    score = analysis["realism_score"] if analysis and "realism_score" in analysis else -1.0
    return (-score, entry.get("file", ""))


def resolve_workers(explicit: int | None, image_count: int) -> int:
    if explicit is not None:
        return min(explicit, MAX_WORKERS)
    parallel = int(os.environ.get("OLLAMA_NUM_PARALLEL", DEFAULT_PIPELINE))
    return max(1, min(image_count, parallel))


def should_reject(entry: dict, threshold: float) -> bool | None:
    keep = entry.get("keep")
    if keep is True:
        return False
    if keep is False:
        return True
    if entry.get("status") == "error":
        return None
    return entry["analysis"]["realism_score"] < threshold


def move_reject(img_path: Path, filter_dir: Path):
    filter_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_path), str(filter_dir / img_path.name))


def apply_from_report(report_path: Path, input_dir: Path, filter_dir: Path, threshold: float):
    _, results = load_report(report_path)
    moved = kept = skipped = 0

    for entry in results:
        filename = entry["file"]
        reject = should_reject(entry, threshold)
        if reject is None:
            print(f"Skipping {filename}: error entry without keep override")
            skipped += 1
            continue

        src = input_dir / filename
        if not src.exists():
            print(f"Warning: {filename} not found, skipping")
            skipped += 1
            continue

        if reject:
            try:
                move_reject(src, filter_dir)
                print(f"  -> Moved filtered image to {filter_dir / filename}")
                moved += 1
            except OSError as e:
                print(f"  -> Error moving {filename}: {e}")
                skipped += 1
        else:
            print(f"  -> Preserved keeper in place ({filename})")
            kept += 1

    print(f"\nApply complete: {moved} moved, {kept} kept, {skipped} skipped.")


def ensure_model(model_name: str):
    print(f"Checking if model '{model_name}' is available locally...")
    try:
        ollama.show(model_name)
        print(f"Model '{model_name}' is ready.")
    except ollama.ResponseError as e:
        if e.status_code == 404:
            print(f"Model '{model_name}' not found. Pulling... (this may take a while)")
            ollama.pull(model_name)
            print(f"Successfully pulled '{model_name}'.")
        else:
            raise SystemExit(f"Error: Ollama returned status {e.status_code} for model '{model_name}': {e}")
    except Exception as e:
        raise SystemExit(f"Error: Cannot reach Ollama. Is 'ollama serve' running? {e}")


def process_image(
    img_path: Path,
    model_name: str,
    threshold: float,
    dry_run: bool,
    filter_dir: Path,
    print_lock: threading.Lock,
) -> dict | None:
    with print_lock:
        print(f"Analyzing {img_path.name}...")
    try:
        analysis_dict = analyze_image(img_path, model_name)
        analysis = RealismAnalysis(**analysis_dict)
    except Exception as e:
        with print_lock:
            print(f"Error processing {img_path.name}: {e}")
        return {"file": img_path.name, "status": "error", "error": str(e)}

    with print_lock:
        print(f"  Score: {analysis.realism_score} - Realistic: {analysis.is_realistic}")

    result = {"file": img_path.name, "analysis": analysis_dict}

    if not dry_run:
        if should_reject(result, threshold):
            try:
                move_reject(img_path, filter_dir)
                with print_lock:
                    print(f"  -> Moved filtered image to {filter_dir / img_path.name}")
            except OSError as e:
                with print_lock:
                    print(f"  -> Error moving {img_path.name}: {e}")
                return None
        else:
            with print_lock:
                print("  -> Preserved keeper in place")

    return result


def analyze_image(img_path: Path, model_name: str) -> dict:
    response = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": "Analyze this image for photorealism. Identify if it is realistic, rate it from 1.0 to 10.0, list any AI-generated artifacts, and explain your reasoning.",
            "images": [str(img_path)]
        }],
        format=RealismAnalysis.model_json_schema(),
        options={"temperature": 0}
    )
    return json.loads(response.message.content)


def run_audit(args, input_dir: Path, filter_dir: Path):
    ensure_model(args.model)

    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()]
    workers = resolve_workers(args.workers, len(image_paths))
    print_lock = threading.Lock()

    def process(img_path: Path) -> dict | None:
        return process_image(img_path, args.model, args.threshold, args.dry_run, filter_dir, print_lock)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = [r for r in executor.map(process, image_paths) if r is not None]

    results.sort(key=score_sort_key)

    if results:
        report_path = input_dir / DEFAULT_REPORT_NAME
        meta = {
            "threshold": args.threshold,
            "model": args.model,
            "workers": workers,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": args.dry_run,
        }
        write_report(report_path, meta, results)
        display_report_path = args.report_path_display or str(report_path)
        print(f"\nAudit complete! {len(results)} of {len(image_paths)} images in report.")
        print(f"Report saved to {display_report_path}")
    else:
        print(f"\nNo images were processed out of {len(image_paths)} found.")


def _self_check():
    assert should_reject({"file": "a.jpg", "keep": True, "analysis": {"realism_score": 1.0}}, 7.0) is False
    assert should_reject({"file": "b.jpg", "keep": False, "analysis": {"realism_score": 10.0}}, 7.0) is True
    assert should_reject({"file": "c.jpg", "analysis": {"realism_score": 6.0}}, 7.0) is True
    assert should_reject({"file": "d.jpg", "analysis": {"realism_score": 8.0}}, 7.0) is False
    assert should_reject({"file": "e.jpg", "status": "error", "error": "boom"}, 7.0) is None
    assert should_reject({"file": "f.jpg", "status": "error", "error": "boom", "keep": False}, 7.0) is True
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump([{"file": "x.jpg", "analysis": {"realism_score": 5.0}}], f)
        f.flush()
        meta, results = load_report(Path(f.name))
    assert meta is None and results[0]["file"] == "x.jpg"
    assert resolve_workers(None, 10) == DEFAULT_PIPELINE
    assert resolve_workers(None, 1) == 1
    assert resolve_workers(1, 10) == 1
    assert resolve_workers(99, 10) == MAX_WORKERS
    assert score_sort_key({"file": "b.jpg", "analysis": {"realism_score": 8.0}}) == (-8.0, "b.jpg")
    assert score_sort_key({"file": "a.jpg", "analysis": {"realism_score": 8.0}}) < score_sort_key(
        {"file": "b.jpg", "analysis": {"realism_score": 8.0}}
    )


def main():
    args = parse_args()

    input_dir = Path(args.dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    filter_dir = Path(args.filter_dir) if args.filter_dir else input_dir / "rejects"

    if args.apply_report is not None:
        report_path = input_dir / DEFAULT_REPORT_NAME if args.apply_report == "__default__" else Path(args.apply_report)
        if not report_path.is_file():
            raise SystemExit(f"Error: Report file '{report_path}' does not exist.")
        apply_from_report(report_path, input_dir, filter_dir, args.threshold)
        return

    run_audit(args, input_dir, filter_dir)


if __name__ == "__main__":
    _self_check()
    main()
