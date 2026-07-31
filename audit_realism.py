from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import ollama
from pydantic import BaseModel

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_THRESHOLD = 7.0
DEFAULT_REPORT_NAME = "realism_audit_report.json"
MAX_IN_FLIGHT = 16  # ponytail: cap client-side requests; Ollama throttles GPU work
DIMENSION_BLOCKS = ("hygiene", "ai", "quality", "generation")
SCORED_DIMENSIONS = ("ai", "quality", "generation")
PROFILE_NAMES = ("mixed", "ai-fun", "photos")
IMPLEMENTED_LENSES = frozenset({"ai"})
LENS_ISSUE = {"generation": "#18", "quality": "#19", "hygiene": "#3"}


class AuditConfig:
    """Resolved profile, active lenses, and per-dimension thresholds."""

    __slots__ = ("profile", "lenses", "thresholds", "multi_dimensional")

    def __init__(
        self,
        *,
        profile: str | None,
        lenses: tuple[str, ...],
        thresholds: dict[str, float | None],
    ):
        self.profile = profile
        self.lenses = lenses
        self.thresholds = thresholds
        self.multi_dimensional = profile is not None


PROFILE_DEFAULTS: dict[str, dict] = {
    "mixed": {
        "lenses": ("ai",),
        "thresholds": {"ai": 7.0, "quality": 6.0, "generation": None},
    },
    "ai-fun": {
        "lenses": ("generation",),
        "thresholds": {"ai": None, "quality": None, "generation": 7.0},
    },
    "photos": {
        "lenses": ("quality",),
        "thresholds": {"ai": None, "quality": 6.0, "generation": None},
    },
}

PROFILE_PRIMARY_THRESHOLD = {"mixed": "ai", "ai-fun": "generation", "photos": "quality"}


class ScanProgress:
    """Thread-safe [n/total] prefix for parallel full scans."""

    def __init__(self, total: int):
        self.total = total
        self._lock = threading.Lock()
        self._next = 0

    def begin(self) -> str:
        with self._lock:
            self._next += 1
            return f"[{self._next}/{self.total}]"


class RealismAnalysis(BaseModel):
    realism_score: float
    is_realistic: bool
    detected_artifacts: list[str]
    reasoning: str


class RealismAnalysisFast(BaseModel):
    realism_score: float
    is_realistic: bool
    detected_artifacts: list[str]


class HygieneVerdict(BaseModel):
    action: str
    exact_dupe_of: str | None = None


class AiVerdict(BaseModel):
    realism_score: float
    is_realistic: bool | None = None
    issues: list[str] = []
    reasoning: str = ""


class QualityVerdict(BaseModel):
    keeper_score: float
    issues: list[str] = []
    reasoning: str = ""


class GenerationVerdict(BaseModel):
    success_score: float
    issues: list[str] = []
    reasoning: str = ""


def default_thresholds(threshold: float) -> dict[str, float | None]:
    return {"ai": threshold, "quality": None, "generation": None}


def empty_thresholds() -> dict[str, float | None]:
    return {"ai": None, "quality": None, "generation": None}


def parse_checks(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    lenses = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not lenses:
        raise ValueError("--checks requires at least one lens name")
    unknown = [l for l in lenses if l not in DIMENSION_BLOCKS]
    if unknown:
        raise ValueError(f"unknown lens(es): {', '.join(unknown)}")
    return lenses


def resolve_audit_config(
    *,
    profile: str | None,
    checks: tuple[str, ...] | None,
    threshold: float,
    threshold_ai: float | None,
    threshold_quality: float | None,
    threshold_generation: float | None,
) -> AuditConfig:
    if profile is None:
        thresholds = default_thresholds(threshold)
        if threshold_ai is not None:
            thresholds["ai"] = threshold_ai
        if threshold_quality is not None:
            thresholds["quality"] = threshold_quality
        if threshold_generation is not None:
            thresholds["generation"] = threshold_generation
        return AuditConfig(profile=None, lenses=("ai",), thresholds=thresholds)

    defaults = PROFILE_DEFAULTS[profile]
    lenses = checks if checks is not None else defaults["lenses"]
    thresholds = dict(empty_thresholds())

    if checks is None:
        for dim, value in defaults["thresholds"].items():
            if value is not None:
                thresholds[dim] = value
        primary = PROFILE_PRIMARY_THRESHOLD[profile]
        if primary in lenses:
            thresholds[primary] = threshold
    else:
        scored_active = [l for l in lenses if l in SCORED_DIMENSIONS]
        if scored_active:
            thresholds[scored_active[0]] = threshold

    if threshold_ai is not None:
        thresholds["ai"] = threshold_ai
    if threshold_quality is not None:
        thresholds["quality"] = threshold_quality
    if threshold_generation is not None:
        thresholds["generation"] = threshold_generation

    return AuditConfig(profile=profile, lenses=lenses, thresholds=thresholds)


def validate_audit_config(config: AuditConfig) -> None:
    missing = [l for l in config.lenses if l not in IMPLEMENTED_LENSES]
    if not missing:
        return
    parts = []
    for lens in missing:
        issue = LENS_ISSUE.get(lens, "open issue")
        parts.append(f"  - {lens}: not implemented yet (see {issue})")
    profile_note = f" (profile={config.profile})" if config.profile else ""
    raise SystemExit(
        "Cannot run audit: requested lens(es) are not implemented"
        f"{profile_note}:\n"
        + "\n".join(parts)
        + "\nUse --profile mixed for AI realism scoring, or --checks to override lenses."
    )


def effective_thresholds(meta: dict | None, cli_threshold: float) -> dict[str, float | None]:
    if meta and "thresholds" in meta:
        thresholds = dict(meta["thresholds"])
        if thresholds.get("ai") is None:
            legacy = meta.get("threshold")
            if legacy is not None:
                thresholds["ai"] = legacy
        return thresholds
    legacy = (meta or {}).get("threshold", cli_threshold)
    return default_thresholds(legacy)


def multi_dimensional_active(thresholds: dict[str, float | None]) -> bool:
    return sum(v is not None for v in thresholds.values()) > 1


def analysis_to_ai_block(analysis: dict) -> dict:
    return {
        "realism_score": analysis["realism_score"],
        "is_realistic": analysis.get("is_realistic"),
        "issues": analysis.get("detected_artifacts", []),
        "reasoning": analysis.get("reasoning", ""),
    }


def dimension_score(entry: dict, dimension: str) -> float | None:
    block = entry.get(dimension)
    if not block:
        return None
    if dimension == "ai":
        return block.get("realism_score")
    if dimension == "quality":
        return block.get("keeper_score")
    if dimension == "generation":
        return block.get("success_score")
    return None


def legacy_analysis_score(entry: dict) -> float | None:
    analysis = entry.get("analysis")
    if analysis and "realism_score" in analysis:
        return analysis["realism_score"]
    return dimension_score(entry, "ai")


def build_report_meta(
    *,
    threshold: float,
    model: str,
    max_dimension: int,
    fast: bool,
    dry_run: bool,
    profile: str | None = None,
    thresholds: dict[str, float | None] | None = None,
) -> dict:
    meta = {
        "threshold": threshold,
        "thresholds": thresholds or default_thresholds(threshold),
        "model": model,
        "max_dimension": max_dimension,
        "fast": fast,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
    }
    if profile is not None:
        meta["profile"] = profile
    return meta


def result_entry_from_lenses(
    filename: str,
    *,
    hygiene: dict | None = None,
    ai: dict | None = None,
    quality: dict | None = None,
    generation: dict | None = None,
    keep: bool | None = None,
) -> dict:
    entry: dict = {"file": filename}
    for key, block in (
        ("hygiene", hygiene),
        ("ai", ai),
        ("quality", quality),
        ("generation", generation),
    ):
        if block is not None:
            entry[key] = block
    if keep is not None:
        entry["keep"] = keep
    return entry


def realism_result_entry(filename: str, analysis: dict, *, multi_dimensional: bool = False) -> dict:
    if multi_dimensional:
        return result_entry_from_lenses(filename, ai=analysis_to_ai_block(analysis))
    return {"file": filename, "analysis": analysis}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and sort AI-generated images based on photorealism using Ollama.")
    parser.add_argument("--dir", default="./photos", help="Input directory containing images (.png, .jpg, .jpeg, .webp)")
    parser.add_argument("--filter-dir", default=None, help="Directory to move filtered-out/rejected files into (default: <input_dir>/rejects)")
    parser.add_argument("--model", default="llava", help="Local vision model to query via Ollama")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum score (1.0 to 10.0) for the profile's primary lens, or AI realism when no --profile; "
        "required unless --dry-run or --profile supplies a default",
    )
    parser.add_argument(
        "--threshold-ai",
        type=float,
        default=None,
        metavar="N",
        help="Override AI realism cutoff (1.0 to 10.0)",
    )
    parser.add_argument(
        "--threshold-quality",
        type=float,
        default=None,
        metavar="N",
        help="Override quality keeper cutoff (1.0 to 10.0)",
    )
    parser.add_argument(
        "--threshold-generation",
        type=float,
        default=None,
        metavar="N",
        help="Override generation success cutoff (1.0 to 10.0)",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default=None,
        help="Audit profile: mixed (AI realism), ai-fun (generation), photos (quality); "
        "default: legacy single-score AI mode (no meta.profile)",
    )
    parser.add_argument(
        "--checks",
        default=None,
        metavar="LENSES",
        help="Comma-separated lenses overriding the profile set (hygiene, ai, quality, generation)",
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
        "--max-dimension",
        type=int,
        default=0,
        metavar="N",
        help="Optional downscale before Ollama: long edge at most N px (0 = disabled, default)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Minimal VLM output (score, flag, artifacts only) for faster bulk triage",
    )
    args = parser.parse_args()

    if args.apply_report is not None and args.dry_run:
        parser.error("--apply-report cannot be used with --dry-run")

    try:
        args.checks = parse_checks(args.checks)
    except ValueError as e:
        parser.error(str(e))

    if args.profile is None and args.checks is not None:
        parser.error("--checks requires --profile")

    if args.dry_run:
        if args.threshold is None:
            if args.profile is not None:
                primary = PROFILE_PRIMARY_THRESHOLD[args.profile]
                args.threshold = PROFILE_DEFAULTS[args.profile]["thresholds"][primary]
            else:
                args.threshold = DEFAULT_THRESHOLD
    elif args.threshold is None:
        if args.profile is not None:
            primary = PROFILE_PRIMARY_THRESHOLD[args.profile]
            args.threshold = PROFILE_DEFAULTS[args.profile]["thresholds"][primary]
        else:
            parser.error("--threshold is required when not using --dry-run")

    for name, value in (
        ("--threshold", args.threshold),
        ("--threshold-ai", args.threshold_ai),
        ("--threshold-quality", args.threshold_quality),
        ("--threshold-generation", args.threshold_generation),
    ):
        if value is not None and (not (1.0 <= value <= 10.0) or math.isnan(value)):
            parser.error(f"{name} must be between 1.0 and 10.0, got {value}")
    if args.max_dimension < 0:
        parser.error(f"--max-dimension must be >= 0, got {args.max_dimension}")
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
    scores = [s for dim in SCORED_DIMENSIONS if (s := dimension_score(entry, dim)) is not None]
    legacy = legacy_analysis_score(entry)
    if legacy is not None:
        scores.append(legacy)
    score = max(scores) if scores else -1.0
    return (-score, entry.get("file", ""))


def pipeline_depth(image_count: int) -> int:
    return max(1, min(image_count, MAX_IN_FLIGHT))


def should_reject(entry: dict, threshold: float, meta: dict | None = None) -> bool | None:
    keep = entry.get("keep")
    if keep is True:
        return False
    if keep is False:
        return True
    if entry.get("status") == "error":
        return None

    hygiene = entry.get("hygiene")
    if hygiene:
        action = hygiene.get("action")
        if action == "reject":
            return True
        if action == "keep":
            return False

    score = legacy_analysis_score(entry)
    if score is None:
        return None
    ai_threshold = effective_thresholds(meta, threshold)["ai"]
    if ai_threshold is None:
        return None
    return score < ai_threshold


def unique_reject_path(filter_dir: Path, filename: str) -> Path:
    dest = filter_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 2
    while (candidate := filter_dir / f"{stem}_{n}{suffix}").exists():
        n += 1
    return candidate


def move_reject(img_path: Path, filter_dir: Path, move_lock: threading.Lock | None = None) -> Path:
    def _move() -> Path:
        filter_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_reject_path(filter_dir, img_path.name)
        shutil.move(str(img_path), str(dest))
        return dest

    if move_lock is None:
        return _move()
    with move_lock:
        return _move()


def apply_from_report(report_path: Path, input_dir: Path, filter_dir: Path, threshold: float):
    meta, results = load_report(report_path)
    moved = kept = skipped = 0

    for entry in results:
        filename = entry["file"]
        reject = should_reject(entry, threshold, meta)
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
                dest = move_reject(src, filter_dir)
                print(f"  -> Moved filtered image to {dest}")
                moved += 1
            except OSError as e:
                print(f"  -> Error moving {filename}: {e}")
                skipped += 1
        else:
            print(f"  -> Preserved keeper in place ({filename})")
            kept += 1

    print(f"\nApply complete: {moved} moved, {kept} kept, {skipped} skipped.")


def scaled_dimensions(width: int, height: int, max_dimension: int) -> tuple[int, int] | None:
    if max_dimension <= 0:
        return None
    long_edge = max(width, height)
    if long_edge <= max_dimension:
        return None
    scale = max_dimension / long_edge
    return max(1, round(width * scale)), max(1, round(height * scale))


def prepare_analysis_image(img_path: Path, max_dimension: int) -> tuple[Path, Path | None]:
    """Return (path for Ollama, temp file to delete after analysis, or None)."""
    if max_dimension <= 0:
        return img_path, None
    from PIL import Image, ImageOps

    with Image.open(img_path) as img:
        img = ImageOps.exif_transpose(img)
        new_size = scaled_dimensions(*img.size, max_dimension)
        if new_size is None:
            return img_path, None
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        fd, temp_name = tempfile.mkstemp(suffix=img_path.suffix.lower())
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            save_kwargs = {}
            if img_path.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs["quality"] = 95
            resized.save(temp_path, **save_kwargs)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return temp_path, temp_path


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
    config: AuditConfig,
    cli_threshold: float,
    dry_run: bool,
    filter_dir: Path,
    max_dimension: int,
    fast: bool,
    print_lock: threading.Lock,
    move_lock: threading.Lock,
    progress: ScanProgress | None = None,
) -> dict:
    tag = f"{progress.begin()} " if progress else ""

    with print_lock:
        print(f"{tag}Analyzing {img_path.name}...")

    blocks: dict[str, dict] = {}
    try:
        if "ai" in config.lenses:
            analysis_dict = analyze_image(img_path, model_name, max_dimension, fast)
            analysis = RealismAnalysis(**analysis_dict)
            blocks["ai"] = analysis_to_ai_block(analysis_dict)
            with print_lock:
                print(f"{tag}  Score: {analysis.realism_score} - Realistic: {analysis.is_realistic}")
    except Exception as e:
        with print_lock:
            traceback.print_exc()
            print(f"{tag}Error processing {img_path.name}: {e}")
        return {"file": img_path.name, "status": "error", "error": str(e)}

    if config.multi_dimensional:
        result = result_entry_from_lenses(img_path.name, **blocks)
    elif blocks.get("ai") is not None:
        # Legacy single-score: rebuild analysis dict from ai block for compat
        ai = blocks["ai"]
        result = realism_result_entry(
            img_path.name,
            {
                "realism_score": ai["realism_score"],
                "is_realistic": ai.get("is_realistic", False),
                "detected_artifacts": ai.get("issues", []),
                "reasoning": ai.get("reasoning", ""),
            },
        )
    else:
        result = {"file": img_path.name}

    if not dry_run:
        meta = {"threshold": cli_threshold, "thresholds": config.thresholds}
        if should_reject(result, cli_threshold, meta):
            try:
                dest = move_reject(img_path, filter_dir, move_lock)
                with print_lock:
                    print(f"{tag}  -> Moved filtered image to {dest}")
            except OSError as e:
                with print_lock:
                    print(f"{tag}  -> Error moving {img_path.name}: {e}")
                result["move_error"] = str(e)
        else:
            with print_lock:
                print(f"{tag}  -> Preserved keeper in place")

    return result


def analyze_image(img_path: Path, model_name: str, max_dimension: int = 0, fast: bool = False) -> dict:
    send_path, temp_path = prepare_analysis_image(img_path, max_dimension)
    if fast:
        prompt = (
            "Analyze this image for photorealism. Return realism_score (1.0-10.0), "
            "is_realistic, and detected_artifacts only. Do not explain."
        )
        schema = RealismAnalysisFast.model_json_schema()
    else:
        prompt = (
            "Analyze this image for photorealism. Identify if it is realistic, rate it from 1.0 to 10.0, "
            "list any AI-generated artifacts, and explain your reasoning."
        )
        schema = RealismAnalysis.model_json_schema()
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [str(send_path)]
            }],
            format=schema,
            options={"temperature": 0}
        )
        result = json.loads(response.message.content)
        if fast:
            RealismAnalysisFast(**result)
            result["reasoning"] = ""
        return result
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def run_audit(args, input_dir: Path, filter_dir: Path):
    config = resolve_audit_config(
        profile=args.profile,
        checks=args.checks,
        threshold=args.threshold,
        threshold_ai=args.threshold_ai,
        threshold_quality=args.threshold_quality,
        threshold_generation=args.threshold_generation,
    )
    validate_audit_config(config)

    ensure_model(args.model)

    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()]
    depth = pipeline_depth(len(image_paths))
    print_lock = threading.Lock()
    move_lock = threading.Lock()
    progress = ScanProgress(len(image_paths)) if image_paths else None

    if config.profile:
        print(f"Profile: {config.profile} (lenses: {', '.join(config.lenses)})")

    def process(img_path: Path) -> dict:
        return process_image(
            img_path,
            args.model,
            config,
            args.threshold,
            args.dry_run,
            filter_dir,
            args.max_dimension,
            args.fast,
            print_lock,
            move_lock,
            progress,
        )

    with ThreadPoolExecutor(max_workers=depth) as executor:
        results = list(executor.map(process, image_paths))

    results.sort(key=score_sort_key)

    if results:
        report_path = input_dir / DEFAULT_REPORT_NAME
        meta = build_report_meta(
            threshold=args.threshold,
            model=args.model,
            max_dimension=args.max_dimension,
            fast=args.fast,
            dry_run=args.dry_run,
            profile=config.profile,
            thresholds=config.thresholds,
        )
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
    assert should_reject({"file": "g.jpg", "hygiene": {"action": "reject", "exact_dupe_of": "orig.jpg"}}, 7.0) is True
    assert should_reject({"file": "h.jpg", "hygiene": {"action": "keep"}}, 7.0) is False
    assert should_reject(
        {"file": "i.jpg", "ai": {"realism_score": 6.0, "issues": [], "reasoning": ""}},
        7.0,
    ) is True
    assert should_reject(
        {"file": "j.jpg", "ai": {"realism_score": 8.0, "issues": [], "reasoning": ""}},
        7.0,
        {"thresholds": {"ai": 7.0, "quality": None, "generation": None}},
    ) is False
    assert should_reject({"file": "k.jpg", "quality": {"keeper_score": 3.0, "issues": []}}, 7.0) is None
    assert effective_thresholds({"threshold": 8.0, "thresholds": {"ai": None, "quality": 6.0, "generation": None}}, 7.0)["ai"] == 8.0
    assert multi_dimensional_active({"ai": 7.0, "quality": 6.0, "generation": None}) is True
    assert multi_dimensional_active({"ai": 7.0, "quality": None, "generation": None}) is False
    mixed = resolve_audit_config(profile="mixed", checks=None, threshold=7.5, threshold_ai=None, threshold_quality=None, threshold_generation=None)
    assert mixed.profile == "mixed" and mixed.lenses == ("ai",) and mixed.multi_dimensional
    assert mixed.thresholds["ai"] == 7.5 and mixed.thresholds["quality"] == 6.0
    photos_ai = resolve_audit_config(profile="photos", checks=("ai",), threshold=8.0, threshold_ai=None, threshold_quality=None, threshold_generation=None)
    assert photos_ai.thresholds["ai"] == 8.0 and photos_ai.thresholds["quality"] is None
    legacy = resolve_audit_config(profile=None, checks=None, threshold=8.0, threshold_ai=None, threshold_quality=None, threshold_generation=None)
    assert legacy.profile is None and not legacy.multi_dimensional and legacy.thresholds["ai"] == 8.0
    assert parse_checks("hygiene,ai") == ("hygiene", "ai")
    try:
        parse_checks("")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty --checks")
    try:
        validate_audit_config(resolve_audit_config(profile="photos", checks=None, threshold=6.0, threshold_ai=None, threshold_quality=None, threshold_generation=None))
    except SystemExit:
        pass
    else:
        raise AssertionError("photos profile should fail validation")
    ai_block = analysis_to_ai_block(
        {"realism_score": 8.0, "is_realistic": True, "detected_artifacts": ["x"], "reasoning": "ok"}
    )
    assert ai_block["issues"] == ["x"] and ai_block["realism_score"] == 8.0
    legacy_entry = realism_result_entry("a.jpg", {"realism_score": 5.0, "is_realistic": False, "detected_artifacts": [], "reasoning": ""})
    assert "analysis" in legacy_entry and "ai" not in legacy_entry
    multi_entry = realism_result_entry(
        "b.jpg",
        {"realism_score": 5.0, "is_realistic": False, "detected_artifacts": [], "reasoning": ""},
        multi_dimensional=True,
    )
    assert "ai" in multi_entry and "analysis" not in multi_entry
    assert score_sort_key({"file": "z.jpg", "quality": {"keeper_score": 9.0}})[0] == -9.0
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump([{"file": "x.jpg", "analysis": {"realism_score": 5.0}}], f)
        f.flush()
        meta, results = load_report(Path(f.name))
    assert meta is None and results[0]["file"] == "x.jpg"
    assert pipeline_depth(0) == 1
    assert pipeline_depth(3) == 3
    assert pipeline_depth(100) == MAX_IN_FLIGHT
    assert scaled_dimensions(100, 50, 0) is None
    assert scaled_dimensions(100, 50, 200) is None
    assert scaled_dimensions(2000, 1000, 1024) == (1024, 512)
    assert scaled_dimensions(1000, 2000, 1024) == (512, 1024)
    assert scaled_dimensions(1, 2, 1) == (1, 1)
    assert scaled_dimensions(2, 1, 1) == (1, 1)
    assert "reasoning" not in RealismAnalysisFast.model_json_schema()["properties"]
    assert "reasoning" in RealismAnalysis.model_json_schema()["properties"]
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "big.png"
        Image.new("RGB", (400, 200), "red").save(src)
        send_path, temp_path = prepare_analysis_image(src, 100)
        assert temp_path is not None
        try:
            assert send_path != src
            assert Image.open(send_path).size == (100, 50)
        finally:
            temp_path.unlink(missing_ok=True)
        assert prepare_analysis_image(src, 0) == (src, None)
        assert prepare_analysis_image(src, 500) == (src, None)
    with tempfile.TemporaryDirectory() as tmp:
        rejects = Path(tmp)
        (rejects / "a.jpg").write_text("old")
        assert unique_reject_path(rejects, "a.jpg") == rejects / "a_2.jpg"
        assert unique_reject_path(rejects, "b.jpg") == rejects / "b.jpg"
    p = ScanProgress(3)
    assert p.begin() == "[1/3]"
    assert p.begin() == "[2/3]"
    seen = []

    def _grab():
        seen.append(p.begin())

    threads = [threading.Thread(target=_grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 8 and len(set(seen)) == 8
    _check_process_image_error_handling()
    _check_run_audit_progress_tags()


def _check_process_image_error_handling():
    import io
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from unittest.mock import patch

    mod = sys.modules[__name__]
    legacy = resolve_audit_config(
        profile=None, checks=None, threshold=7.0,
        threshold_ai=None, threshold_quality=None, threshold_generation=None,
    )
    with tempfile.TemporaryDirectory() as tmp:
        img = Path(tmp) / "x.jpg"
        img.write_bytes(b"x")
        err = io.StringIO()
        out = io.StringIO()
        with patch.object(mod, "analyze_image", side_effect=RuntimeError("unexpected")), redirect_stderr(err), redirect_stdout(out):
            result = process_image(
                img, "llava", legacy, 7.0, True, Path(tmp) / "rejects", 0, False,
                threading.Lock(), threading.Lock(), ScanProgress(2),
            )
        assert result == {"file": "x.jpg", "status": "error", "error": "unexpected"}
        assert "Traceback" in err.getvalue()
        assert "RuntimeError: unexpected" in err.getvalue()
        assert "[1/2] Error processing x.jpg: unexpected" in out.getvalue()


def _check_run_audit_progress_tags():
    import io
    import re
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from unittest.mock import patch

    mod = sys.modules[__name__]
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "photos"
        filter_dir = Path(tmp) / "rejects"
        input_dir.mkdir()
        for name in ("good.jpg", "bad.jpg", "fail.jpg"):
            (input_dir / name).write_bytes(b"x")

        def mock_analyze(img_path: Path, model_name: str, max_dimension: int = 0, fast: bool = False) -> dict:
            if img_path.name == "fail.jpg":
                raise RuntimeError("boom")
            score = 5.0 if img_path.name == "bad.jpg" else 8.0
            return {
                "realism_score": score,
                "is_realistic": score >= 7.0,
                "detected_artifacts": [],
                "reasoning": "test",
            }

        args = argparse.Namespace(
            model="llava",
            threshold=7.0,
            threshold_ai=None,
            threshold_quality=None,
            threshold_generation=None,
            profile=None,
            checks=None,
            dry_run=False,
            max_dimension=0,
            fast=False,
            report_path_display=None,
        )
        captured = io.StringIO()
        with patch.object(mod, "ensure_model"), patch.object(mod, "analyze_image", side_effect=mock_analyze), redirect_stdout(captured), redirect_stderr(io.StringIO()):
            run_audit(args, input_dir, filter_dir)

        out = captured.getvalue()
        analyzing_tags = re.findall(r"(\[\d+/3\]) Analyzing (\S+?)\.\.\.", out)
        assert len(analyzing_tags) == 3
        assert len({tag for tag, _ in analyzing_tags}) == 3

        by_tag = {tag: fname for tag, fname in analyzing_tags}
        for tag, fname in by_tag.items():
            tag_lines = [line for line in out.splitlines() if line.startswith(tag)]
            if fname == "fail.jpg":
                assert any("Error processing fail.jpg" in line for line in tag_lines)
            else:
                assert any("Score:" in line for line in tag_lines)
            if fname == "bad.jpg":
                assert any("Moved filtered image" in line for line in tag_lines)
            elif fname == "good.jpg":
                assert any("Preserved keeper" in line for line in tag_lines)

        _, results = load_report(input_dir / DEFAULT_REPORT_NAME)
        assert len(results) == 3
        assert sum(entry.get("status") == "error" for entry in results) == 1


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
