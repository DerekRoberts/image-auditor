# Image Auditor

A local CLI tool for auditing, deduplicating, and sorting photo collections and AI generations using local vision LLMs via [Ollama](https://ollama.com).

Evaluates image quality, detects generation artifacts and defects (e.g. plastic skin, warped fingers, lighting errors), scores quality from 1.0 to 10.0, and automatically filters unwanted photos into a target directory while preserving keepers in-place.

---

## Features

- **Local & Private:** Queries local vision models (`llava`, `llama3.2-vision`, `qwen2.5-vl`) running via Ollama — zero cloud API costs or data leakage.
- **Structured Pydantic Output:** Enforces JSON schema validation for reliable numerical scores, boolean flags, artifact tags, and forensic reasoning.
- **Automated Image Sorting:** Preserves quality images in-place and moves filtered files to `--filter-dir`.
- **Dry-Run Mode:** Generate diagnostic JSON reports without moving any image files.
- **Containerized CLI:** Pre-packaged wrapper for Podman / Docker to run like a native binary from anywhere on your system.

---

## Quickstart

### 1. Prerequisites
- [Ollama](https://ollama.com) running locally (`ollama serve`).
- Podman or Docker installed.

### 2. Installation
Clone the repository and run the setup script:

```bash
git clone https://github.com/DerekRoberts/image-auditor.git
cd image-auditor
./setup.sh
```

The `./setup.sh` script builds the container image and installs the standalone `image-auditor` binary wrapper into `~/.local/bin/image-auditor`.

---

## Usage

### Run from anywhere
```bash
image-auditor ~/Downloads --threshold 7.5
```

### Dry-run (JSON report only, no file movements)
```bash
image-auditor ~/Downloads --dry-run
```

### Apply moves from a reviewed report (no Ollama)
After reviewing or hand-editing `realism_audit_report.json`:
```bash
image-auditor ~/Downloads --apply-report --threshold 7.5
```

### Specify custom vision model
```bash
image-auditor ~/Downloads --model llava --threshold 8.0
```

---

## CLI Options

| Flag | Default | Description |
| --- | --- | --- |
| `directory` | `.` | Target input directory containing images (`.png`, `.jpg`, `.jpeg`, `.webp`) |
| `--filter-dir` | `<input_dir>/rejects` | Directory to move filtered-out/rejected files into |
| `--model` | `llava` | Local vision model to query via Ollama |
| `--threshold` | `7.0` (dry-run only) | Minimum score (1.0 to 10.0) required to keep image in-place; **required** when not using `--dry-run` |
| `--dry-run` | `False` | Generate report without moving files |
| `--apply-report` | — | Apply file moves from an existing audit report without re-analyzing (optional path; default: `<input_dir>/realism_audit_report.json`) |
| `--max-dimension` | `0` (off) | Optional: downscale before Ollama so the long edge is at most N px (`0` = send originals; **default**) |
| `--fast` | `False` | Minimal VLM output (score, flag, artifacts only); skips reasoning for faster bulk triage |

### Fast vs full mode (`--fast`)

**Full mode (default)** asks the model for forensic reasoning alongside score, realism flag, and artifact tags. Use this for dry-runs, borderline decisions, and any run where you will read the JSON report before moving files.

**Fast mode** (`--fast`) uses a reduced prompt and schema — `realism_score`, `is_realistic`, and `detected_artifacts` only. The report still includes a `reasoning` key (empty string) so the format stays consistent; `meta.fast` is `true`. Expect roughly 10–20% faster per image, with larger gains on verbose models.

**When to use `--fast`:**

- Bulk triage on large folders where obvious rejects/keepers dominate
- A first pass before `--apply-report`, with manual `keep` overrides on edge cases
- Combined with `--dry-run` to generate scores quickly, then re-run borderline files at full resolution

**When to stay on full mode:**

- First audit of a new collection or model
- Scores near your threshold (±1.0) where reasoning helps you decide overrides
- Any workflow where the JSON report is the primary review surface

`--fast` composes with `--dry-run`, parallel Ollama pipelining, and `--max-dimension`.

### Speed vs accuracy (`--max-dimension`)

**Off by default.** Omit the flag (or pass `--max-dimension 0`) to send full-resolution originals — same behavior as before this option existed.

Opt in only when you've measured a speed win on large files and accept the accuracy tradeoffs below. Vision models typically downsample internally (~336–1024 px on the long edge), so `--max-dimension 1024` may shave ~10–30% off 4K+ images with little change for obvious rejects/keepers.

**When not to use this:**

- First pass on a new collection — start at full res until you know where scores land.
- Borderline keepers (roughly threshold ± 1.0) — micro-artifacts (teeth, fingers, hair, small text) may disappear and inflate scores.
- Already-small or upscaled AI images — further downscaling removes the detail that reveals fakeness.
- When JPEG re-encode noise could matter — downscaling re-encodes JPEGs and can add blockiness that looks “AI.”

If you do enable it, dry-run both ways on a sample folder and compare scores before trusting moves. Re-check any borderline keepers at full resolution.

---

## Sample JSON Audit Report (`realism_audit_report.json`)

```json
{
  "meta": {
    "threshold": 7.0,
    "model": "llava",
    "max_dimension": 0,
    "fast": false,
    "generated_at": "2026-07-29T12:00:00+00:00",
    "dry_run": true
  },
  "results": [
    {
      "file": "beach_portrait.jpg",
      "analysis": {
        "realism_score": 8.5,
        "is_realistic": true,
        "detected_artifacts": [
          "AI-generated background texture",
          "Overly uniform skin smoothness"
        ],
        "reasoning": "The image features realistic lighting and natural human posture. However, minor AI artifacts are present in the hair texture and background foliage."
      }
    },
    {
      "file": "broken.jpg",
      "status": "error",
      "error": "Error processing broken.jpg: ...",
      "keep": null
    }
  ]
}
```

Each result may include an optional `keep` field (`true` = never move, `false` = always move). Failed entries require `keep` before apply will move them. Legacy bare-array reports are still accepted on read.

---

## License

MIT
