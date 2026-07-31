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

### Audit profiles (`--profile`)

Profiles select which analysis lenses run and set default thresholds. Without `--profile`, behavior is unchanged: single AI realism score, legacy `analysis` block in the report, no `meta.profile`.

| Profile | Question | Lenses | Status |
| --- | --- | --- | --- |
| `mixed` (recommended for unknown folders) | What is this and should I keep it? | `ai` (+ `hygiene`, `quality` when #3 lands) | **AI lens working** |
| `ai-fun` | Is this render successful / worth keeping? | `generation` (+ optional `hygiene`) | **Generation lens working** |
| `photos` | Is this a keeper real photo? | `quality` (+ optional `hygiene` when #3 lands) | **Quality lens working** |

**Mixed folder (fully working today):**
```bash
image-auditor ~/Downloads --profile mixed --dry-run
image-auditor ~/Downloads --profile mixed --threshold 7.5
```

Reports use the multi-dimensional layout: `meta.profile`, `meta.thresholds`, and per-file `ai` blocks (not legacy `analysis`).

**AI art folder (`ai-fun` — generation success, not photorealism):**

For intentional AI art, photorealism is the wrong metric. The generation lens scores whether a render *succeeded* and is worth keeping — not whether it looks like a photograph.

| Image | Realism score (wrong lens) | Generation success (right lens) |
| --- | --- | --- |
| Octo-rex mashup, coherent | Low | **High** — creative, readable subjects |
| T-rex with six fingers | Low | **Low** — anatomical failure |
| Melted face / garbled text | Low | **Low** — broken render |
| Stylized but clean cartoon | Low | **High** — if that's the intent |

```bash
image-auditor ~/AI-Art --profile ai-fun --dry-run
image-auditor ~/AI-Art --profile ai-fun --threshold 7.5
image-auditor ~/AI-Art --profile ai-fun --fast --max-dimension 1024 --dry-run
```

Reports use `generation` blocks with `success_score`, `issues[]`, and `reasoning`. Rejects when `success_score` is below `--threshold-generation` (or `--threshold`, which maps to the profile's primary lens). Set `"keep": true` on edge cases after dry-run review to preserve them on `--apply-report`.

**Camera roll / Takeout (keeper quality, not AI detection):**

For real photos, the question is *"Would I keep this in an album?"* — not whether it looks AI-generated. The quality lens scores blur, exposure, framing accidents, and screenshots.

| Image | AI realism (wrong lens) | Keeper quality (right lens) |
| --- | --- | --- |
| Sharp sunset, well exposed | High | **High** — intentional keeper |
| Motion-blurred party shot | High | **Low** — motion_blur |
| Pocket / floor accidental capture | High | **Low** — pocket_shot, accidental_frame |
| Screenshot with UI chrome | High | **Low** — screenshot, ui_chrome |
| Eyes closed on group photo | High | **Low** — eyes_closed |

```bash
image-auditor ~/Pictures --profile photos --dry-run
image-auditor ~/Pictures --profile photos --threshold 6.0
image-auditor ~/Pictures --profile photos --fast --max-dimension 1024 --dry-run
```

Reports use `quality` blocks with `keeper_score`, `issues[]`, and `reasoning`. Rejects when `keeper_score` is below `--threshold-quality` (or `--threshold`, which maps to the profile's primary lens). Common issue tags: `motion_blur`, `out_of_focus`, `underexposed`, `overexposed`, `eyes_closed`, `accidental_frame`, `finger_on_lens`, `pocket_shot`, `floor_shot`, `screenshot`, `ui_chrome`, `duplicate_feel`. Set `"keep": true` on edge cases after dry-run review to preserve them on `--apply-report`.

Override the profile’s lens set (still validates implementation):
```bash
image-auditor ~/Downloads --profile mixed --checks ai --dry-run
```

Per-dimension threshold overrides (defaults come from the profile):
```bash
image-auditor ~/Downloads --profile mixed --threshold 7.5 --threshold-quality 6.0 --dry-run
```

---

## CLI Options

| Flag | Default | Description |
| --- | --- | --- |
| `directory` | `.` | Target input directory containing images (`.png`, `.jpg`, `.jpeg`, `.webp`) |
| `--filter-dir` | `<input_dir>/rejects` | Directory to move filtered-out/rejected files into |
| `--model` | `llava` | Local vision model to query via Ollama |
| `--threshold` | `7.0` (dry-run only) | Minimum score for the profile’s primary lens, or AI realism when no `--profile`; with `--profile`, profile default applies if omitted |
| `--threshold-ai` | — | Override AI realism cutoff (1.0 to 10.0) |
| `--threshold-quality` | — | Override quality keeper cutoff (1.0 to 10.0) |
| `--threshold-generation` | — | Override generation success cutoff (1.0 to 10.0) |
| `--profile` | — | `mixed`, `ai-fun`, or `photos`; omit for legacy single-score mode |
| `--checks` | — | Comma-separated lenses overriding the profile set (`hygiene`, `ai`, `quality`, `generation`) |
| `--dry-run` | `False` | Generate report without moving files |
| `--apply-report` | — | Apply file moves from an existing audit report without re-analyzing (optional path; default: `<input_dir>/realism_audit_report.json`) |
| `--max-dimension` | `0` (off) | Optional: downscale before Ollama so the long edge is at most N px (`0` = send originals; **default**) |
| `--fast` | `False` | Minimal VLM output (score, flag, artifacts only); skips reasoning for faster bulk triage |

### Fast vs full mode (`--fast`)

**Full mode (default)** asks the model for forensic reasoning alongside score, realism flag, and artifact tags. Use this for dry-runs, borderline decisions, and any run where you will read the JSON report before moving files.

**Fast mode** (`--fast`) uses a reduced prompt and schema — score and artifact tags only, no model reasoning. The report still includes a `reasoning` key (empty string) so the format stays consistent; `meta.fast` is `true`. Works for `ai` (realism), `generation` (success), and `quality` (keeper) lenses. Speed gains depend on the model, image size, and hardware.

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

### Single-score mode (default)

Current audits emit one `analysis` block per file plus legacy `meta.threshold`. New reports also record `meta.thresholds` for forward compatibility with profiles.

```json
{
  "meta": {
    "threshold": 7.0,
    "thresholds": { "ai": 7.0, "quality": null, "generation": null },
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

### Multi-dimensional layout (`--profile`)

With `--profile`, each result carries **only the populated dimension blocks** — no empty placeholders. `meta.profile` names the active profile; `meta.thresholds` holds per-dimension cutoffs (`null` = lens inactive for this run).

| Block | Purpose | Key fields |
| --- | --- | --- |
| `hygiene` | Deterministic pre-filter (#2, #3): dupes, corruption, min-res | `action` (`reject` \| `keep`), optional `exact_dupe_of` |
| `ai` | Photorealism / AI-artifact detection | `realism_score`, `issues[]`, `reasoning` |
| `quality` | Real-photo keeper scoring (blur, exposure, framing) | `keeper_score`, `issues[]`, `reasoning` |
| `generation` | AI-art success (subject landed, not melted) | `success_score`, `issues[]`, `reasoning` |

```json
{
  "meta": {
    "profile": "mixed",
    "threshold": 7.0,
    "thresholds": { "ai": 7.0, "quality": 6.0, "generation": null },
    "model": "llava",
    "max_dimension": 0,
    "fast": false,
    "generated_at": "2026-07-29T12:00:00+00:00",
    "dry_run": true
  },
  "results": [
    {
      "file": "beach_portrait.jpg",
      "ai": {
        "realism_score": 8.5,
        "is_realistic": true,
        "issues": [
          "AI-generated background texture",
          "Overly uniform skin smoothness"
        ],
        "reasoning": "The image features realistic lighting and natural human posture."
      }
    }
  ]
}
```

Example with multiple lenses once hygiene/quality/generation ship (`ai-fun` / `photos`):

```json
{
  "meta": {
    "profile": "ai-fun",
    "thresholds": { "ai": null, "quality": 6.0, "generation": 7.0 },
    "generated_at": "2026-07-29T12:00:00+00:00",
    "dry_run": true
  },
  "results": [
    {
      "file": "octo-rex.png",
      "generation": {
        "success_score": 8.5,
        "issues": [],
        "reasoning": "Creative mashup, coherent subjects, no garbled anatomy."
      },
      "keep": null
    },
    {
      "file": "IMG_blur.jpg",
      "quality": {
        "keeper_score": 3.2,
        "issues": ["motion_blur", "underexposed"],
        "reasoning": "Subject motion blur and heavy underexposure."
      },
      "keep": null
    },
    {
      "file": "copy.jpg",
      "hygiene": {
        "exact_dupe_of": "original.jpg",
        "action": "reject"
      }
    }
  ]
}
```

**Backward compatibility:** reports with top-level `analysis` / `realism_score` still load and apply using `meta.threshold` (or `meta.thresholds.ai`). Multi-dimensional `ai` and `generation` blocks use per-dimension thresholds from `meta.thresholds`. Composite reject policy across multiple scored lenses is defined in a follow-up issue; until then, `--apply-report` rejects when any scored dimension with an active threshold falls below its cutoff, and honors `hygiene.action` plus `keep` overrides.

---

## License

MIT
