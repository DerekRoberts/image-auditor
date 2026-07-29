# AI Image Realism Auditor & Sorter

A local CLI tool for auditing and sorting AI-generated images based on photorealism using local vision LLMs via [Ollama](https://ollama.com).

Evaluates image quality, detects AI generation artifacts (e.g. plastic skin, warped fingers, lighting inconsistencies), scores photorealism from 1.0 to 10.0, and automatically sorts files into `keepers/` or `rejects/` folders based on a configurable threshold.

---

## Features

- **Local & Private:** Queries local vision models (`llava`, `llama3.2-vision`, `qwen2.5-vl`) running via Ollama — zero cloud API costs or data leakage.
- **Structured Pydantic Output:** Enforces JSON schema validation for reliable numerical scores, boolean flags, artifact tags, and forensic reasoning.
- **Automated Image Sorting:** Automatically segregates images into `<input_dir>/keepers` or `<input_dir>/rejects`.
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
| `--threshold` | `7.0` | Minimum score (1.0 to 10.0) required to keep image in-place |
| `--dry-run` | `False` | Generate report without moving files |

---

## Sample JSON Audit Report (`realism_audit_report.json`)

```json
[
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
  }
]
```

---

## License

MIT
