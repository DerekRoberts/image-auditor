import argparse
import json
import os
import shutil
from pathlib import Path
from pydantic import BaseModel
import ollama

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
    parser.add_argument("--threshold", type=float, default=7.0, help="Minimum floating point realism score (1.0 to 10.0) to keep image in-place")
    parser.add_argument("--dry-run", action="store_true", help="Generate the JSON report without physically moving files into filter_dir")
    parser.add_argument("--report-path-display", default=None, help="Custom path string to display in the final report message")
    return parser.parse_args()

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
            print(f"Warning: Failed to check model '{model_name}': {e}")
    except Exception as e:
        print(f"Warning: Failed to check or pull model '{model_name}': {e}")

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

def main():
    args = parse_args()
    
    active_model = args.model
    ensure_model(active_model)
    
    input_dir = Path(args.dir)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    filter_dir = Path(args.filter_dir) if args.filter_dir else input_dir / "rejects"
    
    if not args.dry_run:
        filter_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in supported_extensions and p.is_file()]

    results = []

    for img_path in image_paths:
        print(f"Analyzing {img_path.name}...")
        try:
            analysis_dict = analyze_image(img_path, active_model)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

        try:
            analysis = RealismAnalysis(**analysis_dict)
            
            result = {
                "file": img_path.name,
                "analysis": analysis_dict
            }
            results.append(result)
            
            print(f"  Score: {analysis.realism_score} - Realistic: {analysis.is_realistic}")
            
            if not args.dry_run:
                # Keepers stay in place in input_dir; rejects get moved out to filter_dir
                if analysis.realism_score < args.threshold:
                    shutil.move(str(img_path), str(filter_dir / img_path.name))
                    print(f"  -> Moved filtered image to {filter_dir / img_path.name}")
                else:
                    print("  -> Preserved keeper in place")
                
        except Exception as e:
            print(f"Error parsing analysis output for {img_path.name}: {e}")

    results.sort(key=lambda x: x["analysis"]["realism_score"], reverse=True)
    
    report_path = input_dir / "realism_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    display_report_path = args.report_path_display or str(report_path)
    print(f"\nAudit complete! Processed {len(image_paths)} images.")
    print(f"Report saved to {display_report_path}")

if __name__ == "__main__":
    main()
