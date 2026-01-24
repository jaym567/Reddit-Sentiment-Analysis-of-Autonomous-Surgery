import subprocess
import sys
import os
import time

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"🚀 STAGE: {description}")
    print(f"Running: {script_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Use sys.executable to ensure we use the same python path
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=False)
        duration = time.time() - start_time
        print(f"\n✅ SUCCESS: {description} completed in {duration:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {description} failed with exit code {e.returncode}")
        return False

def main():
    scripts = [
        ("CollectDataAsJSON_timeframe.py", "Scraping Reddit (Temporal 2020-now)", "reddit_robotic_surgery_temporal_flat.json"),
        ("apply_relevance_filter.py", "Applying Relevance Filters", None),
        ("compare_filtering.py", "Generating Relevance Audit/Rejection Log", None),
        ("run_analysis_on_filtered.py", "GPU Sentiment & Aspect Analysis", None),
        ("view_flattened_sentiment.py", "Flattening Results to CSV", None),
        ("create_figures.py", "Generating 13 Research Visualizations", None)
    ]
    
    pipeline_start = time.time()
    for script, desc, output_check in scripts:
        if output_check and os.path.exists(output_check):
            print(f"\n⏩ SKIPPING: {desc} (Output {output_check} already exists)")
            continue
            
        if not run_script(script, desc):
            print("\n🚨 Pipeline halted due to error.")
            sys.exit(1)
            
    total_time = time.time() - pipeline_start
    print(f"\n{'#'*60}")
    print(f"🎊 FULL PIPELINE COMPLETE! Total Time: {total_time/60:.2f} minutes")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()
