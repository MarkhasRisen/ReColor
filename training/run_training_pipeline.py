"""Run complete training pipeline: dataset generation + centroid computation."""
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from generate_dataset import generate_dataset
from compute_centroids import main as compute_centroids_main


def main():
    """Execute full training pipeline."""
    print("=" * 80)
    print("🎨 ADAPTIVE COLOR CORRECTION - TRAINING PIPELINE")
    print("=" * 80)
    print()
    
    # Step 1: Generate dataset
    print("STEP 1: Dataset Generation")
    print("-" * 80)
    dataset_dir = Path(__file__).parent / "datasets" / "color_varied"
    
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        response = input(f"Dataset exists at {dataset_dir}. Regenerate? (y/N): ")
        if response.lower() != 'y':
            print("⏭️  Skipping dataset generation...")
        else:
            generate_dataset(dataset_dir)
    else:
        generate_dataset(dataset_dir)
    
    print()
    
    # Step 2: Compute centroids
    print("STEP 2: Centroid Computation")
    print("-" * 80)
    compute_centroids_main()
    
    print()
    print("=" * 80)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print()
    print("📊 Next Steps:")
    print("  1. Copy centroids to backend: training/models/centroids/ → backend/models/centroids/")
    print("  2. Run backend tests: cd backend && pytest")
    print("  3. Test with real images via /process/ endpoint")
    print()


if __name__ == "__main__":
    main()
