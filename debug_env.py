"""Debug .env file loading."""
from pathlib import Path

# Show where we're looking
backend_dir = Path(__file__).parent / "backend"
env_path = backend_dir / ".env"

print(f"Current file: {__file__}")
print(f"Backend dir: {backend_dir}")
print(f"Env path: {env_path}")
print(f"Env exists: {env_path.exists()}")

if env_path.exists():
    print("\n.env contents:")
    with open(env_path, 'r') as f:
        print(f.read())
