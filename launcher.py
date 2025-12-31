#!/usr/bin/env python3
"""
Launcher script for Network Routing Optimization System
Provides convenient ways to run the application with different configurations
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
MAIN_FILE = SRC_DIR / "main_integrated.py"

def print_header():
    """Print application header"""
    print("=" * 60)
    print("  Network Routing Optimization System - Launcher")
    print("=" * 60)
    print()

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    required = ['customtkinter', 'networkx', 'matplotlib', 'numpy', 'pandas']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        print(f"\nOr install all requirements:")
        print(f"  pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!\n")
    return True

def run_application(mode="gui"):
    """Run the application"""
    print(f"🚀 Launching application in {mode.upper()} mode...\n")
    
    if not MAIN_FILE.exists():
        print(f"❌ Error: {MAIN_FILE} not found!")
        sys.exit(1)
    
    try:
        subprocess.run([sys.executable, str(MAIN_FILE)], check=False)
    except KeyboardInterrupt:
        print("\n\n👋 Application closed by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

def show_usage():
    """Show usage examples"""
    print("\n📖 USAGE EXAMPLES:\n")
    print("  python launcher.py              → Run GUI")
    print("  python launcher.py --check      → Check dependencies only")
    print("  python launcher.py --help       → Show all options")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Network Routing Optimization System Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                 # Run the GUI application
  python launcher.py --check         # Check if all dependencies installed
  python launcher.py --help          # Show this help message
        """
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check dependencies without running'
    )
    
    parser.add_argument(
        '--no-check',
        action='store_true',
        help='Skip dependency check before running'
    )
    
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )
    
    args = parser.parse_args()
    
    print_header()
    
    # Check version
    if args.version:
        print("Network Routing Optimization System v1.0")
        print("Python:", sys.version.split()[0])
        sys.exit(0)
    
    # Check dependencies
    if not args.no_check:
        if not check_dependencies():
            if not args.check:
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    sys.exit(1)
    
    # If only checking, exit
    if args.check:
        print("✓ Dependency check complete")
        sys.exit(0)
    
    # Run application
    run_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Launcher closed")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
