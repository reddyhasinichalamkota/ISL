import argparse
import sys
import os

# Add workspace directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.camera.camera import run_camera_test
from src.inference.realtime_inference import run_realtime_application
from src.dataset.dataset_validator import inspect_dataset_directory
from src.utils.logger import get_logger

logger = get_logger("MainAppLauncher")

def display_interactive_menu():
    print("\n=======================================================")
    print("   REAL-TIME INDIAN SIGN LANGUAGE RECOGNITION SYSTEM   ")
    print("=======================================================")
    print("Select Operational Mode:")
    print("  1. Test Laptop Camera Hardware (OpenCV Test)")
    print("  2. Run Static Sign Recognition (Live OpenCV Webcam)")
    print("  3. Run Dynamic Sign Recognition (Live OpenCV Webcam)")
    print("  4. Inspect Static Image Dataset (Dataset/ Directory)")
    print("  Q. Quit Application")
    print("=======================================================")
    
    choice = input("Enter option [1-4 or Q]: ").strip().upper()
    return choice

def main():
    parser = argparse.ArgumentParser(description="Real-Time ISL Recognition Launcher")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["camera_test", "static", "dynamic", "inspect"],
        help="Operating mode: camera_test | static | dynamic | inspect"
    )
    args = parser.parse_args()

    mode = args.mode

    if not mode:
        choice = display_interactive_menu()
        if choice == "1":
            mode = "camera_test"
        elif choice == "2":
            mode = "static"
        elif choice == "3":
            mode = "dynamic"
        elif choice == "4":
            mode = "inspect"
        elif choice == "Q":
            print("Exiting ISL Recognition Application. Goodbye!")
            return
        else:
            print("Invalid option selected. Exiting.")
            return

    logger.info(f"Launching ISL Recognition System in mode: {mode.upper()}")

    if mode == "camera_test":
        run_camera_test()
    elif mode == "static":
        run_realtime_application(mode="STATIC")
    elif mode == "dynamic":
        run_realtime_application(mode="DYNAMIC")
    elif mode == "inspect":
        target_dir = "Dataset/Training" if os.path.exists("Dataset/Training") else "Dataset"
        if os.path.exists(target_dir):
            inspect_dataset_directory(target_dir)
        else:
            print(f"[ERROR] Dataset directory '{target_dir}' not found.")

if __name__ == "__main__":
    main()
