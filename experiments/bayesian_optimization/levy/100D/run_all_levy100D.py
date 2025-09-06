import os
import subprocess
import sys

# List of scripts to run - modify this list to add/remove scripts
SCRIPTS_TO_RUN = [
    "stgp_bo.py",
    "gp_network_bo.py",
    # "pogpn_pathwise_1.0beta_ELBO_1.0IPRatio_GIR.py",
    "pogpn_pathwise_1.0beta_PLL_1.0IPRatio_GIR.py",
]


def run_script(script_path):
    """Run a single Python script and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {script_path}")
    print(f"{'=' * 60}")

    try:
        # Run the script using subprocess without changing working directory
        # This preserves the relative paths that the scripts expect
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
        )
        print(f"✓ Successfully completed: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error running {script_path}: {e}")
        return False


def main():
    """Main function to run all experiments."""

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Starting Bayesian Optimization Experiments for Levy 100D")
    print(f"Script directory: {script_dir}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Scripts to run: {len(SCRIPTS_TO_RUN)}")

    # Check if all scripts exist in the script directory
    missing_scripts = []
    for script in SCRIPTS_TO_RUN:
        script_path = os.path.join(script_dir, script)
        if not os.path.exists(script_path):
            missing_scripts.append(script)

    if missing_scripts:
        print("\nWarning: The following scripts were not found:")
        for script in missing_scripts:
            print(f"  - {script}")
        print("Please check the script names in SCRIPTS_TO_RUN list.")
        return

    # Run all scripts
    successful_runs = 0
    total_runs = len(SCRIPTS_TO_RUN)

    for i, script in enumerate(SCRIPTS_TO_RUN, 1):
        script_path = os.path.join(script_dir, script)
        print(f"\n[{i}/{total_runs}] Starting script: {script}")
        if run_script(script_path):
            successful_runs += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total scripts: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {total_runs - successful_runs}")

    if successful_runs == total_runs:
        print("\n🎉 All experiments completed successfully!")
    else:
        print(
            f"\n⚠️  {total_runs - successful_runs} experiment(s) failed. Check the output above for details."
        )


if __name__ == "__main__":
    main()
