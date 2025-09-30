import os
import subprocess
import sys

def run_all_scripts():
    """
    Finds and runs all Python scripts (.py) in the current directory,
    except for itself.
    """
    # Get the name of the current script to avoid running it
    current_script = os.path.basename(__file__)
    
    # Get a list of all files in the current directory
    try:
        all_files = os.listdir('.')
    except OSError as e:
        print(f"Error: Cannot access directory. {e}")
        return

    # Filter for Python scripts, excluding the current one
    python_scripts = [f for f in all_files if f.endswith('.py') and f != current_script]

    if not python_scripts:
        print("No other Python scripts found to run in this directory.")
        return

    print(f"Found scripts to run: {', '.join(python_scripts)}\n")

    # Loop through and run each script
    for script in sorted(python_scripts):
        print(f"--- Running '{script}' ---")
        try:
            # Use the same python executable that is running this script
            # check=True will raise a CalledProcessError if the script returns a non-zero exit code
            subprocess.run([sys.executable, script], check=True)
            print(f"--- Successfully finished '{script}' ---\n")
        except FileNotFoundError:
            print(f"Error: The script '{script}' was not found.")
        except subprocess.CalledProcessError as e:
            print(f"--- Error running '{script}' ---")
            print(f"--- Script exited with error code {e.returncode} ---\n")
        except Exception as e:
            print(f"An unexpected error occurred while running '{script}': {e}\n")

if __name__ == "__main__":
    run_all_scripts()