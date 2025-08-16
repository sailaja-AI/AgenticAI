import subprocess

# This is a safe function that should NOT be flagged.
def run_safe_command():
    # Using a list of arguments is the recommended, safe practice.
    subprocess.call(["ls", "-l"])
