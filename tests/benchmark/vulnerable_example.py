import subprocess

# This is a vulnerable function that should be flagged.
def run_dynamic_command(cmd_str):
    subprocess.call(cmd_str, shell=True)
