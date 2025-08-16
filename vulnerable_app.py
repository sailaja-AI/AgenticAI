
import subprocess

def execute_command(command_string):
    """
    Executes a command from user input. This is vulnerable!
    """
    # This line has a security flaw
    subprocess.call(command_string, shell=True)

if __name__ == "__main__":
    print("Executing a 'safe' command.")
    # Example of calling the vulnerable function
    execute_command("echo 'Hello from the app!'")
