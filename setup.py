import os
import sys
import runpy

def run_script(script_path, script_args):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.argv = [script_path] + script_args
    runpy.run_path(script_path, run_name="__main__")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup.py <script_path> [script_args...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    run_script(script_path, script_args)