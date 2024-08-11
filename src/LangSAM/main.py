import subprocess
import os
import sys
def main():
    setup_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    try:
            subprocess.check_call(['pip', setup_path, 'install', '-q', '-e', '.'],cwd=os.path.dirname(setup_path))
            subprocess.check_call(['python','-c','import torch'])
            #subprocess.check_call(['python', '-c', 'numpy'])
    except subprocess.CalledProcessError as e:
            print(f"Error durante la construcción: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()

