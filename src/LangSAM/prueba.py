import os
setup_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'setup.py')
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))