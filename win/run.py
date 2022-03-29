import subprocess
import os

path = os.path.join(os.getcwd(), "win", "run.bat")
print(path)
subprocess.call([path])