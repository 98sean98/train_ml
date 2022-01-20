import os
import numpy as np

a = np.random.rand(4,4)

print(a)

script_dir = os.path.dirname(__file__)
output_path = os.path.join(script_dir, 'artifacts/output.txt')

np.savetxt(output_path, a, fmt='%4.4f')
