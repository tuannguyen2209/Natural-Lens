import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(10, 6))
ax =  plt.axes()
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, label='Sine Wave')
plt.show()