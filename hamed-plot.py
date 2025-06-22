import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel('./training_history_values_et.xlsx')
print(df.head())


iteration = df['Epoch']                 
training = df['Train RMSE']            
testing = df['Validation RMSE']        

plt.figure(figsize=(10, 6))
plt.plot(iteration, training, label='Training', 
         linestyle='-', linewidth=2, color='blue')
plt.plot(iteration, testing, label='Testing', 
         linestyle='-', linewidth=2, color='orange')


plt.xlim(1, 50)


min_y = 0.04
max_y = max(training.max(), testing.max())
if max_y <= min_y:
    max_y = min_y + 0.05
else:
    max_y = max_y * 1.05   
plt.ylim(min_y, max_y)


ax = plt.gca()
ax.set_xmargin(0)
ax.set_ymargin(0)


plt.xlabel('Iteration', fontsize=13)
plt.ylabel('RMSE (mm/day)', fontsize=13)


plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()


plt.savefig("./curve_et.png", dpi=300)
plt.show()
print('hi')





import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel('./training_history_values_et.xlsx')
print(df.head())


iteration = df['Epoch']                 
training = df['r2_keras']            
testing = df['val_r2_keras']        

plt.figure(figsize=(10, 6))
plt.plot(iteration, training, label='Training', 
         linestyle='-', linewidth=2, color='blue')
plt.plot(iteration, testing, label='Testing', 
         linestyle='-', linewidth=2, color='orange')


plt.xlim(1, 50)


min_y = 0.85
max_y = max(training.max(), testing.max())
if max_y <= min_y:
    max_y = min_y + 0.02
else:
    max_y = max_y * 1.02   
plt.ylim(min_y, max_y)


ax = plt.gca()
ax.set_xmargin(0)
ax.set_ymargin(0)


plt.xticks([1, 10, 20, 30, 40, 50])


plt.xlabel('Iteration', fontsize=13)
plt.ylabel('R²', fontsize=13)


plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()


plt.savefig("./curve_et_r2.png", dpi=300)
plt.show()
print('hi')
