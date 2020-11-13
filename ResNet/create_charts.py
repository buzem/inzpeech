import re

import matplotlib.pyplot as plt

file_name = 'out_17_10_2020.txt'

# Save loss fig
pattern = re.compile(".+loss: ([0-9].[0-9]+)")

loss_y = [re.match(pattern, line).group(1) if re.match(
    pattern, line) else None for line in open(file_name)]
loss_y = [float(x) for x in loss_y if x is not None]
loss_x = [x for x in range(len(loss_y))]

plt.figure(figsize=(18, 12))
plt.plot(loss_x, loss_y)
plt.title('Model Train Loss Chart')
plt.xlabel('# of iterations')
plt.ylabel('Loss')
plt.savefig('loss_{}.png'.format(file_name))

# Save acc fig
val_pattern = re.compile("Val Acc: ([0-9]+.[0-9]+)")
train_pattern = re.compile("Train Acc: ([0-9]+.[0-9]+)")

train_acc_y = [re.match(train_pattern, line).group(1) if re.match(
    train_pattern, line) else None for line in open(file_name)]
train_acc_y = [float(x) for x in train_acc_y if x is not None]

val_acc_y = [re.match(val_pattern, line).group(1) if re.match(
    val_pattern, line) else None for line in open(file_name)]
val_acc_y = [float(x) for x in val_acc_y if x is not None]

acc_x = [x for x in range(len(train_acc_y))]

plt.figure(figsize=(18, 12))
plt.plot(acc_x, val_acc_y, label='Val')
plt.plot(acc_x, train_acc_y, label='Train')
plt.title('Model Train Accuracy Chart')
plt.xlabel('# of epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc_{}.png'.format(file_name))
