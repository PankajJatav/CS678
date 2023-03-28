import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

epochs = [5,10, 15, 20, 25 ]
# create data
dev = [0.480, 0.520, 0.520, 0.520, 0.520]
test = [0.511, 0.519, 0.519, 0.519, 0.519]

# plot lines
plt.plot(epochs, dev, label="Dev Accuracy")
plt.plot(epochs, test, label="Test Accuracy ")

plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
plt.xlabel('Epochs Size', fontweight='bold', fontsize=15)

plt.legend()
plt.savefig('plot/epochs.png')
plt.clf()


batch = [2,4,8,16,32]
dev = [0.492, 0.515, 0.520, 0.510, 0.494]
test = [0.511, 0.517, 0.519, 0.509, 0.507]

# plot lines
plt.plot(batch, dev, label="Dev Accuracy")
plt.plot(batch, test, label="Test Accuracy ")
plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
plt.xlabel('Batch Size', fontweight='bold', fontsize=15)
plt.legend()
plt.savefig('plot/batch.png')
plt.clf()


dropout = [0.1, 0.3, 0.5, 0.7 , 0.9]

dev = [0.512, 0.520, 0.520, 0.509, 0.489]
test = [0.533, 0.519, 0.524, 0.536, 0.511]

# plot lines
plt.plot(dropout, dev, label="Dev Accuracy")
plt.plot(dropout, test, label="Test Accuracy ")
plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
plt.xlabel('Dropout', fontweight='bold', fontsize=15)
plt.legend()
plt.savefig('plot/dropout.png')
plt.clf()

lr = [0.5, 1e-1, 1e-3, 1e-5, 1e-7]

dev = [0.262, 0.262, 0.262, 0.520, 0.243]
test = [0.286, 0.286, 0.286, 0.519, 0.262]


barWidth = 0.25


br1 = np.arange(len(lr))
br2 = [x + barWidth for x in br1]

# plot lines
# plt.plot(br1, dev, label="Dev Accuracy")
# plt.plot(br2, test, label="Test Accuracy ")
# plt.legend()
# plt.savefig('plot/lr_rate.png')
plt.clf()

barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

# Make the plot
plt.bar(br1, dev, color='b', width=barWidth,
        edgecolor='grey', label='Dev Accuracy')
plt.bar(br2, test, color='g', width=barWidth,
        edgecolor='grey', label='Test Accuracy')


# Adding Xticks
plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
plt.xlabel('Learning Rate', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(test))], lr)

plt.legend()
plt.savefig('plot/lr_rate.png')