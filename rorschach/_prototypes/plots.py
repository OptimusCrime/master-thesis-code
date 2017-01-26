
import numpy as np
from matplotlib import pyplot as plt

from rorschach.utilities import Filesystem

epochs = 10

data = {
    'loss': [0, 300, 215, 111, 213, 94, 12, 6, 5, 8, 10],
    'val_loss': [0, 311, 99, 55, 43, 31, 75, 44, 12, 75, 74],
    'acc': [0, 0.3, 0.34, 0.5, 0.6, 0.44, 0.32, 0.66, 0.77, 0.88, 0.975],
    'val_acc': [0, 0.1, 0.11, 0.23, 0.33, 0.35, 0.47, 0.31, 0.55, 0.67, 0.87]
}

fig = plt.figure(figsize=(16, 6), dpi=80)

# Subplots
ax_loss = fig.add_subplot(121)
ax_acc = fig.add_subplot(122)

# Add plots
ax_loss.plot(data['loss'], label="loss")
ax_loss.plot(data['val_loss'], label="val_loss")

ax_acc.plot(data['acc'], label="acc")
ax_acc.plot(data['val_acc'], label="val_acc")

# Set labels and titles
ax_loss.set_title('loss')
ax_loss.set_ylabel('loss')
ax_loss.set_xlabel('epochs')

ax_acc.set_title('accuracy')
ax_acc.set_ylabel('accuracy')
ax_acc.set_xlabel('epochs')

# Ticks
ax_loss.minorticks_on()
ax_loss.tick_params(labeltop=False, labelright=True)

ax_acc.minorticks_on()
ax_acc.tick_params(labeltop=False, labelright=True)

# Set x limit and ticks
ax_loss.set_xlim(1, epochs)
ax_loss.set_xticks(np.arange(1, epochs + 1))

ax_acc.set_xlim(1, epochs)
ax_acc.set_xticks(np.arange(1, epochs + 1))

# Fix legend below the graph
box_loss = ax_loss.get_position()
ax_loss.set_position([box_loss.x0 - box_loss.width * 0.12, # Move to the left
                      box_loss.y0 + box_loss.height * 0.12,
                      box_loss.width,
                      box_loss.height * 0.88])

box_acc = ax_acc.get_position()
ax_acc.set_position([box_acc.x0 + box_acc.width * 0.1, # Move to the right
                     box_acc.y0 + box_acc.height * 0.12,
                     box_acc.width,
                     box_acc.height * 0.88])

ax_loss.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
               fancybox=True, shadow=True, ncol=5)

ax_acc.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
              fancybox=True, shadow=True, ncol=5)

fig.savefig(Filesystem.get_root_path('data/test123.png'))
