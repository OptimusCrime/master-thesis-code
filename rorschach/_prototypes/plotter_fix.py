import re

REGEX_ACC_PATTERN = re.compile('^activation_(?:[0-9]*)_acc')
REGEX_VAL_ACC_PATTERN = re.compile('^val_activation_(?:[0-9]*)_acc')


def clean_logs(logs):
    new_logs = dict()
    if 'loss' in logs:
        new_logs['loss'] = logs['loss']

    if 'val_loss' in logs:
        new_logs['val_loss'] = logs['val_loss']

    average_acc = []
    average_val_acc = []
    for key, value in logs.items():
        if REGEX_ACC_PATTERN.match(key):
            average_acc.append(value)

        if REGEX_VAL_ACC_PATTERN.match(key):
            average_val_acc.append(value)

    if len(average_acc) > 0:
        new_logs['acc'] = sum(average_acc) / float(len(average_acc))

    if len(average_val_acc) > 0:
        new_logs['val_acc'] = sum(average_val_acc) / float(len(average_val_acc))

    return new_logs

log = {
    'loss': 21.4287,
    'activation_1_loss': 0.8317,
    'activation_2_loss': 1.5459,
    'activation_3_loss': 2.0660,
    'activation_4_loss': 2.3131,
    'activation_5_loss': 2.4161,
    'activation_6_loss': 2.4537,
    'activation_7_loss': 2.4917,
    'activation_8_loss': 2.4900,
    'activation_9_loss': 2.2427,
    'activation_10_loss': 2.5778,
    'activation_1_acc': 0.8264,
    'activation_2_acc': 0.6122,
    'activation_3_acc': 0.3988,
    'activation_4_acc': 0.2276,
    'activation_5_acc': 0.2742,
    'activation_6_acc': 0.2712,
    'activation_7_acc': 0.2690,
    'activation_8_acc': 0.2542,
    'activation_9_acc': 0.1966,
    'activation_10_acc': 0.2064,
    'val_loss': 21.1110,
    'val_activation_1_loss': 0.7585,
    'val_activation_2_loss': 1.4730,
    'val_activation_3_loss': 2.0178,
    'val_activation_4_loss': 2.3357,
    'val_activation_5_loss': 2.3890,
    'val_activation_6_loss': 2.4030,
    'val_activation_7_loss': 2.4465,
    'val_activation_8_loss': 2.5132,
    'val_activation_9_loss': 2.2355,
    'val_activation_10_loss': 2.5386,
    'val_activation_1_acc': 0.8420,
    'val_activation_2_acc': 0.6340,
    'val_activation_3_acc': 0.4140,
    'val_activation_4_acc': 0.2220,
    'val_activation_5_acc': 0.2760,
    'val_activation_6_acc': 0.3080,
    'val_activation_7_acc': 0.2760,
    'val_activation_8_acc': 0.2520,
    'val_activation_9_acc': 0.2100,
    'val_activation_10_acc': 0.2380
}

new_logs = clean_logs(log)

print(new_logs)
