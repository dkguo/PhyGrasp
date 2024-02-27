CATEGORY = {'table': 9550, 'trash_can': 357, 'chair': 8097, 'pot': 579, 'faucet': 753, 'storage_furniture': 2201, 'door_set': 210, 'display': 933, 'lamp': 2967, 'keyboard': 103, 'refrigerator': 159, 'bottle': 474, 'hat': 119, 'earphone': 260, 'laptop': 223, 'bed': 181, 'cutting_instrument': 498, 'clock': 156, 'bowl': 104, 'bag': 83, 'scissors': 117, 'mug': 213, 'dishwasher': 190, 'microwave': 72}

OBJECTS = ['table', 'trash_can', 'chair', 'pot', 'faucet', 'storage_furniture', 'door_set', 'display', 'lamp', 'keyboard', 'refrigerator', 'bottle', 'hat', 'earphone', 'laptop', 'bed', 'cutting_instrument', 'clock', 'bowl', 'bag', 'scissors', 'mug', 'dishwasher', 'microwave']

SCALES = {'table': 1.0, 'trash_can': 0.2, 'chair': 0.6, 'pot': 0.2, 'faucet': 0.1, 'storage_furniture': 1.5, 'door_set': 1.4, 'display': 0.3, 'lamp': 0.2, 'keyboard': 0.3, 'refrigerator': 1.2, 'bottle': 0.1, 'hat': 0.1, 'earphone': 0.01, 'laptop': 0.3, 'bed': 1.6, 'cutting_instrument': 0.1, 'clock': 0.2, 'bowl': 0.1, 'bag': 0.3, 'scissors': 0.1, 'mug': 0.1, 'dishwasher': 1.1, 'microwave': 0.7}

SAMPLES = {'table': 1, 'trash_can': 29, 'chair': 1, 'pot': 17, 'faucet': 13, 'storage_furniture': 4, 'door_set': 49, 'display': 11, 'lamp': 3, 'keyboard': 101, 'refrigerator': 65, 'bottle': 21, 'hat': 87, 'earphone': 40, 'laptop': 46, 'bed': 57, 'cutting_instrument': 20, 'clock': 66, 'bowl': 100, 'bag': 125, 'scissors': 89, 'mug': 48, 'dishwasher': 54, 'microwave': 144}

SAMPLES_TEST = {'table': 0, 'trash_can': 1, 'chair': 0, 'pot': 1, 'faucet': 1, 'storage_furniture': 1, 'door_set': 1, 'display': 1, 'lamp': 1, 'keyboard': 1, 'refrigerator': 1, 'bottle': 1, 'hat': 1, 'earphone': 1, 'laptop': 1, 'bed': 1, 'cutting_instrument': 1, 'clock': 1, 'bowl': 1, 'bag': 1, 'scissors': 1, 'mug': 1, 'dishwasher': 1, 'microwave': 1}



if __name__ == '__main__':
    # total = 250000
    test_total = 0
    for k, v in SAMPLES_TEST.items():
        test_total += v * CATEGORY[k]
    print(test_total)
