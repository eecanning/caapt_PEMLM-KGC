def read_triplets_from_txt(file_path):
    triplets = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triplet = (parts[0], parts[1], parts[2])
                triplets.append(triplet)
            else:
                print(f"Ignoring invalid line: {line}")
    return triplets
def read_text_from_txt(file_path):
    dict = {}
    with open(file_path, 'r',encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                dict[parts[0]] = parts[1]
            else:
                print(f"Ignoring invalid line: {line}")
    return dict
def read_entity(file_path):
    entity = {}
    with open(file_path, 'r',encoding='utf-8') as file:
        lines = file.readlines()
        index = 0
        for line in lines:

            parts = line.strip().split('\t')
            if len(parts) == 1:
                entity[parts[0]] = index
            else:
                print(f"Ignoring invalid line: {line}")
            index += 1

    return entity
def count_groundtruth(train_set ,valid_set ,test_set):
    groundtruth = { split: {'tail': {}} for split in ['all', 'train', 'valid', 'test']}

    for triple in train_set:
        h, r, t = triple
        groundtruth['all']['tail'].setdefault((r, h), [])
        groundtruth['all']['tail'][(r, h)].append(t)
        groundtruth['train']['tail'].setdefault((r, h), [])
        groundtruth['train']['tail'][(r, h)].append(t)
    for triple in valid_set:
        h, r, t = triple
        groundtruth['all']['tail'].setdefault((r, h), [])
        groundtruth['all']['tail'][(r, h)].append(t)
        groundtruth['valid']['tail'].setdefault((r, h), [])
        groundtruth['valid']['tail'][(r, h)].append(t)

    for triple in test_set:
        h, r, t = triple
        groundtruth['all']['tail'].setdefault((r, h), [])
        groundtruth['all']['tail'][(r, h)].append(t)
        groundtruth['test']['tail'].setdefault((r, h), [])
        groundtruth['test']['tail'][(r, h)].append(t)
    return groundtruth