from encoder.LLaMA27BEncoder import LLaMA27BEncoder
import dataset.Dataset as Dataset
import random
from multiprocessing import Pool

def feature_extraction(dataset):
    encoder = LLaMA27BEncoder()
    for obj in dataset.data_entries.values():
        for entry in obj.data.values():
            entry.language_feature = encoder.encode(entry.language)
        obj.save()

def feature_extraction_objs(objs):
    encoder = LLaMA27BEncoder()
    dataset = Dataset.Dataset('./data/objects')
    for i, obj in enumerate(objs):
        dataset[obj].load("_v1")
        to_save = False
        for entry in dataset[obj].data.values():
            if not hasattr(entry, 'language_feature_20'):
                to_save = True
                encoded_text = encoder.encode(entry.language, layer_nums=[15, 20, 25])
                # entry.language_feature = encoder.encode(entry.language)
                entry.language_feature_15 = encoded_text[0]
                entry.language_feature_20 = encoded_text[1]
                entry.language_feature_25 = encoded_text[2]
        if to_save:
            dataset[obj].save("_v1")
        dataset[obj].unload()
        if i % 10 == 0:
            print(f'{obj}: {i}/{len(objs)}')

def get_objs():
    dataset = Dataset.Dataset('./data/objects')
    objs = dataset.get_object_ids()
    return objs

def main():
    objs = get_objs()
    random.shuffle(objs)
    # tasks = []
    # NUM_PROCESS = 4
    # for i in range(NUM_PROCESS):
    #     tasks.append(objs[i::NUM_PROCESS])
    # pool = Pool(NUM_PROCESS)
    # pool.map(feature_extraction_objs, tasks)
    # pool.close()
    
    feature_extraction_objs(objs)

if __name__ == '__main__':
    '''
    usage: torchrun --nproc_per_node 1 -m encoder.language_encode
    '''
    main()