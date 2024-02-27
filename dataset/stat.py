from dataset.Dataset import Dataset
from dataset.category import OBJECTS
import pickle

OBJECT_COUNTS = {key: 0 for key in OBJECTS}
assert len(OBJECTS) == len(OBJECT_COUNTS)
MATERIALS_COUNT = {}
material_path = './dataset/material_count.pkl'
object_path = './dataset/object_count.pkl'

dataset = Dataset('./data/objects')
objs = dataset.get_object_ids()
for i, obj_id in enumerate(objs):
    name = dataset[obj_id].name
    dataset[obj_id].load("_v1")
    OBJECT_COUNTS[name] += len(dataset[obj_id].data)
    for config_id, entry in dataset[obj_id].data.items():
        for material in entry.config.materials:
            MATERIALS_COUNT[material] = MATERIALS_COUNT.get(material, 0) + 1
    dataset[obj_id].unload()
    if i % 100 == 0:
        print(f'Processed {i}/{len(objs)} objects')
        print("OBJECT_COUNTS", OBJECT_COUNTS)
        print("MATERIALS_COUNT", MATERIALS_COUNT)
pickle.dump(OBJECT_COUNTS, open(object_path, 'wb'))
pickle.dump(MATERIALS_COUNT, open(material_path, 'wb'))
        