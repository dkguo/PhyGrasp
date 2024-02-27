import pickle

from encoder.LLaMA27BEncoder import LLaMA27BEncoder

if __name__ == '__main__':
    # conda activate bi
    # torchrun --nproc_per_node 1 -m demo.demo_extract_language_feature

    objects_path = '/home/gdk/Repositories/DualArmManipulation/demo/demo_objects'

    objects = [
        {
            "name": "banana",
            "simple_description": "This is a banana.",
            "detailed_description": "The object consists of two parts: the cap and the middle part"
                                    "The object is very ripe, indicating it is very fragile to grasp on the middle part. "
                                    "The cap is easier to grasp."
                                    "I will recommended to grasp the cap."
        },
        {
            "name": "monitor",
            "simple_description": "There is a monitor.",
            "detailed_description": "The display consists of two parts: the display screen and the base. "
                                    "The display screen is made of glass, indicating it is very fragile compared to the base which is made of iron. "
                                    "The base has a higher density compared to the display screen. The base also has a higher friction compared to the display screen. "
                                    "I will recommended to grasp the base."

        },
        {
            "name": "pill_bottle",
            "simple_description": "There is a bottle.",
            "detailed_description": "There is a pill bottle. It consists of two parts: the cap and the bottle body. "
                                    "The bottle body is made of plastic, indicating they have low friction and harder to grasp."
                                    "The cap is made of rubber, which means it has high friction. "
                                    "I would recommend you to grasp the cap on the top."
        },
        {
            "name": "plastic_hammer",
            "simple_description": "There is an object.",
            "detailed_description": "The object consists of two parts: the handle and the base. "
                                    "The handle is made of plastic, indicating it is light and smooth and low friction to grasp. "
                                    "The base is made of metal, indicating the center of mass is near the base. "
                                    "I would recommend to grasp the base."
        },
        {
            "name": "hammer",
            "simple_description": "There is a hammer.",
            "detailed_description": "There is a hammer. It consists of two parts: the handle and the head. "
                                    "The handle is made of plastic, indicating it is light and has low density. "
                                    "The head is made of metal, indicating it is heavy. "
                                    "According to the grasp probabilities, it is recommended to grasp the head."
        }
    ]

    # language feature
    encoder = LLaMA27BEncoder()
    for obj in objects[4:]:
        object_name = obj['name']
        description_simple = obj['simple_description']
        description_detailed = obj['detailed_description']

        encoded_text = encoder.encode(description_simple, layer_nums=[15, 20, 25])
        language_feature_20 = encoded_text[1]
        pickle.dump(language_feature_20, open(f'{objects_path}/{object_name}/simple_language_feature_20.pkl', 'wb'))

        encoded_text = encoder.encode(description_detailed, layer_nums=[15, 20, 25])
        language_feature_20 = encoded_text[1]
        pickle.dump(language_feature_20, open(f'{objects_path}/{object_name}/detailed_language_feature_20.pkl', 'wb'))
