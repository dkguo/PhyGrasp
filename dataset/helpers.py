def count_category(dataset):
    category = {}
    for obj in dataset.data_entries.values():
        num = len(obj.data)
        category[obj.name] = category.get(obj.name, 0) + num
    return category

def get_languages(dataset):
    languages = []
    for obj in dataset.data_entries.values():
        for entry in obj.data.values():
            languages.append(entry.language)
    return languages