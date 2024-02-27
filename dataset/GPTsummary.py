import os
import openai
import argparse
import json
import numpy as np
import random
import dataset.prompt2 as PROMPT
from dataset.Dataset import Dataset, Config, DataEntry, ObjectEntry
import dataset.generate_data as generate_data
from dataset.category import OBJECTS
import time

openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
deployment_name='gpt35' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

PARTS_FILE = 'parts.json'
MATERIALS_FILE = './dataset/materials.json'
FRAGILIGY_LEVELS = {1: 'very fragile', 2: 'fragile', 3: 'normal', 4: 'tough', 5: 'very tough'}


def get_parts_names(object: ObjectEntry):
   parts_path = os.path.join(object.object_dir, PARTS_FILE)
   with open(parts_path) as f:
    parts = json.load(f)
   object_name = parts['name']
   parts_info = parts['parts']
   parts_name = [part['name'] for part in parts_info.values()]
   return object_name, parts_name
  
def get_materials():
  materials_path = MATERIALS_FILE
  with open(materials_path) as f:
    materials = json.load(f)
  return materials

def get_random_material(materials, n=1):
  raise NotImplementedError
  material = []
  for i in range(n):
    material.append(random.choice(list(materials)))
  material_names = ', '.join([m['Material'] for m in material])
  material_frictions = ', '.join([str(m['Friction']) for m in material])
  material_density = ', '.join([str(m['Density']) for m in material])
  material_fragility = ', '.join([str(m['Fragility']) for m in material])
  material_grasp_prob = ', '.join([random.choice(['0.1', '0.5', '0.9']) for m in material])
  return material_names, material_frictions, material_density, material_fragility

def get_config_materials(config: Config, available_materials):
  material_names = ', '.join(m for m in config.materials)
  material_frictions = ', '.join([str(m) for m in config.frictions])
  material_density = ', '.join([str(m) for m in config.densities])
  material_fragility = ', '.join([FRAGILIGY_LEVELS[m] for m in config.fragilities])
  sample_probs = " The grasping probabilities of each part are " + ', '.join([str(round(m, 2)) for m in config.sample_probs]) + '.' if config.grasp_likelihoods is not None else ""

  return material_names, material_frictions, material_density, material_fragility, sample_probs

def summary(config: Config, object: ObjectEntry, available_materials, params=None):
  object_name, parts_name = get_parts_names(object)
  all_parts_name = ', '.join(parts_name)
  if available_materials is None:
    available_materials = get_materials(params)
  material_names, material_frictions, material_density, material_fragility, sample_probs = get_config_materials(config, available_materials)
  description = "There is an %s, it has several parts including %s. The materials of each part are %s, with friction: %s, density: %s, fragility: %s.%s " % (object_name, all_parts_name, material_names, material_frictions, material_density, material_fragility, sample_probs)

  completion = openai.ChatCompletion.create(
    engine=deployment_name,
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": PROMPT.ROLE},

      {"role": "user", "content": PROMPT.EXAMPLES},

      {"role": "user", "content": PROMPT.INSTRUCTION + description},
    ],
  )
  response = completion.choices[0].message['content']

  if params is not None and params['debug']:
    # print(object_name, all_parts_name)
    # print(material_names, material_frictions, material_density, material_fragility)
    print("Before summary:\n", description, '\n')
    print("Attempt summary:\n", response, '\n')
    print('-' * 80 + '\n')

  return response

if __name__ == '__main__':
    # usage: under folder DualArmManipulation, run python dataset/GPTsummary_demo.py 
    args = argparse.ArgumentParser()    
    args.add_argument('--id', type=str, default='38957')
    args.add_argument('--debug', default=False, action='store_true')
    args.add_argument('--data_folder', type=str, default='./data/objects')
    args.add_argument('--parts_file', type=str, default='parts.json')
    args.add_argument('--material_file', type=str, default='./dataset/materials.json')
    params = args.parse_args()
    params = vars(params)

    dataset = Dataset('./data/objects')
    available_materials = json.load(open('./dataset/materials.json', 'r'))

    subdirs = [x for x in os.listdir(params['data_folder']) if os.path.isdir(os.path.join(params['data_folder'], x))]
    object_hit = {}
    to_hit = len(OBJECTS)
    for obj in OBJECTS:
      object_hit[obj] = 0

    while to_hit > 0:
      subdir = random.choice(subdirs)
      params['id'] = subdir
      object_name = subdir
      with open(os.path.join(params['data_folder'], subdir, params['parts_file']), 'r') as f:
        parts = json.load(f)
        if object_hit[parts['name']] > 0:
          continue
        else:
          object_hit[parts['name']] += 1
          to_hit -= 1

      meshes = dataset[object_name].load_meshes()
      config = generate_data.generate_random_config(meshes, available_materials)
      start_time = time.time()
      summary(config, dataset[object_name], available_materials, params)
      print("Time used: ", time.time() - start_time)

    # object_name = '1536'
    # meshes = dataset[object_name].load_meshes()
    # config = generate_random_config(meshes, available_materials)
    # summary(config, dataset[object_name], available_materials, params)



