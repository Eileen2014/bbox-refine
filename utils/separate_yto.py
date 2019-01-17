import numpy as np
import os
import sys
import pickle


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


VOC_CLASSES = (
               'aeroplane',
               'bicycle',
               'bird',
               'boat',
               'bottle',
               'bus',
               'car',
               'cat',
               'chair',
               'cow',
               'diningtable',
               'dog',
               'horse',
               'motorbike',
               'person',
               'pottedplant',
               'sheep',
               'sofa',
               'train',
               'tvmonitor')

YTO_CLASSES = ('aeroplane',
               'bird',
               'boat',
               'car',
               'cat',
               'cow',
               'dog',
               'horse',
               'motorbike',
               'train'
               )

IMG_SET_DIR = '/home/avisek/kv/datasets/VOCdevkit/VOC2012/ImageSets/Main'
XML_ROOT = '/home/avisek/kv/datasets/VOCdevkit/VOC2012/Annotations'
NEW_XML_ROOT = '/home/avisek/kv/datasets/VOCdevkit/YTO_VOC2012/Annotations'
class_wise_files = {}
VOC='VOC2012'

if not os.path.exists(NEW_XML_ROOT):
   os.mkdir(NEW_XML_ROOT)

def create_tree(object, filename):
   annotation = ET.Element("annotation")

   ET.SubElement(annotation, "filename").text = object['filename']
   ET.SubElement(annotation, "folder").text = object['folder']
   
   _object = ET.SubElement(annotation, "object")

   ET.SubElement(_object, "name").text = object["name"]
   bndbox = ET.SubElement(_object, "bndbox")
   ET.SubElement(bndbox, "xmin").text = str(object["xmin"])
   ET.SubElement(bndbox, "ymin").text = str(object["ymin"])
   ET.SubElement(bndbox, "xmax").text = str(object["xmax"])
   ET.SubElement(bndbox, "ymax").text = str(object["ymax"])
  
   ET.SubElement(_object, "difficult").text  = object["difficult"]
   ET.SubElement(_object, "occuluded").text  = object["occluded"]
   ET.SubElement(_object, "pose").text       = object["pose"]
   ET.SubElement(_object, "truncated").text = object["truncated"]
   ET.SubElement(_object, "area").text = str(object["area"])

   size = ET.SubElement(annotation, "size")
   ET.SubElement(size, "depth").text  = str(object["depth"])
   ET.SubElement(size, "height").text = str(object["height"])
   ET.SubElement(size, "width").text  = str(object["width"])

   tree = ET.ElementTree(annotation)
   tree.write(os.path.join(NEW_XML_ROOT, filename))

def parse_single_object(object, filename):
   create_tree(object, filename)
   try:
      class_wise_files[object["name"]] += [filename.replace('.xml', '.jpg')]
   except KeyError:
      class_wise_files[object["name"]]  = [filename.replace('.xml', '.jpg')]


def parse(objects, filename):
   if len(objects) == 1 and objects[0]['name'] in YTO_CLASSES:
      parse_single_object(objects[0], filename)
   else:
      max_area = -1
      best_idx = -1
      for idx, object in enumerate(objects):
         if object["area"] > max_area:
            max_area = object["area"]
            best_idx = idx
      parse_single_object(objects[best_idx], filename)

with open(os.path.join(IMG_SET_DIR, 'train.txt')) as f:
   files = f.readlines()
files = [f.strip() for f in files]

for iii, f in enumerate(files):
   annotation = os.path.join(XML_ROOT, f+'.xml')
   
   tree = ET.parse(annotation)
   objects = []
   
   ss = tree.find('size')
   depth  = int(ss.find('depth').text)
   height = int(ss.find('height').text)
   width  = int(ss.find('width').text)
   fname  = tree.find('filename').text
   folder = tree.find('folder').text

   for obj in tree.findall('object'):
      obj_struct = {}
      obj_struct['name'] = obj.find('name').text
      bbox = obj.find('bndbox')

      xmax = int(bbox.find('xmax').text)-1
      xmin = int(bbox.find('xmin').text)-1
      ymax = int(bbox.find('ymax').text)-1
      ymin = int(bbox.find('ymin').text)-1

      obj_struct['xmin']  = xmin
      obj_struct['xmax']  = xmax
      obj_struct['ymin']  = ymin
      obj_struct['ymax']  = ymax
      
      obj_struct['difficult']  = obj.find('difficult').text
      obj_struct['occluded']  = obj.find('occluded').text
      obj_struct['pose']  = obj.find('pose').text
      obj_struct['truncated']  = obj.find('truncated').text
      
      obj_struct['bbox'] = [xmin, ymin, xmax, ymax]

      h = max(0, ymax - ymin)
      w = max(0, xmax - xmin)
      obj_struct['area'] = h * w

      obj_struct['depth']  = depth
      obj_struct['height'] = height
      obj_struct['width']  = width

      obj_struct['filename']  = fname
      obj_struct['folder']  = folder

      objects.append(obj_struct)
   parse(objects, '{}.xml'.format(f))

with open(os.path.join(NEW_XML_ROOT, 'train_files.txt'), 'wb') as f:
   pickle.dump(class_wise_files, f)
