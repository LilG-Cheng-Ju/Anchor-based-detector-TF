import os
import tensorflow as tf
import numpy as np 
import xml.etree.ElementTree as ET
from skimage import io

class Data_parser:
    def __init__(self,
                 data_folder,
                 Classes = ['Cyan-Blue','Blue','Green','Red'],
                ):
        self.data_folder = data_folder
        self.Classes = Classes
        
    def xml_parser(self):
        path = os.listdir(self.data_folder)
        Label_map = self.cls_to_dir(self.Classes)

        output = []
        for file in path:
            if file[-4:] == ".xml":
                Tree = ET.parse(self.data_folder +'/'+ file)
                box_get = Tree.findall('object')
                filename = Tree.find('filename').text
                width = int(Tree.find('size').find('width').text)
                height = int(Tree.find('size').find('height').text)
                box_num = len(box_get)
                Bbox = np.zeros(shape = (box_num,4))
                Class = np.zeros(shape = (box_num,1))
                for i,box in enumerate(box_get):
                    bbox = box.find('bndbox')
                    Bbox[i][0] = float(bbox.find('xmin').text) - 1
                    Bbox[i][1] = float(bbox.find('ymin').text) - 1
                    Bbox[i][2] = float(bbox.find('xmax').text) - 1
                    Bbox[i][3] = float(bbox.find('ymax').text) - 1
                    cls_name = box.find('name').text
                    if cls_name in Label_map:
                        Class[i][0] =  Label_map[cls_name]
                    else:
                        raise ValueError(f"{cls_name} not define.")
                decoded_data = [filename,width,height,Class,Bbox]
                output.append(decoded_data)
        return output
    
    def generate_example(self):
        encoded_annotation = self.xml_parser()
        example = []
        for data in encoded_annotation:
            image = io.imread(self.data_folder+'/'+data[0])[:,:,0:3]
            image_bytes = image.tobytes()
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': self.int64_feature(int(data[1])),
                'image/width': self.int64_feature(int(data[2])),
                'image/image_bytes': self.bytes_feature(image_bytes),
                'image/bbox': self.bytes_feature(data[4].tobytes()),
                'image/label': self.bytes_feature(data[3].tobytes())
            }))
            example.append(tf_example)
        
        return example
    
    def generate_tfrecord(self,output_path):
        writer = tf.compat.v1.python_io.TFRecordWriter(output_path)
        encoded_annotation = self.xml_parser()
        for i,data in enumerate(encoded_annotation):
            image = io.imread(self.data_folder+'/'+data[0])[:,:,0:3]
            image_bytes = image.tobytes()
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': self.int64_feature(int(data[1])),
                'image/width': self.int64_feature(int(data[2])),
                'image/image_bytes': self.bytes_feature(image_bytes),
                'image/bbox': self.bytes_feature(data[4].tobytes()),
                'image/label': self.bytes_feature(data[3].tobytes())
            }))
            writer.write(tf_example.SerializeToString())
        writer.close()
        print(f'TFrecord generated successfully, including {i+1} pieces of data')
    
    def cls_to_dir(self,classes):
        output = {}
        for i,cls in enumerate(classes):
            output[cls] = i+1
        return output
    
    def int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def int64_list_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


    def bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def bytes_list_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


    def float_feature(self,value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def float_list_feature(self,value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))