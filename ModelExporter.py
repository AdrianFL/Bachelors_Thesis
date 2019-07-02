# IMPORTS
from __future__ import print_function
# Keras' "get_session" function gives us easy access to the session where we train the graph
#from keras import backend as K

import tensorflow as tf
# freeze_graph "screenshots" the graph
from tensorflow.python.tools import freeze_graph
# optimize_for_inference lib optimizes this frozen graph
from tensorflow.python.tools import optimize_for_inference_lib
#For converting .pb files into pbtxt
#from tensorflow.python.platform import gfile

# os and os.path are used to create the output file where we save our frozen graphs
#import os
#import os.path as path

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and return it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def TransformToBytes():
    #We are hard-coding the files now to check if this works
    output_node_name = ["action/Relu",]
    
    #input_node_name = ["state"]
    
    GRAPH_NAME = "Neural_Network_Grades_simple2"
    
    freeze_graph.freeze_graph( 'Neural_Network_Grades_simple2.pbtxt', None, False,
                               GRAPH_NAME + '.ckpt', output_node_name,
                              "save/restore_all", "save/Const:0",
                               GRAPH_NAME + '.bytes', True, "")
    
    freeze_graph.freeze_graph()
    '''
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(GRAPH_NAME + '.bytes', 'rb') as f:
        input_graph_def.ParseFromString(f.read())
        
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
                        input_graph_def, input_node_name, [output_node_name], tf.float32.as_datatype_enum)
    
    with tf.gfile.FastGFile(GRAPH_NAME + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    '''
    
    print('Frozen graph saved succesfully!')
    
    










