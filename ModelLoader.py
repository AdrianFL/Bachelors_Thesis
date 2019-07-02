# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:03:24 2018

@author: Adrián Francés
"""
import tensorflow as tf


def freeze_graph_optimized():   
    input_checkpoint = "Neural_Network_Grades_simple2.ckpt"
    
    output_node_names = ["action/Relu",]
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    
        # We restore the weights
        saver.restore(sess, input_checkpoint)
    
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ['add_default_attributes',
                      'remove_nodes(op=Identity, op=CheckNumerics)',
                      'fold_batch_norms', 'fold_old_batch_norms',
                      'strip_unused_nodes', 'sort_by_execution_order']
        transformed_graph_def = TransformGraph(tf.get_default_graph().as_graph_def(),'state', output_node_names, transforms)
    
    # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, transformed_graph_def,  output_node_names)
        with tf.gfile.GFile("optimised_model.bytes", "wb") as f:
            f.write(output_graph_def.SerializeToString())
            
