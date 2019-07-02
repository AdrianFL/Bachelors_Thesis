# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 08:54:43 2018

@author: Pedro Pérez
"""
import keras
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve, auc,classification_report
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
import pandas as pd
from scipy import interp  
from itertools import cycle
import seaborn as sn 
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model

import ModelLoader as ml


def LSTMModel(n_cols):
    visible = Input(shape=(n_cols,), name="state")
    output   = LSTM(1, name='action')(visible)
    model   = Model(inputs=visible, outputs=output)
    
    return(model)


def BatchNormalizationModel(n_cols):
    visible = Input(shape=(n_cols,),  name="state")
    hidden1 = Dense(20, activation='relu')(visible)
    batch1  = BatchNormalization()(hidden1)
    hidden2 = Dense(20, activation='tanh')(batch1)
    batch2  = BatchNormalization()(hidden2)
    hidden3 = Dense(20, activation='relu')(batch2)
    batch3  = BatchNormalization()(hidden3)
    hidden4 = Dense(20, activation='tanh')(batch3)
    batch4  = BatchNormalization()(hidden4)
    hidden5 = Dense(15, activation='relu')(batch4)
    batch5  = BatchNormalization()(hidden5)
    hidden6 = Dense(10, activation='relu')(batch5)
    batch6  = BatchNormalization()(hidden6)
    hidden7 = Dense(5, activation='tanh')(batch6)
    batch7  = BatchNormalization()(hidden7)
    output  = Dense(1, activation='relu', name='action')(batch7)
    model   = Model(inputs=visible, outputs=output)
    
    return(model)
    
def NeuralNetworkModel(n_cols):
    visible = Input(shape=(n_cols,),  name="state")
    hidden1 = Dense(20, activation='relu')(visible)
    hidden2 = Dense(20, activation='tanh')(hidden1)
    hidden3 = Dense(20, activation='relu')(hidden2)
    hidden4 = Dense(20, activation='tanh')(hidden3)
    hidden5 = Dense(15, activation='relu')(hidden4)
    hidden6 = Dense(10, activation='relu')(hidden5)
    hidden7 = Dense(5, activation='tanh')(hidden6)
    output  = Dense(1, activation='relu', name='action')(hidden7)
    model   = Model(inputs=visible, outputs=output)
    
    return(model)
    
def DropoutModel(n_cols):
    dropoutvalue = 0.2
    
    visible = Input(shape=(n_cols,),  name="state")
    hidden1 = Dense(40, activation='relu')(visible)
    drop1   = Dropout(dropoutvalue)(hidden1)
    hidden2 = Dense(40, activation='tanh')(drop1)
    drop2   = Dropout(dropoutvalue)(hidden2)
    hidden3 = Dense(40, activation='relu')(drop2)
    drop3   = Dropout(dropoutvalue)(hidden3)
    hidden4 = Dense(40, activation='tanh')(drop3)
    drop4   = Dropout(dropoutvalue)(hidden4)
    hidden5 = Dense(30, activation='relu')(drop4)
    drop5   = Dropout(dropoutvalue)(hidden5)
    hidden6 = Dense(20, activation='relu')(drop5)
    drop6   = Dropout(dropoutvalue)(hidden6)
    hidden7 = Dense(10, activation='tanh')(drop6)
    drop7   = Dropout(dropoutvalue)(hidden7)
    output  = Dense(1, activation='relu', name='action')(drop7)
    model   = Model(inputs=visible, outputs=output)
    
    return(model)
    
def NeuralNetworkAvanceDataframe(x_train, x_test, y_train, y_test):
    
    batch_size = 8192
    epochs = 100
    
    model_name = 'Neural_Network_Grades'
    
    n_cols = x_train.shape[1]
    
    #x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    #x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    #print('X_train.shape: ', np.array(x_train).shape)
    #print('X_test.shape: ', np.array(x_test).shape)
    
    model = LSTMModel(n_cols)
    
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss=keras.losses.mean_squared_error, optimizer=sgd, metrics=['mse'])
    
    print("model.summary(): ",model.summary())
    
    nn = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    
    model.save('NewSave.h5')
    
    new_model = keras.models.load_model('NewSave.h5')
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model_json = model.to_json()
    
    with PdfPages(model_name+'_HistoryNN.pdf') as HistoryNNPdf:
    
        plt.plot(nn.history['mean_squared_error'],'r')  
        plt.plot(nn.history['val_mean_squared_error'],'g')  
        plt.xticks(np.arange(0, 5000, 500.0))  
        plt.rcParams['figure.figsize'] = (8, 6)  
        plt.xlabel("Num of Epochs")  
        plt.ylabel("Mse")  
        plt.title("Training Mse vs Validation Mse")  
        plt.legend(['train','validation'])
        plt.savefig(HistoryNNPdf, format='pdf')
        plt.clf()
        
        plt.plot(nn.history['loss'],'r')  
        plt.plot(nn.history['val_loss'],'g')  
        plt.xticks(np.arange(0, 5000, 500.0))  
        plt.rcParams['figure.figsize'] = (8, 6)  
        plt.xlabel("Num of Epochs")  
        plt.ylabel("Loss")  
        plt.title("Training Loss vs Validation Loss")  
        plt.legend(['train','validation'])
        plt.savefig(HistoryNNPdf, format='pdf')
        plt.clf()
    
    y_pred = nn.model.predict(x_test)
    
    score2 = new_model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss 2:', score2[0])
    print('Test accuracy 2:', score2[1])
    
    y_pred6 = nn.model.predict(x_test)
        
    print('Output model --------------')
    with open(model_name+"_json", "w") as json_file:  
      json_file.write(model_json)
    model.save_weights(model_name+".h5")

    with K.get_session() as sess:
        
        K.tf.keras.models.load_model('NewSave.h5')
            
        y_pred6 = nn.model.predict(x_test)
    
        graph = sess.graph
        
        tf.train.Saver().save(sess, './'+model_name+'_simple2.ckpt')
        tf.train.write_graph(graph.as_graph_def(), logdir='./', name= model_name+'_simple_as_binary2.pb', as_text=False)
        tf.train.write_graph(graph.as_graph_def(), logdir='./', name= model_name+'_simple2.pbtxt', as_text=True)
        
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open(model_name+'_simple_as_binary2.pb', "rb" ) as f:
            input_graph_def.ParseFromString(f.read())
                
        tf.import_graph_def(input_graph_def)
    
        y_pred2 = nn.model.predict(x_test)
            
        ml.freeze_graph_optimized()
      
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open('optimised_model.bytes', "rb" ) as f:
            input_graph_def.ParseFromString(f.read())
            
        tf.import_graph_def(input_graph_def)
            
        y_pred3 = nn.model.predict(x_test)

        

    return(y_pred, y_pred2, y_pred3, y_pred6)
    



def NeuralNetworkAvance(img_cols,img_rows, x_train, x_test, y_train, y_test,targetCoordinates):
    
    batch_size = 64
    epochs = 10
    
    if targetCoordinates:
        final_layers = 2  #variables to predict
        model_name = 'Model_Avance'
    else:
        final_layers = 1  #variables to predict
        model_name = 'Model_grades'

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
    
    model = Sequential()    
    model.add(Conv2D(32, kernel_size=(2, 3),
                     activation='relu',
                     input_shape=input_shape,
                     name="state"))
    """
    model.add(Conv2D(64, (2, 3), activation='tanh'))
    model.add(BatchNormalization())
    """
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    """
    model.add(Dense(128, activation='tanh'))
    model.add(BatchNormalization())
    """
    model.add(Dense(final_layers, activation='relu', name="action"))
    
    sgd = keras.optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=sgd,
                  metrics=['mse'])
    
       
    print("model.layers: ",model.layers)
    print("model.inputs: ",model.inputs)
    print("model.outputs: ",model.outputs)
    print("model.summary(): ",model.summary())
    print("model.get_config(): ",model.get_config())
    #print("model.get_weights(): ",model.get_weights())
    
    
    nn = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    
    #print("model.get_weights(): ",model.get_weights())
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model_json = model.model.to_json()
    
    with PdfPages(model_name+'_HistoryNN.pdf') as HistoryNNPdf:
    
        plt.plot(nn.history['mean_squared_error'],'r')  
        plt.plot(nn.history['val_mean_squared_error'],'g')  
        plt.xticks(np.arange(0, 11, 2.0))  
        plt.rcParams['figure.figsize'] = (8, 6)  
        plt.xlabel("Num of Epochs")  
        plt.ylabel("Mse")  
        plt.title("Training Mse vs Validation Mse")  
        plt.legend(['train','validation'])
        plt.savefig(HistoryNNPdf, format='pdf')
        plt.clf()
        
        plt.plot(nn.history['loss'],'r')  
        plt.plot(nn.history['val_loss'],'g')  
        plt.xticks(np.arange(0, 11, 2.0))  
        plt.rcParams['figure.figsize'] = (8, 6)  
        plt.xlabel("Num of Epochs")  
        plt.ylabel("Loss")  
        plt.title("Training Loss vs Validation Loss")  
        plt.legend(['train','validation'])
        plt.savefig(HistoryNNPdf, format='pdf')
        plt.clf()
        
        
    y_pred = nn.model.predict(x_test)

   
    print('Output model --------------')
    with open(model_name+"_json", "w") as json_file:  
      json_file.write(model_json)
    model.model.save_weights(model_name+".h5")
    """
    print('Output weights --------------')
    weightsModel = model.model.get_weights()
    np.set_printoptions(threshold=np.nan)
    with open(model_name+"_weightsModel.txt", "w") as weights_file:
        for modelWeights in weightsModel:
            weights_file.write("shape: %s\n\n%s\n" % (np.shape(modelWeights), modelWeights)) 
    """        
    print('Output graph --------------')            
    with tf.Session() as sess:
        h5 =  model_name+".h5"
        json = h5.replace(".h5", "_json")
        
        #sgd = SGD(lr=0.1, decay=.1, momentum=.9, nesterov=False)
        model = model_from_json(open(json).read())
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=sgd,
                      metrics=['mse'])
        model.load_weights(h5)
        
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        graph = sess.graph
    
        tf.train.Saver().save(sess, './'+model_name+'_simple.ckpt')
        tf.train.write_graph(graph.as_graph_def(), logdir='./', name= model_name+'_simple_as_binary.pb', as_text=False)

    return(y_pred)
  
            
def MatrizConfusion (y, y_predicted,n_classes):

    #Creamos la matriz de confusión
    cm = confusion_matrix(y, y_predicted)

    # Visualizamos la matriz de confusión
    df_cm = pd.DataFrame(cm, range(n_classes), range(n_classes))  
    plt.figure(figsize = (20,14))  
    sn.set(font_scale=1.4) #for label size  
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size  
    plt.show()
    
    snn_report = classification_report(y, y_predicted)  
    print(snn_report)
 
 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          Method='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0

        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:.1f}%".format(100*cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:.0f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
            

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    
    with PdfPages('ConfusionMatrix'+Method+'.pdf') as ConfusionMatrixPdf:
        plt.savefig(ConfusionMatrixPdf, format='pdf')
        
    plt.clf()
    

def ConfusionMatrixPlot(NameModel, y,y_pred,predict_y_classes):
    
    cm = confusion_matrix(y_pred, y)
    np.set_printoptions(precision=3)

    plt.figure()
    
    plot_confusion_matrix(cm, classes=predict_y_classes,
                          normalize=False,
                          title='Confusion Normalized matrix',
                          Method = NameModel+'_' +'NoNorm')
                          
    plot_confusion_matrix(cm, classes=predict_y_classes,
                          normalize=True,
                          title='Confusion Normalized matrix',
                          Method = NameModel+'_' +'Norm')
                          
    #precision    
    PrecissionElements = 0
    
    for element in range(0,len(cm)):
        PrecissionElements += cm[element,element]
    
    PrecissionElementsPor = PrecissionElements/np.sum(cm)
    print('Precission Confussion Matrix: ',PrecissionElementsPor)
    
    

def ROCCurvePlot (predict_y,num_bins, y_prob,Method,Model):
    
    # Binarize the output
    ClassesList = list(range(0,num_bins))
    
    y = label_binarize(predict_y, classes=ClassesList)
    n_classes = y.shape[1]
        
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

   
    for i in range(n_classes):
      
        fpr[i], tpr[i], _ = roc_curve(y[:,i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    ##############################################################################
    # Plot ROC curves for the multiclass problem
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    
    with PdfPages('ROCCurve'+Method+'_'+Model+'.pdf') as ROCCurvePdf:
        plt.savefig(ROCCurvePdf, format='pdf')
        
    plt.clf()

def NeuralNetworkDispara(img_cols,img_rows, x_train, x_test, y_train_original, y_test_original):
    
    num_classes = 2
    batch_size = 128
    epochs = 50
    
    final_layers = 2  #variables to predict
    model_name = 'Model_dispara'
    

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
              
    y_train = keras.utils.to_categorical(y_train_original, num_classes)
    y_test = keras.utils.to_categorical(y_test_original, num_classes)    
         
    
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 3),
                     activation='relu',
                     input_shape=input_shape))
       
    model.add(Conv2D(64, (2, 3), activation='relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 3)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(final_layers, activation='softmax'))
    
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    """
    
    class_weights = class_weight.compute_class_weight(
                   'balanced',
                    np.unique(y_train_original), 
                    y_train_original)
      
    class_weights = {0:class_weights[0],1:class_weights[1]}
    
    print('class_weights: ', class_weights)
     
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(final_layers, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    

    #print("model.layers: ",model.layers)
    print("model.inputs: ",model.inputs)
    print("model.outputs: ",model.outputs)
    print("model.summary(): ",model.summary())
    #print("model.get_config(): ",model.get_config())
    #print("model.get_weights(): ",model.get_weights())
    
    
    nn = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              class_weight = class_weights)
    
    #print("model.get_weights(): ",model.get_weights())
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model_json = model.model.to_json()
    

    with PdfPages(model_name+'_HistoryNN.pdf') as HistoryNNPdf:
    
    
        plt.plot(nn.history['acc'],'r')  
        plt.plot(nn.history['val_acc'],'g')  
        plt.xticks(np.arange(0, 11, 2.0))  
        plt.rcParams['figure.figsize'] = (8, 6)  
        plt.xlabel("Num of Epochs")  
        plt.ylabel("Accuracy")  
        plt.title("Training Accuracy vs Validation Accuracy")  
        plt.legend(['train','validation'])
        plt.savefig(HistoryNNPdf, format='pdf')
        plt.clf()

        plt.plot(nn.history['loss'],'r')  
        plt.plot(nn.history['val_loss'],'g')  
        plt.xticks(np.arange(0, 11, 2.0))  
        plt.rcParams['figure.figsize'] = (8, 6)  
        plt.xlabel("Num of Epochs")  
        plt.ylabel("Loss")  
        plt.title("Training Loss vs Validation Loss")  
        plt.legend(['train','validation'])
        plt.savefig(HistoryNNPdf, format='pdf')
        plt.clf()  

    
    print('Output model --------------')   
    with open(model_name+"_json", "w") as json_file:  
      json_file.write(model_json)
    model.model.save_weights(model_name+".h5")


    print('---------------------------- Confusion Matrix Test')         
    y_pred = model.predict(x_test, batch_size=32, verbose=1)
    y_predicted = np.argmax(y_pred, axis=1)  
      
    predict_y_classes = ['no dispara', 'dispara']
    ConfusionMatrixPlot(model_name, y_test_original,y_predicted,predict_y_classes)
    
    
    print('Output weights --------------')
    weightsModel = model.model.get_weights()
    np.set_printoptions(threshold=np.nan)
    with open(model_name+"_weightsModel.txt", "w") as weights_file:
        for modelWeights in weightsModel:
            weights_file.write("shape: %s\n\n%s\n" % (np.shape(modelWeights), modelWeights)) 
            
    print('Output graph --------------')            
    with tf.Session() as sess:
        h5 =  model_name+".h5"
        json = h5.replace(".h5", "_json")
        
        #sgd = SGD(lr=0.1, decay=.1, momentum=.9, nesterov=False)
        model = model_from_json(open(json).read())
        model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
        model.load_weights(h5)
        
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        graph = sess.graph
    
        tf.train.Saver().save(sess, './'+model_name+'_simple.ckpt')
        tf.train.write_graph(graph.as_graph_def(), logdir='./', name= model_name+'_simple_as_binary.pb', as_text=False)
        
        
    #print('HiddenWeights', str(nn_weights[0]))
    #print('OutputWeights', str(nn_weights[1]))
    
    
    

    #CurvaRoc(final_layers+1,y_test, y_pred)
    
    