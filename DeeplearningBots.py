# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:50:17 2018

@author: Pedro Pérez

"""

import numpy as np
import pandas as pd
#import pandas_profiling
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mysql.connector as my
import sys
import ImagesTransform as it
import time
from sklearn.model_selection import train_test_split
import PredictionsModels as pm
from imblearn.over_sampling import SMOTE



try:

    userDb = sys.argv [1]
    passwordDb = sys.argv [2]
    hostDb = sys.argv [3]
    databaseDB = sys.argv [4]
           
except:
    
    # The following part has been redacted for privacy reasons
    
    userDb = 
    passwordDb = 
    hostDb = 
    databaseDB = 
    
    


#Initialization variables
matches = 0
dataRead = []
AllDataRead = []
dataRows = []
InitialData = [] #Division/modo de juego
NumMatchesMax = 200
printMatches = False
ProcessModoJuego2vs2 = True
axisFieldSize = [-25, 31.54, -16.48, 13.352] 
axisFieldSizeX = [axisFieldSize[0],axisFieldSize[1]]
axisFieldSizeY = [axisFieldSize[2],axisFieldSize[3]]
img_cols = 112
PrintHistograms = False
ColumnsMatchesData = ['id',
                      'name',
                      'NumFrames',
                      'NoMovesFrames',
                      'PorcNoMovesFrames']

#Target variables
targetCoordinates = False
IsPredictionAvance = True
IsPredictionDispara = False

     
#Other variables                           
if ProcessModoJuego2vs2:
    ColumnsName = ['jugador_avance_x',
                    'jugador_avance_y',
                    'disparo',
                    'jugador_posicion_x',
                    'jugador_posicion_y',
                    'pelota_posicion_x',
                    'pelota_posicion_y',
                    'compañero_posicion_x',
                    'compañero_posicion_y',
                    'contrincante1_posicion_x',
                    'contrincante1_posicion_y',
                    'contrincante2_posicion_x',
                    'contrincante2_posicion_y',
                    'gol']

    NameFile = 'matches2vs2_santiacho.pdf'
    colors = ['green','black', 'blue','brown', 'red']

    labelsPlot = ['jugador','pelota','compañero','contrincante1','contrincante2'] 
    #colorsGray = [0.587,0,0.114,0.0598,0.299]
    colorsGray = [0.5,0,0.75,0.25,0.25]
    positionsReferential = [[0,[3,4]],[1,[5,6]],[2,[7,8]],[3,[9,10]],[4,[11,12]]]
    
else:
    ColumnsName = ['jugador_avance_x',
                    'jugador_avance_y',
                    'disparo',
                    'jugador_posicion_x',
                    'jugador_posicion_y',
                    'pelota_posicion_x',
                    'pelota_posicion_y',
                    'contrincante_posicion_x',
                    'contrincante_posicion_y',
                    'gol']
    

    NameFile = 'matches1vs1.pdf'
    colors = ['green','black', 'brown']
    labelsPlot = ['jugador','pelota','contrincante']
    colorsGray = [0.25,0,0.75]
    positionsReferential = [[0,[3,4]],[1,[5,6]],[2,[7,8]]]

    
        
def scatterPlotMatch (plotSome, size, data):
                      
    NumFeature = 0
    plt.axis(axisFieldSize)
    for i in plotSome:
        if i[0] == 1:
            if size[NumFeature] == 0:
                plt.scatter(data[ColumnsName[i[1][0]]], 
                            data[ColumnsName[i[1][1]]],
                            c=colors[NumFeature],
                            label=labelsPlot[NumFeature])
          
            else:
                plt.scatter(data[ColumnsName[i[1][0]]], 
                            data[ColumnsName[i[1][1]]],
                            c=colors[NumFeature],
                            label=labelsPlot[NumFeature],
                            s=size[NumFeature])      

        NumFeature +=1
    
    plt.legend()
    
    return(plt)


def obtainDataDataBase(userDb,passwordDb, hostDb, databaseDB):
        
    cnx = my.connect(user=userDb, password=passwordDb,
                                  host=hostDb,
                                  database=databaseDB)
    
    # id >= 87
    SelectMatches = ("SELECT id, date, if(data is null,'#',data) as data "+ \
                          "FROM "+databaseDB+".matches_log a where id >= 127  "+ \
                            " LIMIT "+str(NumMatchesMax))
    
    
    try:
         curA = cnx.cursor()
         query = SelectMatches
         curA.execute(query)
         
    except:
         print('Something went Wrong Selecting information')
                    
         print(query, sys.exc_info()[0])
         
         curA.close()   
    
    
    matchesData = []
    AllMatchDataRead = []
    matches = 0
    
    for (id, date, data) in curA:
          
            BeginMatch, PorcNoMovesFrames, matches, \
                NoMovesFrames, NumFrames,AllMatchDataRead = ArrageData(id, date, data, matches)
            
            if BeginMatch:              
                
                PorcNoMovesFrames = float(np.round(100*NoMovesFrames/NumFrames,2))
                 
                try:
                    name = InitialData[3]
                except:
                    name = ''
                          
                print('id: %s , name: %s, Frames: %s , NoMovesFrames: %s , Porc: %s ' %(id,name, NumFrames, NoMovesFrames, \
                                                                           np.round(100*NoMovesFrames/NumFrames,2)))  
            
                if PorcNoMovesFrames <= 20: #if Porc No Moves Frames under 20 (percentil 75), is not outlier match
                      for line in AllMatchDataRead:
                          AllDataRead.append(line)
                          
                      matchesData.append([id,name, int(NumFrames), int(NoMovesFrames),\
                                          PorcNoMovesFrames]) 
                                
            AllMatchDataRead = []
            
    curA.close()    

    return(AllDataRead,matchesData)

"""
def obtainDataFile():
    
    import csv

    with open('matches_log') as csvfile:
        lines = csv.reader(csvfile, delimiter='&')
            
        matchesData = []
        AllMatchDataRead = []
        matches = 0
        
        for row in lines:
            print(row)
            print(type(row))
            id = int(row['id'])
            date = row['date']
            data = row['data']
            
            
            if id >= 87:
              
                BeginMatch, PorcNoMovesFrames, matches, NoMovesFrames, NumFrames = ArrageData(id, date, data,matches)
                
                if BeginMatch:              
                    
                    PorcNoMovesFrames = float(np.round(100*NoMovesFrames/NumFrames,2))
                     
                    try:
                        name = InitialData[3]
                    except:
                        name = ''
                              
                    print('id: %s , name: %s, Frames: %s , NoMovesFrames: %s , Porc: %s ' %(id,name, NumFrames, NoMovesFrames, \
                                                                               np.round(100*NoMovesFrames/NumFrames,2)))  
                
                    if PorcNoMovesFrames <= 20: #if Porc No Moves Frames under 20 (percentil 75), is not outlier match
                          for line in AllMatchDataRead:
                              AllDataRead.append(line)
                              
                          matchesData.append([id,name, int(NumFrames), int(NoMovesFrames),\
                                              PorcNoMovesFrames]) 
                                    
                AllMatchDataRead = []
            
    return(AllDataRead,matchesData)

"""
def ArrageData(id, date, data,matches):

        AllMatchDataRead = []
        BeginMatch = False
        PorcNoMovesFrames = 0
        NoMovesFrames = 0
        NumFrames = 0

        
        if data[-1].replace('\n','') ==  '$':
            ProcessMatch = True
        else:
            ProcessMatch = False

        if ProcessMatch:
            BeginMatch = False
            dataRead = []
            InitialData = []
            
            dataRows = data[:-1].split('\n')
            NumFrames = 0
            NoMovesFrames = 0
            PorcNoMovesFrames = 0
            
            for line in dataRows:
                
                #División y Modo de juego
                if not(BeginMatch):
                    InitialData.append(line) 
                    
                if line == '$':
                    #Modo de juego?
                    if (InitialData[1] == '2' and ProcessModoJuego2vs2) \
                        or (InitialData[1] == '1' and not(ProcessModoJuego2vs2)):

                        BeginMatch = True
                        matches += 1
                    else:
                        break
                    
                elif BeginMatch:
                    
                    if not(line == '#'):
                        dataRead.append(line)
                                                    
                    elif len(dataRead)>0:
                        AllMatchDataRead.append(dataRead)
                        
                        NumFrames += 1
                        #print(dataRead)
                        if float(dataRead[0]) == 0.0 and float(dataRead[1]) == 0.0:
                             NoMovesFrames += 1 
                               
                        dataRead = []

        return (BeginMatch, PorcNoMovesFrames, matches, NoMovesFrames, NumFrames,AllMatchDataRead)    
        
def PrintHistogramsData(DataFrame, NameFile):
    
        print('#Histograms')
    
        ColumnsPrintHistogram = []
        num_bins_histogram = 100
        with PdfPages('Data_Histograms_'+NameFile+'.pdf') as HistogramsPdf:

            for Column in DataFrame.columns:
                if np.issubdtype(DataFrame[Column].dtype, np.number):
                    if (len(ColumnsPrintHistogram)>0) and (Column in ColumnsPrintHistogram):
                        # histogram of the data
                        plt.hist(DataFrame[Column], num_bins_histogram, facecolor='blue')
                        plt.xlabel(Column)
                        plt.title('Histogram of '+Column)
                        plt.savefig(HistogramsPdf, format='pdf')
                        plt.clf()
                    elif len(ColumnsPrintHistogram) == 0:
                        # histogram of the data
                        plt.hist(DataFrame[Column], num_bins_histogram, facecolor='blue')
                        plt.xlabel(Column)
                        plt.title('Histogram of '+Column)
                        plt.savefig(HistogramsPdf, format='pdf')
                        plt.clf()

            if 'jugador_avance_x' in DataFrame.columns and 'jugador_avance_y' in DataFrame.columns:
                    plt.hist(DataFrame.query('jugador_avance_x!= 0')['jugador_avance_y'], num_bins_histogram, facecolor='blue')
                    plt.xlabel(Column)
                    plt.title('Histogram of jugador_avance_x!= 0')
                    plt.savefig(HistogramsPdf, format='pdf')
                    plt.clf()
        
                    plt.hist(DataFrame.query('jugador_avance_y!= 0')['jugador_avance_y'], num_bins_histogram, facecolor='blue')
                    plt.xlabel(Column)
                    plt.title('Histogram of jugador_avance_y!= 0')
                    plt.savefig(HistogramsPdf, format='pdf')
                    plt.clf()                    


    
def PredictionAvance (AllDataReadFrame,targetCoordinates):
    if targetCoordinates:
       
        y_train_origin = AllDataReadFrame[['jugador_avance_x', 'jugador_avance_y']].copy(deep=True)
        
        print("Not using angles!!!")
        
    else:
        
        AllDataReadFrame = it.TranformtoGrades (AllDataReadFrame)
        #images_train_origin, img_rows = it.TranformToImage (AllDataReadFrame,axisFieldSizeX,axisFieldSizeY,img_cols,\
        #                               colorsGray,positionsReferential,ColumnsName)   
        
        PrintHistogramsData(AllDataReadFrame,'GradesData')
            
        y_train_origin = AllDataReadFrame['angle'].copy(deep=True)
        AllDataReadFrame = AllDataReadFrame.drop('angle', axis=1)
        AllDataReadFrame = AllDataReadFrame.drop('jugador_avance_x', axis=1)
        AllDataReadFrame = AllDataReadFrame.drop('jugador_avance_y', axis=1)
        AllDataReadFrame = AllDataReadFrame.drop('disparo', axis=1)
        AllDataReadFrame = AllDataReadFrame.drop('gol', axis=1)
        #AllDataReadFrame, y_train_origin = it.PrepareToLSTM(AllDataReadFrame, 3)

        #AllDataReadFrame = AllDataReadFrame.drop('angle', axis=0)
        PrintHistogramsData(AllDataReadFrame, 'GradesDataFinal')
    
    #y_train_origin =  np.array(y_train_origin, dtype='float64')
    #images_train_origin = np.array(images_train_origin, dtype='float64')
    #AllDataReadArray = np.array(AllDataReadFrame)
    
    AllDataReadArray = AllDataReadFrame.values
    y_train_origin   = y_train_origin.values
    
    #AllDataReadArray = AllDataReadArray.reshape(AllDataReadArray.shape[0], 1, AllDataReadArray[1])
    
    x_train, x_test, y_train, y_test = \
                train_test_split(AllDataReadArray, y_train_origin, test_size=0.25, random_state=42)
                
    #x_train = x_train.reshape(x_train.shape[0], 3, x_train.shape[1])
                
    #sm = SMOTE(random_state=42)
    
    #print('X_train.shape: ', np.array(x_train).shape)
    #print('y_train.shape: ', np.array(y_train).shape)
    
    #images, rows, cols = np.array(x_train).shape
    
    #x_train = np.reshape(x_train, (images, rows*cols))
    
    #x_train, y_train = sm.fit_sample(x_train, y_train)
        
    #x_train = np.reshape(x_train, (images, rows, cols))
    
    #OversampledDataFrame = pd.DataFrame(y_train, columns = ['y_train',], dtype='float64')
    
    #PrintHistogramsData(OversampledDataFrame, 'OversampledData')
                    
    print('X_train.shape: ', np.array(x_train).shape)
    print('X_test.shape: ', np.array(x_test).shape)
    print('y_train.shape: ', np.array(y_train).shape)
    print('y_test.shape: ', np.array(y_test).shape)
    
    
    #print(AllDataReadFrame.describe())
    
    y_pred, y_pred2, y_pred3, y_pred6 = pm.NeuralNetworkAvanceDataframe(x_train, x_test, y_train, y_test)
    #print(y_pred)
    PredictionDataFrame = pd.DataFrame(y_pred, columns = ['y_pred',], dtype='float64')
    PredictionDataFrame2 = pd.DataFrame(y_pred2, columns = ['y_pred2',], dtype='float64' )
    PredictionDataFrame3 = pd.DataFrame(y_pred3, columns = ['y_pred3',], dtype='float64' )
    PredictionDataFrame6 = pd.DataFrame(y_pred6, columns = ['y_pred6',], dtype='float64' )
    #print(PredictionDataFrame.describe())
    PrintHistogramsData(PredictionDataFrame,'PredictionData')
    PrintHistogramsData(PredictionDataFrame2, 'PredictionData2')
    PrintHistogramsData(PredictionDataFrame3, 'PredictionData3')
    PrintHistogramsData(PredictionDataFrame6, 'PredictionData6')


    


    
def PredictionDispara (AllDataReadFrame,targetCoordinates):
       
    y_train_origin = AllDataReadFrame['disparo'].copy(deep=True)
          
    y_train_origin =  np.array(y_train_origin, dtype='int')
    #images_train_origin = np.array(images_train_origin, dtype='float64')
       
    '''
   
    x_train, x_test, y_train, y_test = \
                train_test_split(images_train_origin, y_train_origin, test_size=0.25, random_state=42)

    print('y_test sample',y_test[:25])              
        
    print('X_train.shape: ', np.array(x_train).shape)
    print('X_test.shape: ', np.array(x_test).shape)
    print('y_train.shape: ', np.array(y_train).shape)
    print('y_test.shape: ', np.array(y_test).shape)
    
    pm.NeuralNetworkDispara(img_cols,img_rows, x_train, x_test, y_train, y_test)
    '''
    
    

    
if __name__ == '__main__':

    print('Begin process: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
   
    AllDataRead, matchesData = obtainDataDataBase(userDb,passwordDb, hostDb, databaseDB)
    
    #AllDataRead, matchesData = obtainDataFile()
                    
    AllDataReadFrame = pd.DataFrame(AllDataRead, columns = ColumnsName, dtype='float64')
    matchesDataFrame = pd.DataFrame(matchesData, columns = ColumnsMatchesData, dtype='float64')
    
        
    print(AllDataReadFrame.describe())
    print("--------------------------")
    print("Num matches : ", matches)
    
    print(matchesDataFrame.describe())

       
    if printMatches:
        with PdfPages(NameFile) as pdf:
            
            #plot jugador, pelota y compñero
            if ProcessModoJuego2vs2:
                plotSome = [[1,[3,4]],[1,[5,6]],[1,[7,8]],[0,[9,10]],[0,[11,12]]] #Jugador,[Numero columnas],Pelota,Compañero, Contrincante1, Contrincante2
                sizePlot = [0,0,0,0,0]
            else:
                plotSome = [[1,[3,4]],[1,[5,6]],[0,[7,8]]] #Jugador,[Numero columnas],Pelota,Contrincante
                sizePlot = [0,0,0]         
            
            plotMatch = scatterPlotMatch (plotSome, sizePlot, AllDataReadFrame)       
            pdf.savefig()
            plotMatch.clf()
    
            #plot contrincante/s
            if ProcessModoJuego2vs2:
                plotSome = [[0,[3,4]],[0,[5,6]],[0,[7,8]],[1,[9,10]],[1,[11,12]]] #Jugador,[Numero columnas],Pelota,Compañero, Contrincante1, Contrincante2
                sizePlot = [0,0,0,0,0]
            else:
                plotSome = [[0,[3,4]],[0,[5,6]],[1,[7,8]]] #Jugador,[Numero columnas],Pelota,Contrincante
                sizePlot = [0,0,0]         
            
            plotMatch = scatterPlotMatch (plotSome, sizePlot, AllDataReadFrame)       
            pdf.savefig()
            plotMatch.clf()
        
            #plot all frames
            if ProcessModoJuego2vs2:
                plotSome = [[1,[3,4]],[1,[5,6]],[1,[7,8]],[1,[9,10]],[1,[11,12]]] #Jugador,[Numero columnas],Pelota,Compañero, Contrincante1, Contrincante2
                sizePlot = [170,100,170,170,170]
            else:
                plotSome = [[1,[3,4]],[1,[5,6]],[1,[7,8]]] #Jugador,[Numero columnas],Pelota,Contrincante
                sizePlot = [170,100,170] 
            
            for index, row in AllDataReadFrame.iterrows():
                plotMatch = scatterPlotMatch (plotSome, sizePlot, row)       
                pdf.savefig()
                plotMatch.clf()
    
    
    # Print histograms
    if PrintHistograms:  
        PrintHistogramsData(AllDataReadFrame,'AllDataRead')
        PrintHistogramsData(matchesDataFrame,'MatchesData')
                
        
    #print('Begin TranformToImage: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    #images_train_origin, img_rows = it.TranformToImage (AllDataReadFrame,axisFieldSizeX,axisFieldSizeY,img_cols,\
    #                                   colorsGray,positionsReferential,ColumnsName)    

    #print('End Tranform: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    
    #import random
    '''
    #images_train_origin = np.array(images_train_origin)
    first_image = images_train_origin[0]
    #first_image = np.array(first_image, dtype='float')
    #print('images_train.shape: ',images_train.shape)
    plt.imshow(first_image, cmap='gray')
    plt.show()

    r = random.randrange(len(images_train_origin))
    #print(y_train[r])
    random_image = images_train_origin[r]
    random_image = np.array(random_image, dtype='float')
    plt.imshow(random_image, cmap='gray')
    plt.show()
    '''
    
    if IsPredictionAvance:
        PredictionAvance(AllDataReadFrame,targetCoordinates)
        
    
    elif IsPredictionDispara:
        PredictionDispara(AllDataReadFrame,targetCoordinates)
    
        
    print('End process: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    