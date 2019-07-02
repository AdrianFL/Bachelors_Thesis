# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:07:25 2018

@author: Pedro Pérez
"""

import numpy as np
import pandas as pd

def CalculateImageDimensions(axisFieldSizeX, axisFieldSizeY,img_cols):
    
    SizeX = axisFieldSizeX[1]-axisFieldSizeX[0]
    SizeY = axisFieldSizeY[1]-axisFieldSizeY[0]
    
    RelXY = SizeX/SizeY
    
    img_rows = int(img_cols/RelXY)
    
    print('img_rows: ', img_rows)
    
    return (SizeX,SizeY, img_rows)
    

def TranformDimensions(img_cols,img_rows,axisFieldSizeX,axisFieldSizeY, SizeX, SizeY, positionX,positionY):
    
    X = int(img_rows*(positionY-axisFieldSizeY[0])/SizeY)
    Y = int(img_cols*(positionX-axisFieldSizeX[0])/SizeX)
                
    return (X,Y)

# Prepare the data to receive more than 1 frame as input
def PrepareToLSTM(AllDataReadFrame, numEpochs):
    #Array with each n elements of the DataFrame as a row. n = numEpochs
    NewData = []
    #Variable to predict
    Labels = AllDataReadFrame['angle']
    #Eliminate the variable to predict from the original DataFrame
    AllDataReadFrame = AllDataReadFrame.drop('angle', axis=1)
    #Eliminate the first n elements, so the length of the label matches the length of the new data
    for i in range(0, numEpochs):
        Labels.pop(i)
    #Obtain rows with n DataFrame rows on them. n = numEpochs
    for i in range(len(AllDataReadFrame)):
        end = i + numEpochs
        if end > len(AllDataReadFrame)-1:
            break
        seq = AllDataReadFrame.iloc[i:end]
        #print(seq.describe())
        #print(i, " of ", len(AllDataReadFrame))
        NewData.append(seq)
    
    #Transform the array to a DataFrame format
    NewDataFrame = pd.DataFrame(NewData)
    
    #Return both the new dataframe and the labels
    return NewDataFrame, Labels
    
    
    
def TranformToImage (AllDataReadFrame,axisFieldSizeX,axisFieldSizeY,img_cols,\
                     colorsGray,positionsReferential,ColumnsName):
    
    
    SizeX,SizeY,img_rows = CalculateImageDimensions(axisFieldSizeX, axisFieldSizeY,img_cols)
    
    images = []
    
    image = np.ones((img_rows, img_cols))
    #print(image.shape)
    for index, row in AllDataReadFrame.iterrows():
        
        image = np.ones((img_rows, img_cols))
        
        
        for positionRef in positionsReferential:
            positionX = row[ColumnsName[positionRef[1][0]]]
            positionY = row[ColumnsName[positionRef[1][1]]]
            X,Y = TranformDimensions(img_cols, img_rows, axisFieldSizeX,axisFieldSizeY,\
                                     SizeX, SizeY, positionX,positionY)
            image[X,Y] = colorsGray[positionRef[0]]
            image[X+1,Y] = colorsGray[positionRef[0]]
            image[X-1,Y] = colorsGray[positionRef[0]]
            image[X,Y+1] = colorsGray[positionRef[0]]
            image[X,Y-1] = colorsGray[positionRef[0]]
            image[X+1,Y+1] = colorsGray[positionRef[0]]
            image[X-1,Y-1] = colorsGray[positionRef[0]]
            image[X-1,Y+1] = colorsGray[positionRef[0]]
            image[X+1,Y-1] = colorsGray[positionRef[0]]
                            
        images.append(image)
        
    return (images,img_rows)



def TranformtoGrades (AllDataReadFrame):
    
    
    AllDataReadFrame['angle'] = np.degrees(np.arctan (np.abs(AllDataReadFrame['jugador_avance_y']/
                                         AllDataReadFrame['jugador_avance_x'])))

    #x >0 ,y <0
    AllDataReadFrame['angle'] =\
                    np.where(np.logical_and(AllDataReadFrame['jugador_avance_x'] > 0,AllDataReadFrame['jugador_avance_y'] < 0),\
                    360.0-AllDataReadFrame['angle'],AllDataReadFrame['angle'])

    #x <0 ,y >0
    AllDataReadFrame['angle'] =\
                    np.where(np.logical_and(AllDataReadFrame['jugador_avance_x'] < 0,AllDataReadFrame['jugador_avance_y'] > 0),\
                    90+AllDataReadFrame['angle'],AllDataReadFrame['angle'])

    #x <0 ,y <0
    AllDataReadFrame['angle'] =\
                    np.where(np.logical_and(AllDataReadFrame['jugador_avance_x'] < 0,AllDataReadFrame['jugador_avance_y'] < 0),\
                    180+AllDataReadFrame['angle'],AllDataReadFrame['angle'])                    

    
    AllDataReadFrame['angle'] = AllDataReadFrame['angle']/360.
    
    
    # Create Angle to Ball variable
    
    '''
    BallForwardX = AllDataReadFrame['pelota_posicion_x'] - AllDataReadFrame['jugador_posicion_x']
    BallForwardY = AllDataReadFrame['pelota_posicion_y'] - AllDataReadFrame['jugador_posicion_y']
    
    AllDataReadFrame['AngleToBall'] = np.degrees(np.arctan (np.abs(BallForwardY/
                                         BallForwardX)))
    
    #x >0 ,y <0
    AllDataReadFrame['AngleToBall'] =\
                    np.where(np.logical_and(BallForwardX > 0,BallForwardY < 0),\
                    360.0-AllDataReadFrame['AngleToBall'],AllDataReadFrame['AngleToBall'])

    #x <0 ,y >0
    AllDataReadFrame['AngleToBall'] =\
                    np.where(np.logical_and(BallForwardX < 0,BallForwardY > 0),\
                    90+AllDataReadFrame['AngleToBall'],AllDataReadFrame['AngleToBall'])
                    
    #x <0 ,y <0
    AllDataReadFrame['AngleToBall'] =\
                    np.where(np.logical_and(BallForwardX < 0,BallForwardY < 0),\
                    180+AllDataReadFrame['AngleToBall'],AllDataReadFrame['AngleToBall'])  
                    
    AllDataReadFrame['AngleToBall'] = AllDataReadFrame['AngleToBall']/360.
    '''
    
    # Inverse the data along the Y axis
    
    AllDataReadFrame['angle'] = 1 - AllDataReadFrame['angle']
    
    #AllDataReadFrame['AngleToBall'] = 1 - AllDataReadFrame['AngleToBall']
    
    minY = np.amin(AllDataReadFrame['jugador_posicion_y'])
    maxY = np.amax(AllDataReadFrame['jugador_posicion_y'])
    
    AllDataReadFrame['jugador_posicion_y'] = (minY - AllDataReadFrame['jugador_posicion_y']) + maxY
    
    minY = np.amin(AllDataReadFrame['jugador_avance_y'])
    maxY = np.amax(AllDataReadFrame['jugador_avance_y'])
    
    AllDataReadFrame['jugador_avance_y'] = (minY - AllDataReadFrame['jugador_avance_y']) + maxY
    
    minY = np.amin(AllDataReadFrame['pelota_posicion_y'])
    maxY = np.amax(AllDataReadFrame['pelota_posicion_y'])
    
    print('MinY: ',minY)
    print('MaxY: ', maxY)
    
    AllDataReadFrame['pelota_posicion_y'] = (minY - AllDataReadFrame['pelota_posicion_y']) + maxY

    minY = np.amin(AllDataReadFrame['compañero_posicion_y'])
    maxY = np.amax(AllDataReadFrame['compañero_posicion_y'])
    
    AllDataReadFrame['compañero_posicion_y'] = (minY - AllDataReadFrame['compañero_posicion_y']) + maxY

    minY = np.amin(AllDataReadFrame['contrincante1_posicion_y'])
    maxY = np.amax(AllDataReadFrame['contrincante1_posicion_y'])
    
    AllDataReadFrame['contrincante1_posicion_y'] = (minY - AllDataReadFrame['contrincante1_posicion_y']) + maxY    
    
    minY = np.amin(AllDataReadFrame['contrincante2_posicion_y'])
    maxY = np.amax(AllDataReadFrame['contrincante2_posicion_y'])
    
    AllDataReadFrame['contrincante2_posicion_y'] = (minY - AllDataReadFrame['contrincante2_posicion_y']) + maxY    
    
    # Normalize the data
    
    # Player
    
    minX = -24.25
    maxX = 30.9
    
    minY = -15.6
    maxY = 12.5
    
    '''
    minX = np.amin(AllDataReadFrame['jugador_posicion_x'])
    maxX = np.amax(AllDataReadFrame['jugador_posicion_x'])
    
    print('minX: ',minX)
    print('maxX: ',maxX)
    '''
    
    AllDataReadFrame['jugador_posicion_x'] = (AllDataReadFrame['jugador_posicion_x'] - minX) / (maxX - minX)
    
    '''
    minY = np.amin(AllDataReadFrame['jugador_posicion_y'])
    maxY = np.amax(AllDataReadFrame['jugador_posicion_y'])
    
    print('minY: ',minY)
    print('maxY: ',maxY)
    '''

    AllDataReadFrame['jugador_posicion_y'] = (AllDataReadFrame['jugador_posicion_y'] - minY) / (maxY - minY)

    # Companion
    
    
    '''
    minX = np.amin(AllDataReadFrame['compañero_posicion_x'])
    maxX = np.amax(AllDataReadFrame['compañero_posicion_x'])
    
    print('minX: ',minX)
    print('maxX: ',maxX)
    '''

    AllDataReadFrame['compañero_posicion_x'] = (AllDataReadFrame['compañero_posicion_x'] - minX) / (maxX - minX)
    
    '''
    minY = np.amin(AllDataReadFrame['compañero_posicion_y'])
    maxY = np.amax(AllDataReadFrame['compañero_posicion_y'])
    
    print('minY: ',minY)
    print('maxY: ',maxY)
    '''

    AllDataReadFrame['compañero_posicion_y'] = (AllDataReadFrame['compañero_posicion_y'] - minY) / (maxY - minY)    

    # Ball
    
    '''
    minX = np.amin(AllDataReadFrame['pelota_posicion_x'])
    maxX = np.amax(AllDataReadFrame['pelota_posicion_x'])
    
    print('minX: ',minX)
    print('maxX: ',maxX)
    '''

    AllDataReadFrame['pelota_posicion_x'] = (AllDataReadFrame['pelota_posicion_x'] - minX) / (maxX - minX)
    
    '''
    minY = np.amin(AllDataReadFrame['pelota_posicion_y'])
    maxY = np.amax(AllDataReadFrame['pelota_posicion_y'])
    
    print('minY: ',minY)
    print('maxY: ',maxY)
    '''

    AllDataReadFrame['pelota_posicion_y'] = (AllDataReadFrame['pelota_posicion_y'] - minY) / (maxY - minY)

    # Enemy 1

    '''
    minX = np.amin(AllDataReadFrame['contrincante1_posicion_x'])
    maxX = np.amax(AllDataReadFrame['contrincante1_posicion_x'])
    
    print('minX: ',minX)
    print('maxX: ',maxX)
    '''

    AllDataReadFrame['contrincante1_posicion_x'] = (AllDataReadFrame['contrincante1_posicion_x'] - minX) / (maxX - minX)
    
    '''
    minY = np.amin(AllDataReadFrame['contrincante1_posicion_y'])
    maxY = np.amax(AllDataReadFrame['contrincante1_posicion_y'])
    
    print('minY: ',minY)
    print('maxY: ',maxY)
    '''

    AllDataReadFrame['contrincante1_posicion_y'] = (AllDataReadFrame['contrincante1_posicion_y'] - minY) / (maxY - minY)
    
    # Enemy 2
    
    '''
    minX = np.amin(AllDataReadFrame['contrincante2_posicion_x'])
    maxX = np.amax(AllDataReadFrame['contrincante2_posicion_x'])
    
    print('minX: ',minX)
    print('maxX: ',maxX)
    '''

    AllDataReadFrame['contrincante2_posicion_x'] = (AllDataReadFrame['contrincante2_posicion_x'] - minX) / (maxX - minX)
    
    '''
    minY = np.amin(AllDataReadFrame['contrincante2_posicion_y'])
    maxY = np.amax(AllDataReadFrame['contrincante2_posicion_y'])
    
    print('minY: ',minY)
    print('maxY: ',maxY)
    '''

    AllDataReadFrame['contrincante2_posicion_y'] = (AllDataReadFrame['contrincante2_posicion_y'] - minY) / (maxY - minY)
    
    
    #AllDataReadFrame['DistanceToBall'] = ((AllDataReadFrame['jugador_posicion_x'] - AllDataReadFrame['pelota_posicion_x']) ** 2) + ( (AllDataReadFrame['jugador_posicion_y'] - AllDataReadFrame['pelota_posicion_y']) ** 2 )
    
    #AllDataReadFrame['DistancePlayer1ToBall'] = ((AllDataReadFrame['contrincante1_posicion_x'] - AllDataReadFrame['pelota_posicion_x']) ** 2) + ( (AllDataReadFrame['contrincante1_posicion_y'] - AllDataReadFrame['pelota_posicion_y']) ** 2 )

    #AllDataReadFrame['DistancePlayer2ToBall'] = ((AllDataReadFrame['contrincante2_posicion_x'] - AllDataReadFrame['pelota_posicion_x']) ** 2) + ( (AllDataReadFrame['contrincante2_posicion_y'] - AllDataReadFrame['pelota_posicion_y']) ** 2 )

    #AllDataReadFrame['DistanceCompanionToBall'] = ((AllDataReadFrame['compañero_posicion_x'] - AllDataReadFrame['pelota_posicion_x']) ** 2) + ( (AllDataReadFrame['compañero_posicion_y'] - AllDataReadFrame['pelota_posicion_y']) ** 2 )

    #AllDataReadFrame['angle'] = AllDataReadFrame['angle'].replace({np.nan:-1.0}, regex = True)
    #AllDataReadFrame['angle'] = AllDataReadFrame['angle'].replace({0.0:0.5}, regex = True)
    AllDataReadFrame = AllDataReadFrame.dropna()
    
    print('Any nan: ', AllDataReadFrame.isnull().values.any())


    
    #x = AllDataReadFrame['angle']
    
    #x = x[~np.isnan(x)]
    
    #AllDataReadFrame['angle'] = x

    """
    print('------------------ angle > 0')
    print(AllDataReadFrame[['jugador_avance_x','jugador_avance_y','angle']].query('angle != -1'))
    """
                    
       
    return (AllDataReadFrame)
