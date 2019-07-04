# Bachelors Thesis
Public repository of my Bachelor's Degree Thesis. The work's title is: "Artificial Intelligence for Videogames with Deep Learning", and the final version of the document can be accesed in the pdf attached to this repository. The rest of the code files compose the Deep Learning development pipeline developed to support the experiments carried out during this research, and can be differenciated in two parts: the C# code, responsible of the integration of Deep Learning models on the Unity Engine, and the Python code, where the Deep Learning agents were created, trained and saved in the correspondent data formats.

# Artificial Intelligence for Videogames with Deep Learning
Nowadays, videogames are commonly used in the Deep Learning field as a research tool, since it allows the researchers to easily identify which agents are performing better than others using the in-game score for that purpose. In this project, we have proposed to develop Artificial Intelligence for Videogames with Deep Learning techniques, changing the approach from videogames as a research tool to Deep Learning as a tool for Videogame developers to develop the Game AI.

For that reason, in this document first we have researched both the current state-of-the-art of both Game AI since its conception, and the Deep Learning solutions that used or are related to videogames. Then, we developed a Deep Learning Development Pipeline, capable of creating and training Deep Learning models in Python using Tensorflow and Keras, and integrating them inside an Android videogame in Unity. Finally, we developed many different neural networks, following different approaches: Artificial Neural Networks, Convolutional Neural Networks and Long Short-Term Memory Networks.

# C# Code

The C# Code of this project, composed of a single file called CNNInput, is the responsible of the integration of the Deep Learning models developed in Unity. For that purpose, it uses TensorflowSharp, an open source binding library to work with Tensorflow on C#. The code in that file is also responsible of building the input the network receives in each frame, and of supporting various debug options. The whole C# code of this project was developed by Adrián Francés, owner of this repository.

# Python Code

The Python Code is responsible of obtaining and curating the dataset used for training, transforming the input into the desired format, defining the Neural Network models on Keras (using Tensorflow as a background), saving the models in the desired format and obtaining the data necessary to analyze the results of each agent. The Python code of this project was developed by Adrián Francés and Pedro Perez, from From The Bench Games.
