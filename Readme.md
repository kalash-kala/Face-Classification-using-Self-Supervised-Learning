# Face Classification using Self Supervised Learning

## Problem Statement :

- An Attempt to find a method to make personalised face classification model

## Motivation :

- Creating a generalised model for Face Classification can be tricky as finding labelled data for so many people is very cumbersome
- But we have lots of unlabelled face data on the internet
- We can make use of the fact that faces have similar structure and can be grouped into clusters based on certain properties.
- So in order to create face classification model for a small community of people, we can extract features / representations from the unlabelled data and update it with respect to the little labelled data that we have

## What is Self Supervised Learning (SSL) :

- Self Supervised Learning (SSL) is a type of machine learning technique where a model learns to understand the underlying patterns and structures in the data without relying on explicit labels or annotations.
- Instead, it leverages the inherent structure of the data itself to create useful representations or features. This is typically done by defining pretext tasks, such as predicting missing patches in an image or generating contextually relevant embeddings, and training the model to solve these tasks.
- SSL has shown great potential in various domains, including computer vision, natural language processing, and speech recognition.

## Difference between Supervised and Self Supervised Learning :

### Supervised Learning :

- In this learning paradigm we are provided with the actual labels of the input
- Using the provided labels, we can divide the data into training, test and validation dataset and train the model
- Labelling the data is a cumbersome process requiring lots of human intervention, due to its time consuming nature, it could lead to erroneous labelling of the input data
- Due to this problem, finding lots of labelled data can be difficult, especially if you are working on a niche topic.

### Self Supervised Learning :

- In the modern day and age, there is no limit to the amount of unlabelled data that we have, but while training a model we need labelled data so that we can not only train the model but also to check if its working or not
- To solve this issue, Self Supervised Learning  paradigm uses the unlabelled data to learn the representation / distribution of the unlabelled data and uses it while training on less quantity of labelled data.
- These underlying Representation can be learnt by applying the pretext tasks like
    - Masked Predictions
    - Transformation  Predictions
    - Clustering
- Using the above pretext tasks we generate Pseudo labels $z$ which are then used to learn representation by running the model over it with the unlabelled data
- This representation is then used in the Downstream task, where we only change the classifier layer with respect to our need and train the model on the little labelled data that we have
- This leads to faster convergence of the model (due to pre training on unlabelled data)

![Screenshot 2023-10-22 at 4.22.53 PM.png](Face%20Classification%20using%20Self%20Supervised%20Learning/Screenshot_2023-10-22_at_4.22.53_PM.png)

### Visualisation of the Self Supervised Learning process in a nutshell

![Screenshot 2023-10-22 at 4.24.00 PM.png](Face%20Classification%20using%20Self%20Supervised%20Learning/Screenshot_2023-10-22_at_4.24.00_PM.png)

## Difference Learning between Transfer Learning and SSL :

### Transfer Learning :

- In this paradigm, we use the weights of large pre trained models (Resnet and etc) to train our model
- The difference between this and SSL lies in the fact that in Transfer learning we don’t update the weights which we are borrowing from the large models, while in SSL, we not only borrow the representations from the pre training of unlabelled data, but also update it while training on the little training data that we have

## Pretext Task :

### Clustering :

- For our use case, I have used Clustering as a Pretext task to generate Pseudo Labels
- For Clustering the input, I have flattened the face image and used K means clustering to divide the data into 10 clusters
    
    ![Screenshot 2023-10-22 at 4.31.56 PM.png](Face%20Classification%20using%20Self%20Supervised%20Learning/Screenshot_2023-10-22_at_4.31.56_PM.png)
    

### Why Clustering ?

- Clustering seemed a logical choice because, every face has some similarity with other faces like same complexion, eye colour, facial structure and etc. Hence in order to generate some meaningful pseudo labels, clustering seems to be the right choice

## Process :

### Data Preprocessing :

- Extracting unlabelled data and labelled data from the Labelled Faces in the Wild
- Created a PyTorch dataset using the CustomDataset class
- Created DataLoader for Test / Training and Validation data

### Pseudo Label Generation :

- Using K means to get Pseudo Labels from Unlabelled data

### Model Architecture :

- Created TinyVGG Architecture for model

### Training SSL :

- Self Supervised Pre training of Unlabelled data and Pseudo Labels
- Used to pre trained feature extractor from the Self Supervised Pre training for the downstream task of classifying the labelled data
- Measured and Plotted Accuracy and Loss

### Training Without SSL

- Create a new model with the TinyVGG Architecture
- Run the model directly with the labelled data
- Measured and Plotted Accuracy and Loss

### Training the Transfer Learning model

- Create a model using Pre trained Efficient Net model
- Change the classifier of this model
- Train the model with the labelled data
- Measured and Plotted Accuracy and Loss

## Results :

- The SSL model gave similar Accuracy to the Supervised model in half the epochs
- Transfer Learning model performed poorly

## References :

- [Self-Supervised Representation Learning: Introduction, Advances and Challenges - Linus Ericsson, Henry Gouk, Chen Change Loy, Timothy M. Hospedales](https://arxiv.org/abs/2110.09327)
- [Labeled Faces in the Wild (LFW) Face Dataset](http://vis-www.cs.umass.edu/lfw/)
- [Tiny VGG Architecture](https://www.researchgate.net/figure/Main-architecture-of-VGGFace-network_tbl1_339634041)