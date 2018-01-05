# Youtube-Video-Label-Classification
Youtube Video Label Classification using Single Frame model and Long-term Recurrent Convolutional Networks (LRCN) model

Objective
Aim of the project is to develop classification algorithms which accurately classify multiple labels of videos.

Problem
There are billions of videos available on Internet and billions of videos are added every year on Internet. These videos don’t necessarily have labels embedded with them. Therefore, there is no easy way for websites like YouTube, Dailymotion, Hulu, etc. can recommend these videos to User’s based on their interest or based on similar video criteria. It is therefore necessary to classify these videos beforehand (single or multiple labels) based on its type and content. Thus, classifying videos is important and presents unique challenge for machine learning models.

Brief Introduction
In this report, we first review the related work of existing benchmarks for image and video classification. We then present the details of our dataset. Then we focus on current approaches and our approach in this project. Further, we evaluate the approaches on the models. Finally, we discuss regarding our future work. 

YouTube-8M Dataset
YouTube-8M is a large-scale labeled video dataset that consists of millions of YouTube video IDs and associated labels from a diverse vocabulary of 4716 visual entities(classes).

The dataset contains videos and labels that are publicly available on YouTube, each of which must satisfy the following:
Must be public and have at least 1000 views
Must be between 120 and 500 seconds long
Must be associated with at least one entity from YouTube target vocabulary

Some of the key highlights about the dataset are:
This dataset contains over 7 million YouTube videos
Video Files are stored in tfrecord format files
Each tfrecord file has approximately 1200 videos
There are total of 4716 labels (each video with multiple labels)
Each video can have up to 10 labels

The following are the top 10 labels in the Dataset
Games
Arts and Entertainment
Autos and Vehicles
Computer and Electronics
Food and Drink
Business and Industrial
 Sports
Miscellaneous
Pets and Animals
Science

Dataset can be downloaded using the following weblink:

https://research.google.com/youtube8m/download.html

Approaches
The approaches that we followed in our project are as follows:
Approach A: Classifying one frame at a time with a CNN (Single Frame Model)
Approach B: Long term Recurrent Convolutional Networks (LRCN)

Approach A: Classifying one frame at a time with a CNN (Single Frame Model)

For the 2D CNN we have ignored temporal features of videos and attempted to classify each clip by looking at a single frame 
As part of demo and experiments we have only used approximately 720,000 training examples(videos) (approximately 16% of the available training data)
We have used two types of optimizers namely: “rmsprop” and “adam” 
RmsProp is an optimizer that utilizes the magnitude of recent gradients to normalize the gradients whereas Adam is an optimizer which instead of adapting the parameter learning rates based on the average first moment (the mean) as in RmsProp, it makes use of the average of the second moments of the gradients (the uncentered variance).
For pooling we have used max pooling which uses the maximum value from each of a cluster of neurons at the prior layer.
We have used ReLU (Rectified linear unit) as the activation function for each convolutional block.
Global Average Precision score on the validation set obtained is 0.5924987350077567

Approach B: Long term recurrent convolutional networks (LRCN) 

First, we ran every frame from every video through inception, thus saving the output from the final pool layer of the network. 
Then, we converted the extracted features into sequence of extracted features and turned each video into 120 frame sequence.
Finally, we passed on this sequence to train RNN model without needing to continuously pass our images through CNN every time we read the sample or when we trained a new network architecture.
For the RNN, we used a single, 4096-wide LSTM layer, followed by a 1024-wide dense layer with some dropout in between.

Validation and Testing

We have used Holdout Method for validation for the following reasons:
We have an enormous amount of data to train on. So, we aren’t losing out on the number of available examples.
It takes huge amount of time for a single iteration of training and testing so using k-fold cross validation is not a good option.

Testing has been done on completely unseen dataset provided by Google.
Evaluation Metrics

Global Average Precision score is used as evaluation metrics for our project. Accuracy is not a good evaluation metric as we can have a lot false-positives and false-negatives. Moreover the number of labels to be predicted is not known.

Cost Incurred

We have used Google Cloud Machine learning platform which gives $300 credit free for first 12 months and we used approximately $57 of instance hour credits to do the computation for both models.


Software and Hardware

We have used Python platform for implementation due to the availability of several popular Deep Learning libraries like Tensorflow to load the tfrecords and Keras that helped in creating CNN for the Single Frame Model and LRCN model. Also, we have used a 16GB RAM computer system to execute our implementation on the dataset.

Software: Python using libraries - numpy, pandas, tensorflow and keras
Hardware: ThinkPad T460p with i7-6820HQ processor and 16GB of RAM

Conclusion

Approach B using LRCN gives a better Global Average Precision score than Approach A using Single Frame Model but it is computationally more expensive and time consuming to train a model because it requires huge amount of memory to store all the frames of each video.

References

[1]
S. Abu-El-Haija, N. Kothari, J. Lee, P. Natsev, G. Toderici, B.  Varadarajan, and S.  Vijayanarasimhan. Youtube-8m: A   large-scale   video   classification   benchmark. CoRR, abs/1609.08675, 2016.
[2]
Po-Yao Huang, Ye Yuan, Zhenzhong Lan, Lu Jiang, Alexander G. Hauptmann. Video Representation Learning and Latent Concept Mining for Large-scale Multi-Label Video Classification.
[3]
Jeff Donahue, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan, Sergio    Guadarrama, Kate Saenko, Trevor Darrell. Long-term Recurrent Convolutional Networks for Visual Recognition and Description.
[4]
Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vijayanarasimhan, Oriol Vinyals,
Rajat Monga, George Toderici. Beyond Short Snippets: Deep Networks for Video Classification
[5]
Christian Szegedy, Wei Liu, Chapel Hill, Yangqing Jia, Pierre Sermanet, Scott Reed,Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich. Going deeper with convolutions.

