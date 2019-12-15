# ai-final-project
Final Project for CS-315

To run:
Open app.py, uncomment the model you wish to evaluate, then run that file. 

Translator.py
In our translator.py file, we perform all of the data wrangling methods necessary for the data to be fed into each of our models. The data is very diverse containing ints, floats, strings, etc. We used the python module pandas throughout this script. First, we read the csv (downloaded from kaggle.com) so that we can create a pandas dataframe. Next we convert the column names to lowercase, so that each form of entry is uniform. 

We also convert the UTC time of snap and handoff columns to classify what time of the game an individual play is happening. We wrote a function named encode_game_time that accepts our data and a column. This function splits the data on the ‘T’ and ‘.’ in the UTC time format and uses the other parts of the entry like the hour, minute, second, and quarter. The first thing we had to consider was that the UTC time was 5 hours ahead of EST that NFL times are set on. We established a classify to put the games into 3 categories, noon, afternoon and night, represented as a 0, 1, or 2.

In our analysis of the overall problem we were trying to solve, as a group we decided that some of the columns in the dataset provided from kaggle wouldn’t be particularly relevant in predicting our final target. We decided to drop these columns from the dataframe so that we weren’t feeding data into the model that we didn’t think was relevant for prediction. These columns included 'windspeed', 'winddirection', 'temperature','gameweather', 'stadiumtype', 'nflid','gameid', 'nflidrusher', and 'humidity'.

In parsing through the database, we also noticed that there were some columns that were missing information for certain cells. Three of these columns included ‘orientation’, ‘offenseformation’, and ‘direction’. To fix this issue, we decided to fill the empty cells in ‘orientation’ and ‘direction’ with the mean value for that column, since those columns contained numerical decimal values and thus using the average value seemed like the best way to represent the data. We decided to use the mean value since we figured that it would likely give the best representation for the missing information, rather than taking a maximum or minimum value. However, for the offensive formation, we decided to use the most frequent value in the column (mode) since there were a discrete number of options for the formation, and we felt the most commonly used formation would likely give the best representation.

Some columns that only contain two different values for instance like team, fieldposition, and hometeamabbrev, we decided to one-hot encode the values with one value as 1 and  the other value as 0.

Additionally, we also needed to encode the data for the gameclock column in order to represent as a single integer number of seconds rather than a time with minutes and seconds. We also encoded the player height into a single integer representation, converting the feet and inches into a single integer value of inches. Similarly, we also encoded birthdates of the players by subtracting the date represented in the dataframe from the current date in order to calculate the number of days that they have been alive (age), which we thought would be a more relevant representation

Once we had encoded all of the data that we wanted to keep, we created an array of all of our target values (yards gained on a particular play). Since each play is represented by 22 rows in the database (1 row for each player on the field during the play), we decided that we should condense each 22 rows into a single representation to pass to the model rather than feed in each player separate from one another. Thus, instead of needing a target value for each row, we only needed a target value for each play. So we created an array to keep track of all of the target values, however only included the target for each play once in the array rather than including the same yards gained for a play 22 different times. To do this, we simply iterated through the rows in the database, taking every 22nd target value to add to the array. 

Once we had gathered the target values from the database, we then needed to grab the play id column so that we could condense the players in a single play down into a single representation for the model. We dropped the ‘yards’ and ‘playid’ columns from the database as they were no longer needed, and didn’t need to be presented to the model as part of the input. We then condensed each 22 players for each play into a single, flattened array of length 22 to be fed into our models. To do this we compared rows that contained the same playID, and added the flattened row to an array containing the data for each individual play. Once we encountered a new playid, we added the array containing all of the data for the single play to our large array of the data for every play, and reset the array containing the data for the single play. We continued this process, iterating over every row in the database. Once we had obtained the flattened representation of the dataframe for each play, we returned the model inputs along with the labels to be passed into our models.

Eval.py
When deciding how we wanted to evaluate our model (i.e. what metric to use), we decided that we would use the same metric that was being used for the kaggle Big Data Bowl competition. 
https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview/evaluation
The submissions were evaluated based on the Continuous Ranked Probability Score. For each play in the database, the model was supposed to predict the probability of the team obtaining less than x number of yards for -99<=x<=99. If the model predicted that the team would gain less than x number of yards, then it would return a 1, else it would return a 0. Likewise, if the actual number of yards was less than x number of yards, then the model would return a 1. Thus, the goal of the model was to predict a team would gain more than x number of yards for all x that was less than the actual number of yards, and predict that the team would gain less than x number of yards for all x that was greater than or equal to the actual number of yards gained. Thus for each integer increment of yards that was predicted incorrectly, the model received a one. The sum of these incorrect predictions was taken across all numbers of yards from -99 to 99 for all plays, and then was divided by 199*number of plays in order to calculate the score for each model. Since keras has no metric for this scoring, we needed to write our own custom metric for the model to implement. To do so, we took in the actual number of yards gained as well as the prediction from our model. Then for the range from -99 to 99, we compared whether our model predicted x number of yards to be gained from the model, and whether x was greater than the actual number of yards gained. We added up the total number of these entries, squared it to remove the effect of negatives, and then returned the total divided by 199. Then the keras accuracy metric took the sum of all of these values returned and divided by the number of plays in order to return our overall CRPS.


Models:
In our project, we are attempting to predict the number of yards gained by a rusher on play given the data of each play. Our first attempt to model this target function entailed using a feed-forward neural network. We have now expanded our project to include a number of different networks and ways to predict the target using deep learning. We have created a LSTM network, 2-dimensional CNN, and an Autoencoder. Our goal is to definitively say whether or not deep learning is a useful technique in solving our problem.

FFNN
	As our original model, we used the FFNN as a learning platform to explore the data and its complications. In our first iteration, we used a shallow fully connected feed forward neural network. As input, we fed one row of data to the model at a time. The hidden layer comprised of 22 nodes and an activation function of Leaky Relu. We chose to use Leaky Relu to properly model the possibility of a player rushing for negative yards. The output layer consisted of one node, also with a Leaky Relu activation function. Since the target output is the number of yards a rusher gained, an output layer of one paired with the Leaky Relu function made sense. We also implemented a dropout rate of 30% in order to make the model more robust and limit dependencies on a single path through the network. Despite continuous tuning of hyperparameters, we were not confident in the accuracy we were getting out of this model. These initial struggles prompted us to explore other model architectures and fundamentally change the way our FFNN was laid-out.
	Our final FFNN model consists of an input layer with the number of nodes equal to the size of the input (all 22 player rows for each play). The first hidden layer has double the number of input nodes, with the Leaky Relu activation function and Batch normalization. By normalizing the input data, Batch Normalization allows for faster training and reduces the possibility of vanishing gradient. The second hidden layer is the same as the first hidden layer. We chose to implement a second hidden layer in order to extract more abstract features from the data in order to make a more accurate prediction. The output layer of the model consists of one layer with the Leaky Relu activation function.
We used the Adam optimizer, the Mean Squared Error loss function, and the CRPS metric. 
The main reason that the FFNN did not perform as well as other models is the complex nature of the data. By representing each play as one input to the model, there were some repeated pieces of data, and a lack of consistency in position order through multiple plays. This prevented the model from learning proper dependencies and gaining a low CRPS score. 

LSTM
We are feeding the dataset into an embedding layer 15500(I am not sure what the error was but once I increased the maxFeatures size which is the embedding size, it began to work). We also started with an LSTM layer of size 40 and a dropout rate of 30%. The last layer is a single node dense layer with Leaky Relu activation function. Leaky Relu seems the most appropriate for our problem given that a person can either rush for positive yards or negative yards on any play. The default learning rate of the Leaky Relu function is 0.3. We kept the learning rate at the default for the first and second runs of the tuning phase. The accuracy after the first run of the LSTM was 9.16%. On the second run, I changed the output dimension of the Embedding layer and and the units of the LSTM to 200. Then I ran the program. For the third run, I have change the activation function within the LSTM layer to Leaky Relu. This still lead to a low accuracy. Then we added another LSTM layer to the architecture, since we had a lot of data. Due to the condition of the data, the sequential aspect of the data was not prominent in the data which was leading us to a very low accuracy. Note that each of the runs was with the data being formatted as one row containing information for one player. So, theoretically, each set of 22 rows contain the information for a given play(I say theoretically because some rows have NA values which are being dropped at that time. We eventually address the missing values with another method.). We decided to change the data so that each row contained all the information for a given play. 
	After changing the data and the evaluation metric, we began to test the LSTM model again and received a large increase in the CRPS accuracy. The change allowed the model to handle data as grouping 22 rows by 38 columns which displays each player on the field and the data associated with the play. We initially began testing with a deep, deep LSTM network and quickly realized that we should not be using that architecture due to a lack of data. An LSTM layer requires a lot of data and a deep, deep LSTM requires even more data. Thus the deep, deep model was unable to create enough dependencies to lead to a correct classification. It received close to a .035 CPRS accuracy. Our next step was to drop the second LSTM layer and begin tuning the LSTM model. We initially began around a .025 CPRS accuracy and tuned the model until it received 0.205 CPRS accuracy. The changes made to lead to the best accuracy include changing the batch size from 128 to 64, increasing the LSTM layer to 200, and reducing the epochs to 10. With those changes, we were able to achieve a model with .0205 CPRS accuracy. The most interesting part was no matter what change we made at after getting this accuracy the model never increased. We believe this is due to the lack of data. The changes we made to get the best accuracy all were due to the lack of data. So it would make sense that we maxed out arbitrarily increasing the amount of data and hit the best possible accuracy. 
	As for the code for the LSTM, there was no embedding layer because the translate file does all the embedding. We are also using mean squared error as the loss function, because we are predicting one number at the end and we want the error of that number to be propagated back to the weights so that the weight matrix can be adjusted properly. 

Autoencoder
We also implemented an autoencoder which was fed into the feed forward neural network. Our thought process with this approach was that the autoencoder could identify the most prominent features contained within the dataset, create a smaller latent representation containing the most prominent features, and perhaps increase the accuracy of the models.
	To be precise the autoencoder is a deep autoencoder consisting of three encoder layers and 3 decoder layers. We trained it using one hundred epochs with leakyRelu as the activation function. We also used mean squared error as the loss function. We extracted the latent representation and fed it into the feed-forward. The accuracy was extremely low. We tried adjusting the parameters of both the autoencoder and the feed-forward to increase the accuracy, but it was to no avail. We believe the autoencoder reduced the information too much. Perhaps, the network needs most or even all of the columns to make the correct classification. This section is rather short given that we did not explore this idea fully due to the other networks performing rather well. But we wanted to note this idea as something we attempted and found that the results were not as good as the others due to the reduction of information needed for classification.

CNN
We also used a two-dimensional CNN as one of our models for comparison. We were able to present the model with a matrix which represents the data we have for each play. Therefore, it has a shape of (22, 38), since we have data for each player on the field and have 38 different pieces of information corresponding to each. We were able to achieve a CRPS score of 0.0196 using this model.
	We used 32 filters for the first layer and increased the number of filters as the model extracted more complex features. Initially, the model was overfitting to the training set so we introduced a variety of normalization techniques to reduce the effects of this problem. We were able to achieve the best results using a combination of Dropout, Kernel Regularization, and Batch Normalization. We found the best results using a Dropout rate of 0.3 and the l2 kernel regularize with a decay of 1e-4.

Output

