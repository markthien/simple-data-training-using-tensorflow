# simple-model-training-using-tensorflow
This is a tutorial to guide you how to train your model with data in CSV using tensorflow. After training, you can use some test data to see if your model train correctly. You need to configure epoch, batch size and etc in order to optimise your training.

We will use an simple case where a person is defined to be healthy based on sex, age, number of gram of meat consume each day, number of gram of vegetable consume each day and number of meter walking each day. For example, female age 20 consider healthy if she is taking 100 gram of meat everyday, 10 gram of vegetable everyday and walking 500 meters everyday. However, a female age 20 consider not healthy if she is taking 500 gram of meat everyday, 10 gram of vegetable everyday and walking 500 meters everyday.

I have preapred 60 rows of dataset to train my model.

You need to setup tensorflow in your local machine by following the installation guide here https://www.tensorflow.org/install/

This tutoial consist of 5 files, which are
<ul>
<li>dataset.csv - for training our model</li>
<li>config.json - configuration file for our python program</li>
<li>healthTraining.py - for training our model</li>
<li>isHealthy.py - for testing or predicting our model</li>
<li>testData.csv - test data for trained model</li>
</ul>

My finding
If you have less data to train your model, then you need to increase epoch in order to better prediction.
If you have a lotof data to train your model, then you need less epoch and your prediction should be pretty good.

Most of codes here credited to git repo here https://github.com/tflearn/tflearn/blob/master/tutorials/intro/quickstart.md
I have split tlearn into 2 program which are training and predicting. In addition, I also make it configurable using json format.
