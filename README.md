# Colorization-of-black-and-white-images-with-hint-using-deep
# This project is done by <a href="https://github.com/Devarsh23 ">Devarsh Patel</a>  , <a href="https://github.com/shubhambavishi">Shubham Bavishi</a> , <a href="https://github.com/Rutviklathiya">Rutvik Lathiya</a> And <a href="https://github.com/fallen2112">Hiten Patel</a>
## Table of content
[1. Demo](#demo-of-the-output) <br />
[2.The overview of this repository](#the-overview-of-this-repository) <br />
[3.Motivation behind the project](#motivation-behind-the-project) <br />
[4.To Do](#to-do) <br />
[5.Directory structure](#directory-structure) <br />
[6.Detailed Description of code](#detailed-description-of-code) <br />
[7.Special Thanks](#special-thanks) <br />



# Demo Of the output
## Before

![Image of Line](https://github.com/Devarsh23/Colorization-of-black-and-white-images-with-hint-using-deep-learning/blob/master/Output/line2.png) ![Image of Line1](https://github.com/Devarsh23/Colorization-of-black-and-white-images-with-hint-using-deep-learning/blob/master/Output/line3.png)



## After

![Final Image](https://github.com/Devarsh23/Colorization-of-black-and-white-images-with-hint-using-deep-learning/blob/master/Output/color2.png) ![Final Image1](https://github.com/Devarsh23/Colorization-of-black-and-white-images-with-hint-using-deep-learning/blob/master/Output/color3.png)


# The overview of this repository
Describes about the usefulness of deep learning and computer vision to colorize the black and white pictures with hint <br />
# Motivation behind the project:
The picture without a colour is like a boat without a helm. This sentence specifies the importance of colour in aspect of viewing picture so we decided to perform this project by which we can provide a better alternative coloured image despite of black and white. The added feature, here is that we can give the hint to the various areas to black and white to colour according to that hint. <br />
So, in such situation by using the technology like deep learning  we can help the people in certain public area to colourized their old photos. Hence the aim of this project is to colourized the black and white image. <br />
# To do 
Basically, you can run the colourization.ipynb file after cloning this GitHub repository. <br /> 
If you want to incorporate this with the outer camera the you can add the url + “/video” in the VideoCapture argument to use it on any mobile with ipwebcam or cctv camera. <br />
Dataset can be downloaded from this <a href="https://cocodataset.org/#home">Link</a><br/>
To download pretrained model click <a href="https://cocodataset.org/#home">Here</a> for draft model and <a href="https://cocodataset.org/#home">Here</a> for refined model
# Directory Structure
The directory structure here is very simple it includes the Folder named Model which includes two python files which contains various classes in it to help both the model to train. It also include Preprocess folder which includes python file named dataset.py which helps to prepreocess the dataset and change it in to the model input specific dataset.
Other than this it contains train python file named refined_train and draft_train to actually train both the required model. Hyperparameter.py and subnet.py are the supportive file which indicates the halping variable and function useful for training purpose. Lastly, it includes the test file to test our ready model for real world scenario.
# Detailed Description of code
For detailed understanding follow the python notebook named Colorization this will help you to understand whole code and also it will recapitulate the whole code for you in to one place in chronological order. 
# Special Thanks
We would like to thanks krish naik for encouraging us to do such a lovely project of colorization.




