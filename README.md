# handWrittenDigitRecognition

Simple LENet architecture was used to train the model on the MNIST dataset.

Model skeleton --> (CONV => ACTIVATION => POOL layers) * 2 => DENSE => ACTIVATION => DENSE => SOFTMAX

The input image is segmented to separate each digit in the image. The Segmented ROI is passed through the trained model to predict the class of the ROI. Then the predicted classes are combined to get the entire number.

train_model.py -- To train the model using MNIST dataset and the model is saved as trained_model.h5

architecture.py -- The architecture of the LENet model, making it easier to define a model as and when required

utility.py -- Contains utility functions

predict.py -- Loads the trained model from the disc and predicts the hand written number

Run Instructions
1) pip install -r requirements.txt		# to install all the required dependencies
2) python train_model.py				# to train the model on MNIST dataset using LENet architecture
3) python predict.py --image <entire_path_to_image_to_be_predicted>       # pass the path of the image that has to be predicted as a command line argument (script will not work without the path)

Finally output will be saved as output.jpg
