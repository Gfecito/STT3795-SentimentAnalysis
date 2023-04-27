# STT3795-SentimentAnalysis

Our experiments currently work on Linux, and Windows. Mac has problem with Tensorflow.
The experiments run on Python 3.10.9. Later versions ought to work fine as well.

IMPORTANT! Currently our experiment only supports Logistic Regression as Tensorflow seems to have some issue that we haven't been able to solve yet. Notebooks will be available to run the other experiments.

## To replicate our experiments, please follow the following steps:
  ### 1) Create a virtual environment first (optional) with the following shell command:
      python3 -m venv venv 
   ### and activate it using: 
      source venv/bin/activate

  ### 2) Install all dependencies with: 
      pip3 install -r requirements.txt

  ### 3) If the data is not installed, download it at the following URL : https://www.kaggle.com/datasets/kazanova/sentiment140
     
  ### 4) To run our experiments, it is essential to modify the dataset path in the state.json file where the hyperparameters are defined. You should make sure it matches the path of the dataset in your system. You're also free to change any of the other hyperparameters to your liking.

  ### 5) After that, you only need to execute the command: 
      python3 experiment.py -p state.json

