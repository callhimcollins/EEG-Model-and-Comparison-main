#Version: v0.1
#Date Last Updated: 12-20-2023

#%% STANDARDS   -DO NOT include this block in a new module
'''
Unless otherwise required, use the following guidelines
* Style:
    - Sort all alphabatically
    - Write the code in aesthetically-pleasing style
    - Names should be self-explanatory
    - Add brief comments
    - Use relative path
    - Use generic coding instead of manually-entered constant values

* Performance and Safety:
    - Avoid if-block in a loop-block
    - Avoid declarations in a loop-block
    - Initialize an array if size is known

    - Use immutable types
    - Use deep-copy
    - Use [None for i in Sequence] instead of [None]*len(Sequence)

'''

#%% MODULE BEGINS
module_name = 'DTmodel'

'''
Version: <***>

Description:
    <***>

Authors:
    <***>

Date Created     :  <***>
Date Last Updated:  <***>

Doc:
    <***>

Notes:
    <***>
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports


#other imports
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here



#Function definitions Start Here
def main():
    pass
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define the channels
channels = ['P3', 'P4', 'Pz']

# Function to load, preprocess, and split the data
def load_preprocess_split(channel):
    # Load the data
    train_validate_set = pd.read_csv(f'INPUT\\TrainValidateData{channel}.csv')
    validate_set = pd.read_csv(f'INPUT\\ValidateData{channel}.csv')

    # Concatenate the dataframes
    all_data = pd.concat([train_validate_set, validate_set], ignore_index=True)


    # Shuffle the data
    all_data_shuffled = all_data.sample(frac=1).reset_index(drop=True)


    # Split the shuffled data
    train_size = int(0.6 * len(all_data_shuffled))
    validate_size = int(0.2 * len(all_data_shuffled))

    train_data = all_data_shuffled[:train_size]
    validate_data = all_data_shuffled[train_size:train_size + validate_size]
    test_data = all_data_shuffled[train_size + validate_size:]

    # Data Preprocessing
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_data.iloc[:,1:-1].values)
    validate_X = scaler.transform(validate_data.iloc[:, 1:-1].values)
    test_X = scaler.transform(test_data.iloc[:, 1:-1].values)  

    train_Y = train_data.iloc[:,-1].values
    validate_Y = validate_data.iloc[:, -1].values
    test_Y = test_data.iloc[:, -1].values

    # Read and preprocess the TestData.csv file
    test2 = pd.read_csv(f'INPUT\\TestData{channel}.csv')
    test2_X = scaler.transform(test2.iloc[:, 1:-1].values)  
    test2_Y = test2.iloc[:, -1].values
    
    channel= channel
    return train_X, train_Y, validate_X, validate_Y, test_X, test_Y, test2_X, test2_Y, channel

        
#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")

# Loop over the channels
for channel in channels:
    train_X, train_Y, validate_X, validate_Y, test_X, test_Y, test2_X, test2_Y, channel = load_preprocess_split(channel)

    # Initialize the model
    clf = DecisionTreeClassifier()

    # Model Training
    clf.fit(train_X,train_Y)

    # After training the model, save the model parameters
    with open(f'Model/Parameters/DTParam_{channel}.pkl', 'wb') as f:
        pickle.dump(clf, f)

#Test Using Pre-Trained Model
#-------------------------------------------------------------------------------------------------------------------------------------
    with open(f'Model/Parameters/DTParam_{channel}.pkl', 'rb') as f: clf_loaded = pickle.load(f)

    #Generating Prediction on first test
    prediction = clf_loaded.predict(test_X)

    # Evaluate the model on the first test set
    test_accuracy = metrics.accuracy_score(test_Y, prediction)
    print(f"Test1 {channel} accuracy:",test_accuracy)
    
    #Generating Prediction on second test
    prediction2 = clf_loaded.predict(test2_X)
    
    # Evaluate the model on the second test set
    test2_accuracy = metrics.accuracy_score(test2_Y, prediction2)
    print(f"Test2 {channel} accuracy:",test2_accuracy)
    
    # Compute confusion matrix
    cm = confusion_matrix(test2_Y, prediction2)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for {channel}')
    plt.savefig(f'OUTPUT/DT/confusion_matrix_{channel}.png')  # Save the figure
    plt.close()

    # Calculate the metrics
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    # Create labels for the metrics
    labels = ['Sensitivity', 'Specificity', 'Precision', 'F1 Score']

    # Create values for the metrics
    values = [sensitivity, specificity, precision, f1_score]

    # Plot metrics
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['blue', 'green', 'orange', 'red'])
    ax.set_ylim(0, 1)  # Setting ylim to match the range of metric values
    ax.set_ylabel('Value')
    ax.set_title(f'{channel}Metrics')
    plt.savefig(f'OUTPUT/DT/Metrics_{channel}.png')  # Save the figure
    plt.close()  
    
    main()