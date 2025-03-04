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
module_name = 'ANN Model'

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

# Import necessary libraries
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

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
   
   
    return train_X, train_Y, validate_X, validate_Y, test_X, test_Y,  test2_X, test2_Y, scaler

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    #Define the channels
    channels = ['P3', 'P4', 'Pz']
    
    print(f"{module_name} module begins.")

for channel in channels:
    # Load and preprocess the data
    train_X, train_Y, validate_X, validate_Y, test_X, test_Y,  test2_X, test2_Y, scaler = load_preprocess_split(channel)

    # Initialize the model
    ann = tf.keras.models.Sequential()

    # Model Architecture
    ann.add(tf.keras.layers.Dense(units=32, activation='relu'))
    ann.add(Dropout(0.2))  # Dropout layer
    ann.add(tf.keras.layers.Dense(units=32, activation='relu')) 
    ann.add(Dropout(0.2))  # Dropout layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Optimization and Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adam optimizer
    ann.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Model Training
    history = ann.fit(train_X, train_Y, batch_size=10, epochs=200, 
                      validation_data=(validate_X, validate_Y),
                      callbacks=[early_stopping]) 

    # Save the model in Keras format so we can run test
    ann.save(f'Model/Parameters/ANNParam_{channel}.keras')
    
#Test Using Pre-Trained Model
#------------------------------------------------------------------------------------------------
    # Load the model from the Keras format file
    model = load_model(f'Model/Parameters/ANNParam_{channel}.keras')


    # Evaluate the model on the first test data
    print('Evaluate on Test Data')
    results = model.evaluate(test_X, test_Y)
    print(f"test{channel} loss, test acc:", results)

    # Make predictions on the first test data
    probabilities = model.predict(test_X)


    # Evaluate the model on the Second test data
    print('Evaluate on 2nd Test Data')
    results = model.evaluate(test2_X, test2_Y)
    print(f"test{channel} loss, test acc:", results)

    # Make predictions on the Second test data
    probabilities = model.predict(test2_X)

    # Apply threshold to positive probabilities to create labels
    threshold = 0.5
    predictions = [1 if prob > threshold else 0 for prob in probabilities]

    # Compute confusion matrix
    cm = confusion_matrix(test2_Y, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for {channel}')
    plt.savefig(f'OUTPUT/ANN/confusion_matrix_{channel}.png')  # Save the figure
    plt.close()

    # Plotting the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss for {channel}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f'OUTPUT/ANN/epoch_vs_loss_{channel}.png')  # Save the figure
    plt.close()

        # Compute confusion matrix
    cm = confusion_matrix(test2_Y, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for {channel}')
    plt.savefig(f'OUTPUT/ANN/confusion_matrix_{channel}.png')  # Save the figure
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
    plt.savefig(f'OUTPUT/ANN/Metrics_{channel}.png')  # Save the figure
    plt.close()

    main()