import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.math import argmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape, Conv1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
import imageio.v3 as iio
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import re
from tensorflow.keras.models import Sequential
import optuna
import optuna.visualization as ov
import optuna.visualization.matplotlib
from optuna.pruners import MedianPruner
from keras.regularizers import l2


# Save directory
savedir = '/directory/where/you/want/results/to/save/to' + '/'

# Set the working directories
dir1 = '/directory/that/contains/csv/files'
dir2 = '/directory/that/contains/images'
os.chdir(dir1)

# Choose file import type
#pic_type = '.jpg'  # or '.tif' for TIFs
pic_type = '.tif'
div = 1 if pic_type == '.jpg' else 255

# Trait name
# List of possiblities to run M4, M5, or M6:
    # M4: traitname = 'FPC1_Combined'
    # M5: traitname = 'FPC1_RCC'
    # M6: traitname = 'FPC1_TNDGR'
traitname = 'FPC1_Combined'

# Import scores
# Always remove NAs based on "Score" column
scores = pd.read_csv("df4.csv")
scores_clean = scores.dropna(subset=["Score"]).copy()
HB = scores_clean[scores_clean['Test'] == 'HB'] # HB will later become the validation set
OGR = scores_clean[scores_clean['Test'] == 'OGR']

os.chdir(dir2)
DL_files = scores_clean['File_Name_JPGs']

pic_dim = 163
epoch_num = 50
trn_test_split = 0.5
sandwichsize = 3 * len(set(scores_clean['DAT']))
num_optuna_trials = 250


# List all files of the chosen type
pics = [file for file in DL_files if file.endswith(pic_type)]

# Function to read images
def read_image(file):
    img = iio.imread(file)
    return np.array(img)

# Create a dictionary to store images keyed by filenames
mypics_dict = {pic: read_image(pic) for pic in pics}

# Function to resize images
def resize_image(image_array):
    return tf.image.resize(image_array, [pic_dim, pic_dim]).numpy()

# Resize and reshape images
mypics_resized = {name: resize_image(img) for name, img in mypics_dict.items()}

# Remove alpha channel (4th band, sometimes an alpha channel is accidentally exported in Metashape)
for name, img in mypics_resized.items():
    if img.shape[2] == 4:  # Check if the image has a 4th channel
        mypics_resized[name] = img[:, :, :3]  # Remove the 4th channel

mypics_resized['20230724-3025_001-180.jpg'].shape # Check dimensions of first image in dictionary

# Display and check the first image (run all 4 lines simultaneously)
first_image_name = next(iter(mypics_resized))  # More efficient way to get the first key
plt.imshow(mypics_resized[first_image_name].astype(np.uint8)/div)  # Assuming images are in uint8 format for display
plt.title(first_image_name)
plt.show()

# Print the dimensions of the first image
print(f"Dimensions of '{first_image_name}':", mypics_resized[first_image_name].shape)

totalsize = len(OGR['Pltg_ID_Key_JPG'].unique()) # Total number of unique genotypes in E1
trainsize = int(totalsize * trn_test_split) # Number of 80% of genotypes

DL_all_ids = pd.DataFrame(scores_clean['Pltg_ID_Key_JPG'].unique(), columns=['Pltg_ID_Key_JPG']) # Data frame of unique genotype names
# from both experiments

# Create the time-series images / image sandwiches
keys = scores_clean['Pltg_ID_Key_JPG'].unique()
tsi = {}

# np.concatenate appends tensors across the third dimension, thereby flattening the color channels
# If testing the loop, check sorted(images_each_geno.keys()) to ensure the images are sorted temporally
# Check sandwich.shape to ensure 163x163x42
# Call tsi["4015_001-0.jpg"].shape to check dimensions of first tsi
for key in keys:
    # Filter images for the current genotype
    images_each_geno = {name: img for name, img in mypics_resized.items() if re.search(str(key), name)}
    
    # Since filenames sort correctly, directly use sorted keys to order images
    images_sorted = [images_each_geno[name] for name in sorted(images_each_geno.keys())]

    # Combine images for the current genotype
    sandwich = np.concatenate(images_sorted, axis=2)
    
    # Store the sandwich in the dictionary with the key as its name
    tsi[key] = sandwich

# Function to convert a list of arrays into a 4D array
def convert_to_4d_array(list_of_arrays):
    # Stack the list of 3D arrays (images) along a new axis to create a 4D array
    array_4d = np.stack(list_of_arrays, axis=0)
    return array_4d


#### Optuna ####
# We're using Optuna to build the model with only E1. Later, we'll perform regression
# using the E1-optimized model to regress on E2, which is completely unseen.
# Instead of choosing randomly from DL_all_ids, we need to choose from only E1 keys
OGR_ids = pd.DataFrame(OGR['Pltg_ID_Key_JPG'].unique(), columns=['Pltg_ID_Key_JPG'])

# For accurate model comparison, set seed so the visual, RCC, and TNDGR models have the same training set
np.random.seed(1)
DL_train = np.random.choice(OGR_ids['Pltg_ID_Key_JPG'], size=trainsize, replace=False)

# Specify the validation genotypes as those not in DL_train
DL_test = OGR_ids[~OGR_ids['Pltg_ID_Key_JPG'].isin(DL_train)]['Pltg_ID_Key_JPG'].values
    
# Create dictionaries to ensure images match their categories
# Use trainx_dict.keys() to ensure that TSIs were added in order of DL_train
trainx_dict = {pedigree: tsi[pedigree] for pedigree in DL_train if pedigree in tsi}
testx_dict = {pedigree: tsi[pedigree] for pedigree in DL_test if pedigree in tsi}
    
# Subset scores_clean
OGR_clean_subset = OGR.drop_duplicates(subset=['Pltg_ID_Key_JPG'])

# Initialize empty lists for trainy and testy labels
trainy = []
testy = []
    
# Create trainy and testy
for pltg_id in DL_train:
    # Find the row in OGR_clean_subset with this pltg_id and extract the label
    label = OGR_clean_subset[OGR_clean_subset['Pltg_ID_Key_JPG'] == pltg_id][traitname]
    if not label.empty:
        trainy.extend(label.values)  # Use extend() to flatten the array into the list
            
for pltg_id in DL_test:
    # Find the row in OGR_clean_subset with this pltg_id and extract the label
    label = OGR_clean_subset[OGR_clean_subset['Pltg_ID_Key_JPG'] == pltg_id][traitname]
    if not label.empty:
        testy.extend(label.values)
            
trainy = np.array(trainy)
testy = np.array(testy)

# Convert dictionaries to lists of arrays for CNN
trainx_list = list(trainx_dict.values())
testx_list = list(testx_dict.values())

# Convert image lists to 4D arrays
trainx_4d = convert_to_4d_array(trainx_list) / 255.0
testx_4d = convert_to_4d_array(testx_list) / 255.0

# Ensure min and max values are [0,1]
np.min(trainx_4d[0])   
np.max(trainx_4d[0])
np.min(testx_4d[0])   
np.max(testx_4d[0])

# Quality control - ensure labels match within the training and validation sets
index_check = 15
print(list(testx_dict.keys())[index_check]) 
check_geno = list(testx_dict.keys())[index_check]
print(testy[index_check])
print(OGR_clean_subset.loc[OGR_clean_subset['Pltg_ID_Key_JPG'] == check_geno, traitname])

print(list(trainx_dict.keys())[index_check]) 
check_geno = list(trainx_dict.keys())[index_check]
print(trainy[index_check])
print(OGR_clean_subset.loc[OGR_clean_subset['Pltg_ID_Key_JPG'] == check_geno, traitname])

# Searching for the optimal number of convolutional layers in addition to:
    # learning rate
    # dense neurons
    # dropout rate
    # activation function type
    # number of filters
    # regularization
    # number of convolutional layers
    # first kernel size (n, n)

def objective(trial):
    # Hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dense_neurons = trial.suggest_categorical('dense_neurons', [128, 256, 512, 1024])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'tanh', 'linear'])
    regularization = trial.suggest_float('regularization', 1e-4, 1e-2, log=True)
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 5)
    first_kernel = trial.suggest_categorical('first_kernel', [2, 3, 4, 5, 6, 7])
    
    # Set the initial number of filters
    first_filter = trial.suggest_categorical('num_filters_layer_0', [16, 32])
    num_filters = [first_filter]

    # Calculate subsequent layer filters by doubling
    for i in range(1, num_conv_layers):
        num_filters.append(num_filters[i-1] * 2)  # Double the number of filters from the previous layer

    # Report the configured number of filters for each layer
    print(f"Configured number of filters per layer: {num_filters}")

    kernel_size = (first_kernel, first_kernel)

    model = Sequential()
    model.add(Conv2D(num_filters[0], kernel_size=kernel_size, activation=activation_function, input_shape=(pic_dim, pic_dim, sandwichsize), kernel_regularizer=l2(regularization)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    for i in range(1, len(num_filters)):
        model.add(Conv2D(num_filters[i], kernel_size=(3, 3), activation=activation_function, kernel_regularizer=l2(regularization)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(dense_neurons, activation=activation_function))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(optimizer=RMSprop(learning_rate=lr), loss='mean_squared_error', metrics=['mae'])
    
    for epoch in range(50):
        history = model.fit(trainx_4d, trainy, batch_size=32, validation_split=0.2, verbose=1)
        intermediate_value = model.evaluate(testx_4d, testy, verbose=0)[0]
        trial.report(intermediate_value, epoch)
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    evaluation = model.evaluate(testx_4d, testy, verbose=1)
    tf.keras.backend.clear_session()
    
    return evaluation[0]


# Create study
study = optuna.create_study(
    direction='minimize', 
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=20,  # Allow initial 10 trials to run without pruning
        n_warmup_steps=10,    # Start pruning checks after 10 epochs
        interval_steps=5      # Check for pruning every 5 epochs
    )
)


# Execute the optimization for n number of trials
study.optimize(objective, n_trials=num_optuna_trials)
study.trials_dataframe().to_csv(savedir + 'optuna_trials.csv', index=False)
best_parameters = study.best_params

# Best parameters
print('Best parameters:', study.best_params)

# Best score achieved during the optimization
print('Best score:', study.best_value)

# Detailed report of the optimization
print('Optimization history:', study.trials_dataframe())


#### Optuna visualizations ####
width_inches = 7
height_inches = 4
dpi = 600

# Modify y axes
optuna.visualization.matplotlib.plot_optimization_history(study)
ax = plt.gca()
ax.set_ylim([0, 60])
plt.gcf().set_size_inches(width_inches, height_inches)
plt.tight_layout()
plt.savefig(savedir + 'optimization_history.jpeg', dpi=dpi)
plt.show()


optuna.visualization.matplotlib.plot_parallel_coordinate(study)

optuna.visualization.matplotlib.plot_slice(study, params=['lr', 'dense_neurons', 'dropout_rate', 'activation_function', 'num_filters_layer_0', 'regularization', 'num_conv_layers', 'first_kernel'])
ax = plt.gca()
ax.set_ylim([0, 100])
plt.gcf().set_size_inches(width_inches+8, height_inches+3)
plt.savefig(savedir + 'plot_slice.jpeg', dpi=dpi)
plt.show()

optuna.visualization.matplotlib.plot_param_importances(study)
plt.gcf().set_size_inches(width_inches, height_inches+3)
plt.tight_layout()
plt.savefig(savedir + 'param_importance.jpeg', dpi=dpi)
plt.show()


#### Create new model using best parameters as determined by Optuna ####
#best_parameters = {'lr': 0.0001835100161864574, 'dense_neurons': 256, 'dropout_rate': 0.0011790374105916214, 'activation_function': 'tanh', 'num_filters': 16, 'regularization': 0.0003503154368917454, 'num_conv_layers': 5, 'first_kernel': 2}
best_lr = best_parameters['lr']
best_dense_neurons = best_parameters['dense_neurons']
best_dropout_rate = best_parameters['dropout_rate']
best_activation_function = best_parameters['activation_function']
best_regularization = best_parameters['regularization']
best_num_conv_layers = best_parameters['num_conv_layers']
best_first_kernel = best_parameters['first_kernel']

first_filter = best_parameters['num_filters_layer_0']
num_filters = [first_filter]
for i in range(1, best_num_conv_layers):
    num_filters.append(num_filters[i-1] * 2)

# Convert the first_kernel from Optuna's suggestion (if necessary)
if isinstance(best_first_kernel, int):
    kernel_size = (best_first_kernel, best_first_kernel)
else:
    kernel_size = best_first_kernel


# Define the CNN best model architecture
eval_list = {}
metrics_list = {}
labels_list = {}

num_reps = 25

for i in range(num_reps):
    
    np.random.seed(i)
    
    # Set E2 as the validation set
    HB_keys = HB['Pltg_ID_Key_JPG'].unique()
    
    DL_test = DL_all_ids[DL_all_ids['Pltg_ID_Key_JPG'].isin(HB_keys)]['Pltg_ID_Key_JPG'].values
    
    DL_train = DL_all_ids[~DL_all_ids['Pltg_ID_Key_JPG'].isin(HB_keys)]['Pltg_ID_Key_JPG'].values
        
    # Create dictionaries to ensure images match their categories
    # Use trainx_dict.keys() to ensure that TSIs were added in order of DL_train
    trainx_dict = {pedigree: tsi[pedigree] for pedigree in DL_train if pedigree in tsi}
    testx_dict = {pedigree: tsi[pedigree] for pedigree in DL_test if pedigree in tsi}
        
    # Subset scores_clean
    scores_clean_subset = scores_clean.drop_duplicates(subset=['Pltg_ID_Key_JPG'])

    # Initialize empty lists for trainy and testy labels
    trainy = []
    testy = []
        
    # Create trainy and testy
    for pltg_id in DL_train:
        # Find the row in scores_clean_subset with this pltg_id and extract the label
        label = scores_clean_subset[scores_clean_subset['Pltg_ID_Key_JPG'] == pltg_id][traitname]
        if not label.empty:
            trainy.extend(label.values)  # Use extend() to flatten the array into the list
                
    for pltg_id in DL_test:
        # Find the row in scores_clean_subset with this pltg_id and extract the label
        label = scores_clean_subset[scores_clean_subset['Pltg_ID_Key_JPG'] == pltg_id][traitname]
        if not label.empty:
            testy.extend(label.values)  # Use extend() to flatten the array into the list
                
    trainy = np.array(trainy)
    testy = np.array(testy)

    # Convert dictionaries to lists of arrays for CNN
    trainx_list = list(trainx_dict.values())
    testx_list = list(testx_dict.values())

    # Convert image lists to 4D arrays
    trainx_4d = convert_to_4d_array(trainx_list) / 255.0
    testx_4d = convert_to_4d_array(testx_list) / 255.0

    # Ensure min and max values are [0,1]
    np.min(trainx_4d[0])   
    np.max(trainx_4d[0])
    np.min(testx_4d[0])   
    np.max(testx_4d[0])

    # Quality control - ensure labels match within the training and validation sets
    index_check = 15
    print(list(testx_dict.keys())[index_check]) 
    check_geno = list(testx_dict.keys())[index_check]
    print(testy[index_check])
    print(scores_clean_subset.loc[scores_clean_subset['Pltg_ID_Key_JPG'] == check_geno, traitname])

    print(list(trainx_dict.keys())[index_check]) 
    check_geno = list(trainx_dict.keys())[index_check]
    print(trainy[index_check])
    print(scores_clean_subset.loc[scores_clean_subset['Pltg_ID_Key_JPG'] == check_geno, traitname])

    bestmodel = Sequential()

    # First Conv2D layer
    bestmodel.add(Conv2D(num_filters[0], kernel_size=kernel_size, activation=best_activation_function, 
                         input_shape=(pic_dim, pic_dim, sandwichsize),
                         kernel_regularizer=tf.keras.regularizers.l2(best_regularization)))
    bestmodel.add(MaxPooling2D(pool_size=(2, 2)))

    # Additional Conv2D layers dynamically added based on the number of convolutional layers determined in the study
    for j in range(1, best_num_conv_layers):
        bestmodel.add(Conv2D(num_filters[j], kernel_size=(3, 3), activation=best_activation_function,
                             kernel_regularizer=tf.keras.regularizers.l2(best_regularization)))
        bestmodel.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output and add Dense and Dropout layers
    bestmodel.add(Flatten())
    bestmodel.add(Dense(best_dense_neurons, activation=best_activation_function))
    bestmodel.add(Dropout(best_dropout_rate))
    bestmodel.add(Dense(1))

    # Compile the bestmodel
    bestmodel.compile(optimizer=RMSprop(learning_rate=best_lr), loss='mean_squared_error', metrics=['mae'])

    # Fit the model
    history = bestmodel.fit(
        trainx_4d,
        trainy,
        epochs=epoch_num,
        batch_size=32,
        validation_split=0.2
        )


    evaluation = bestmodel.evaluate(testx_4d, testy)
    evaluation_df = pd.DataFrame([evaluation], columns=['loss', 'mae'])
    evaluation_df['seed'] = i
    
    predictions = bestmodel.predict(testx_4d)
    predictions = predictions.flatten()
    
    # Convert the history metrics to a DataFrame
    history_df = pd.DataFrame(history.history)
    # Add an 'epoch' column (Python is 0-indexed, so add 1 to start epochs from 1)
    history_df['epoch'] = history_df.index + 1
    # Add a column for the seed
    history_df['seed'] = i
    
    # Labels df
    labels_vec = list(testx_dict.keys())
    labels_df = pd.DataFrame({'Pltg_ID_Key_JPG': labels_vec, 'Actual': testy, 'Predicted': predictions})
    labels_df['seed'] = i
    
    eval_list[i] = evaluation_df
    metrics_list[i] = history_df
    labels_list[i] = labels_df
    
    if i == (num_reps-1):
        # Concatenate and Save eval_list DataFrames
        eval_concat = pd.concat(eval_list, ignore_index=True)
        eval_concat.to_csv(savedir + 'eval_list.csv', index=False)
        
        # Concatenate and Save metrics_list DataFrames
        metrics_concat = pd.concat(metrics_list, ignore_index=True)
        metrics_concat.to_csv(savedir + 'metrics_list.csv', index=False)
    
        # Save actual and predicted values
        labels_concat = pd.concat(labels_list, ignore_index=True)
        labels_concat.to_csv(savedir + 'labels_list.csv', index=False)