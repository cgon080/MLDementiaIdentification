import os
import pandas as pd
import numpy as np
import functools 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import torch.nn.init as init
from itertools import product
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample
from tqdm import tqdm
import openpyxl

gpu = 2
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print('Importing data...')

#Visits
#base_path = r'C:\Users\cgon080\OneDrive - The University of Auckland\Documents\'
#base_path = r'D:\cgon080\ML\DifferentTest\Time0m_scaled\'
#visits_df = pd.read_csv(r'D:\cgon080\ML\DifferentTest\Time96m_retrain\Codes_visit1.csv')
visits_df = pd.read_csv(r'D:\cgon080\ML\Data\Features2\Time6m_retrain\Codes_visit1.csv')

timevisit_dict = {}
codes_dict = {}

for id, group_data in visits_df.groupby('id_patient'):
    # Extract timestamps and values for the current id
    timestamps = group_data.iloc[:,1].tolist()
    values = group_data.iloc[:,2].tolist()

    # Append lists to the dictionaries
    if id not in timevisit_dict:
        timevisit_dict[id] = []
        codes_dict[id] = []

    timevisit_dict[id].append(timestamps)
    codes_dict[id].append(values)

# Load each longitudinal variable CSV file

#data_folder = r"D:\cgon080\ML\DifferentTest\Time96m_retrain\DataPython"
data_folder = r"D:\cgon080\ML\Data\Features2\Time6m_retrain\DataPython"   
csv_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]
longitudinal_dfs = [pd.read_csv(os.path.join(data_folder, file)) for file in csv_files]

timestamps_dict = {}
values_dict = {}
feature_dict = {}

# Iterate through the list of DataFrames
for df in longitudinal_dfs:
    #Name of the variable
    feature_name = df.columns[2]
    # Group DataFrame by 'id'
    grouped_data = df.groupby('id_patient')

    # Iterate through each group (id)
    for group_id, group_data in grouped_data:
        # Extract timestamps and values for the current id
        timestamps = group_data.iloc[:,1].tolist()
        values = group_data.iloc[:,2].tolist()

        # Append lists to the dictionaries
        if group_id not in timestamps_dict:
            timestamps_dict[group_id] = []
            values_dict[group_id] = []
            feature_dict[group_id] = []

        timestamps_dict[group_id].append(timestamps)
        values_dict[group_id].append(values)
        feature_dict[group_id].append(feature_name)

max_length_visit1 = max(len(seq) for patient_sequences in codes_dict.values() for seq in patient_sequences)
max_length_1 = max(len(seq) for patient_sequences in values_dict.values() for seq in patient_sequences)
max_length_visit = max(max_length_visit1, max_length_1)

final_timevisit = []

# Step 3: Iterate over each patient and pad the sequences
for patient_id, patient_sequences in timevisit_dict.items():
    padded_sequences = []
    for variable_sequences in patient_sequences:
        # Pad the sequences with zeros
        padded_sequence = variable_sequences + [0] * (max_length_visit - len(variable_sequences))
        padded_sequences.append(padded_sequence)

    final_timevisit.append(padded_sequences)

final_codes = []

# Step 3: Iterate over each patient and pad the sequences
for patient_id, patient_sequences in codes_dict.items():
    padded_sequences = []
    for variable_sequences in patient_sequences:
        # Pad the sequences with zeros
        padded_sequence = variable_sequences + [0] * (max_length_visit - len(variable_sequences))
        padded_sequences.append(padded_sequence)

    final_codes.append(padded_sequences)
       
final_timevisit_array = np.array(final_timevisit)
final_timevisit_array = np.transpose(final_timevisit_array, (0, 2, 1))
final_timevisit_array.shape
final_timevisit_array = np.round(final_timevisit_array, decimals=0)
timevisit_tensors = torch.from_numpy(final_timevisit_array)
timevisit_tensors.shape
timevisit_tensors = timevisit_tensors.long()
#timevisit_tensors = timevisit_tensors.view(173, 237) # Delete the extra dimension 

final_codes_array = np.array(final_codes)
final_codes_array = np.transpose(final_codes_array, (0, 2, 1))
final_codes_array.shape
codes_tensors = torch.from_numpy(final_codes_array)
codes_tensors.shape
n_size, _, _ = codes_tensors.shape
codes_tensors = codes_tensors.view(n_size, max_length_visit)
codes_tensors = codes_tensors.long()

final_timestamps = []

# Step 3: Iterate over each patient and pad the sequences
for patient_id, patient_sequences in timestamps_dict.items():
    padded_sequences = []
    for variable_sequences in patient_sequences:
        # Pad the sequences with zeros
        padded_sequence = variable_sequences + [0] * (max_length_visit - len(variable_sequences))
        padded_sequences.append(padded_sequence)

    final_timestamps.append(padded_sequences)

final_values = []

# Step 3: Iterate over each patient and pad the sequences
for patient_id, patient_sequences in values_dict.items():
    padded_sequences = []
    for variable_sequences in patient_sequences:
        # Pad the sequences with zeros
        padded_sequence = variable_sequences + [0] * (max_length_visit - len(variable_sequences))
        padded_sequences.append(padded_sequence)

    final_values.append(padded_sequences)

final_timestamp_array = np.array(final_timestamps)
final_timestamp_array.shape
final_timestamp_array = np.transpose(final_timestamp_array, (0, 2, 1))

timestamps_tensors = torch.from_numpy(final_timestamp_array).float()
timestamps_tensors.shape

final_value_array = np.array(final_values)
final_value_array = np.transpose(final_value_array, (0, 2, 1))
values_tensors = torch.from_numpy(final_value_array).float()
values_tensors.shape


#Static variables
#static_df = pd.read_csv(r'D:\cgon080\ML\DifferentTest\Time96m_retrain\Static_features.csv')
static_df = pd.read_csv(r'D:\cgon080\ML\Data\Features2\Time6m_retrain\Static_features.csv')
static_df['Diagnosis'] = static_df['Diagnosis'].map({'No dementia': 0, 'Dementia': 1})

#static_df.shape
#static_df.info()
#static_df.columns

_, ncol = static_df.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
static_df_scaled = scaler.fit_transform(static_df.iloc[:,range(11,ncol)].values)

static_vars_tensor = torch.FloatTensor(static_df_scaled)
static_vars_tensor.size()

#Labels 

labels = static_df['Diagnosis'].values
labels = torch.from_numpy(labels).float()
labels.dtype

time_visit_train, time_visit_test, codes_train, codes_test, sequences_train, sequences_test, timestamps_train, timestamps_test, static_train, static_test, labels_train, labels_test = train_test_split(
    timevisit_tensors, codes_tensors, values_tensors, timestamps_tensors, static_vars_tensor, labels, test_size=0.2, random_state=43, stratify=labels
)

time_visit_train, time_visit_val, codes_train, codes_val, sequences_train, sequences_val, timestamps_train, timestamps_val, static_train, static_val, labels_train, labels_val = train_test_split(
    time_visit_train, codes_train, sequences_train, timestamps_train, static_train, labels_train, test_size=0.25, random_state=43, stratify=labels_train
)


#RNN
print('Permutation Importance for the RNN+MLP model...')

class GRUModel_1(nn.Module):
    def __init__(self, embedding_dim, input_size, hidden_size, static_size, output_size):
        super(GRUModel_1, self).__init__()
        self.embedding = nn.Embedding(5460, embedding_dim) # Dim all the vocabulary
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size, 500),
            nn.ReLU(),
            nn.Linear(500,output_size)
            )

    def forward(self, timevisits, codes, sequences, timestamps, static_vars):
        # Apply the embedding layer
        embedded_codes = self.embedding(codes)
        #codes_with_time  = torch.cat([embedded_codes, timevisits.unsqueeze(2).repeat(1, 1, embedded_codes.size(2))], dim=-1)

        # Concatenate timestamps with the sequences at each time step
        sequences_with_time = torch.cat([sequences, timestamps, embedded_codes, timevisits], dim=-1)

        # Concatenate static variables to each time step
        #sequences_with_static = torch.cat([sequences_with_time, static_vars.unsqueeze(1).repeat(1, sequences.size(1), 1)], dim=-1)

        h0 = torch.zeros(1,sequences_with_time.size(0), self.hidden_size).to(sequences_with_time.device)

        # Feed the concatenated input to the GRU model
        gru_out, _ = self.gru(sequences_with_time.float(), h0)

        # Extract output at the last time step
        last_output = gru_out[:, -1, :]

        last_output_stat = torch.cat([last_output, static_vars], dim=-1)

        # Final classification layer
        output = self.fc(last_output_stat)
        return torch.sigmoid(output)


batch_size = 64
embedding_dim = 100
input_size = sequences_train.shape[-1] + timestamps_train.shape[-1] + embedding_dim + 1
static_size = static_train.shape[-1] 

# Step 3: Define Loss Function and Optimizer

model = GRUModel_1(embedding_dim=embedding_dim, input_size=input_size, hidden_size=64, static_size=static_size, output_size=1).to(device)  
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.load_state_dict(torch.load(r'D:\cgon080\ML\Data\Features2\RNN_6.pth'))
model.to(device)

test_dataset = TensorDataset(time_visit_test, codes_test, sequences_test, timestamps_test, static_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = len(labels_test)
model.eval()
with torch.no_grad():
    all_predictions = []
    all_labels = []
    all_probs = []
    for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels_batch in test_loader:
        time_visit_batch = time_visit_batch.to(device)
        codes_batch = codes_batch.to(device)
        sequences_batch = sequences_batch.to(device)
        timestamps_batch = timestamps_batch.to(device)
        static_batch = static_batch.to(device)
        labels_batch = labels_batch.to(device)
        y_pred = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
        predictions = y_pred.squeeze().round()
        correct += torch.sum((predictions == labels_batch).float())
        #print(f'Correct: {correct:.4f}')
        all_probs.extend(y_pred.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

def pred_test(test_loader, model):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels_batch in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels_batch = labels_batch.to(device)
            y_pred = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
            predictions = y_pred.squeeze().round()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    return all_predictions, all_labels

print('Permutation importance started...')

prediction, labels_pred_test = pred_test(test_loader=test_loader, model=model) 

baseline_accuracy = metrics.accuracy_score(labels_pred_test, prediction)
print(f'Baseline accuracy: {baseline_accuracy:.4f}')
baseline_sensitivity = metrics.recall_score(labels_pred_test, prediction)
print(f'Baseline sens: {baseline_sensitivity:.4f}')
baseline_specificity = metrics.recall_score(labels_pred_test, prediction, pos_label=0)
print(f'Baseline spec: {baseline_specificity:.4f}')

print('Sequence features permutation importance started...')

def permute_feature_and_evaluate(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences = sequences.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_sequences = sequences.clone()
            permuted_sequences[:, :, feature_index] = permuted_sequences[torch.randperm(permuted_sequences.size(0)), :, feature_index]
            
            #outputs = model(timevisits, codes, permuted_sequences, timestamps, static_vars)
            outputs = model(time_visit_batch.float(), codes_batch, permuted_sequences.float(), timestamps_batch.float(), static_batch.float()).squeeze()
            preds = outputs.round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Permuration importance 

import numpy as np

# Initialize matrices to store results of each iteration
num_features = values_tensors.size(2)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate(model, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance)
#$print("Mean Specificity Importance per Feature:", mean_specificity_importance)

df = pd.DataFrame(mean_accuracy_importance)
df1 = pd.DataFrame(mean_sensitivity_importance)
df2 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_RNN.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet1', index=False)
#    df1.to_excel(writer, sheet_name='Sheet2', index=False)
#    df2.to_excel(writer, sheet_name='Sheet3', index=False)

print('Timestamps sequence features permutation importance started...')

def permute_feature_and_evaluate_time_seq(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_timestamps = timestamps_batch.clone()
            permuted_timestamps[:, :, feature_index] = permuted_timestamps[torch.randperm(permuted_timestamps.size(0)), :, feature_index]
            
            outputs = model(time_visit_batch.float(), codes_batch, sequences_batch.float(), permuted_timestamps.float(), static_batch.float()).squeeze()
            #outputs = model(timevisits, codes, sequences, permuted_timestamps, static_vars)
            preds = outputs.round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Initialize matrices to store results of each iteration
num_features = timestamps_test.size(2)
num_iterations = 100

#For month 0, is t

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_time_seq(model, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"TIMESTAMP: Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_t3)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_t3)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_t3)

df3 = pd.DataFrame(mean_accuracy_importance)
df4 = pd.DataFrame(mean_sensitivity_importance)
df5 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_RNN.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet4', index=False)
#    df1.to_excel(writer, sheet_name='Sheet5', index=False)
#    df2.to_excel(writer, sheet_name='Sheet6', index=False)

print('Static features permutation importance started...')

def permute_feature_and_evaluate_stat(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_static_vars = static_batch.clone()
            permuted_static_vars[:, feature_index] = permuted_static_vars[torch.randperm(permuted_static_vars.size(0)), feature_index]
            
            outputs = model(time_visit_batch.float(), codes_batch, sequences_batch.float(), timestamps_batch.float(), permuted_static_vars.float()).squeeze()
            #outputs = model(timevisits, codes, sequences, timestamps, permuted_static_vars)
            preds = outputs.round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Initialize matrices to store results of each iteration
num_features = static_test.size(1)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_stat(model, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"STATIC: Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_s4)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_s4)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_s4)

df6 = pd.DataFrame(mean_accuracy_importance)
df7 = pd.DataFrame(mean_sensitivity_importance)
df8 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_RNN.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet7', index=False)
#    df1.to_excel(writer, sheet_name='Sheet8', index=False)
#    df2.to_excel(writer, sheet_name='Sheet9', index=False)

print('ICD-10 features permutation importance started...')

def permute_feature_and_evaluate_codes(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_codes = codes_batch.clone()
            permuted_codes = permuted_codes[torch.randperm(permuted_codes.size(0)), :]
            
            outputs = model(time_visit_batch.float(), permuted_codes, sequences_batch.float(), timestamps_batch.float(), static_batch.float()).squeeze()
            #outputs = model(timevisits, permuted_codes, sequences, timestamps, static_vars)
            preds = outputs.round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

def permute_feature_and_evaluate_codes_time(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for timevisits, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            timevisits = timevisits.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_timevisits = timevisits.clone()
            permuted_timevisits = permuted_timevisits[torch.randperm(permuted_timevisits.size(0)), :]
            
            #outputs = model(permuted_timevisits, codes, sequences, timestamps, static_vars)
            outputs = model(permuted_timevisits.float(), codes_batch, sequences_batch.float(), timestamps_batch.float(), static_batch.float()).squeeze()
            preds = outputs.round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

num_features = 2
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 50 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes(model, test_loader)

    accuracy_importance = baseline_accuracy - permuted_accuracy
    sensitivity_importance = baseline_sensitivity - permuted_sensitivity
    specificity_importance = baseline_specificity - permuted_specificity

    accuracy_importances.append(accuracy_importance)
    sensitivity_importances.append(sensitivity_importance)
    specificity_importances.append(specificity_importance)

    #print(f"CODES: Iteration {iteration + 1}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes_time(model, test_loader)

    accuracy_importance = baseline_accuracy - permuted_accuracy
    sensitivity_importance = baseline_sensitivity - permuted_sensitivity
    specificity_importance = baseline_specificity - permuted_specificity

    accuracy_importances.append(accuracy_importance)
    sensitivity_importances.append(sensitivity_importance)
    specificity_importances.append(specificity_importance)

    #print(f"TIMESTAMP: Iteration {iteration + 1}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_c4)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_c4)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_c4)

df9 = pd.DataFrame(mean_accuracy_importance)
df10 = pd.DataFrame(mean_sensitivity_importance)
df11 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_RNN.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet10', index=False)
#    df1.to_excel(writer, sheet_name='Sheet11', index=False)
#    df2.to_excel(writer, sheet_name='Sheet12', index=False)        

with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_RNN_6.xlsx') as writer:
    # Save the first set of sheets
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    df1.to_excel(writer, sheet_name='Sheet2', index=False)
    df2.to_excel(writer, sheet_name='Sheet3', index=False)
    
    # Save the second set of sheets
    df3.to_excel(writer, sheet_name='Sheet4', index=False)
    df4.to_excel(writer, sheet_name='Sheet5', index=False)
    df5.to_excel(writer, sheet_name='Sheet6', index=False)
    
    # Save the third set of sheets
    df6.to_excel(writer, sheet_name='Sheet7', index=False)
    df7.to_excel(writer, sheet_name='Sheet8', index=False)
    df8.to_excel(writer, sheet_name='Sheet9', index=False)
    
    # Save the fourth set of sheets
    df9.to_excel(writer, sheet_name='Sheet10', index=False)
    df10.to_excel(writer, sheet_name='Sheet11', index=False)
    df11.to_excel(writer, sheet_name='Sheet12', index=False)    


print('Permutation Importance for the Att2 model...')   


import torch
import torch.nn as nn
import math

class FeatureAttention(nn.Module):
    def __init__(self, input_size):
        super(FeatureAttention, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.input_size = input_size

    def forward(self, x):

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(-2, -1)) / math.sqrt(self.input_size)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, values)

        return attention_weights, attended_values

class SelfAttention(nn.Module):
    def __init__(self, input_size, device):
        super(SelfAttention, self).__init__()
        self.device = device
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.input_size = input_size

    def forward(self, x):
        batch_size, seq_length, d_model = x.size()

        # Add positional embeddings using the timestamps
        position_embeddings = self._get_positional_embeddings(batch_size = batch_size, seq_length = seq_length, d_model=d_model)
        x = x + position_embeddings
        x = x.transpose(2,1)

        # Apply linear projections
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Calculate attention scores and apply attention
        scores = torch.bmm(queries, keys.transpose(-2, -1)) / math.sqrt(self.input_size)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, values)

        return attention_weights, attended_values
    
    def _get_positional_embeddings(self, batch_size, seq_length, d_model):

        # Generate position indices and angle rates
        pos = torch.arange(seq_length, device = self.device).unsqueeze(1)  # Shape: [seq_length, 1]
        i = torch.arange(d_model, device = self.device).unsqueeze(0)       # Shape: [1, d_model]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(d_model))
    
        # Compute the positional encodings for one sequence
        pos_encoding = pos * angle_rates  # Shape: [seq_length, d_model]
    
        # Apply sin to even indices (2i) and cos to odd indices (2i+1)
        pos_encoding[:, 0::2] = torch.sin(pos_encoding[:, 0::2])  # Apply sin to even indices
        pos_encoding[:, 1::2] = torch.cos(pos_encoding[:, 1::2])  # Apply cos to odd indices
    
        # Add the batch dimension by repeating for the batch size
        pos_encoding_batch = pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: [batch_size, seq_length, d_model]
    
        return pos_encoding_batch
    
class FeatureAttentionStats(nn.Module):
    def __init__(self, input_size):
        super(FeatureAttentionStats, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.input_size = input_size

    def forward(self, x):

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.mm(queries, keys.transpose(-2, -1)) / math.sqrt(self.input_size)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attended_values = torch.mm(attention_weights, values)

        return attention_weights, attended_values    

class SelfAttentionModel(nn.Module):
    def __init__(self, input_size_code, hidden_size, output_size, static_size, input_size_feature, input_size_seq):
        super(SelfAttentionModel, self).__init__()
        self.embedding = nn.Embedding(input_size_code, hidden_size)                        #input(64, 463) -> output(64, 463, 100)
        self.feature_attention = FeatureAttention(hidden_size + 2*input_size_feature + 1)  #input(64, 463, 100+63+63+1) -> output(64, 463, 100+63+63+1)
        self.temporal_attention = SelfAttention(input_size_seq, device=device)             #input(64, 463, 100+63+63) -> output(64, 100+63+63, 463)
        self.static_attention = FeatureAttentionStats(static_size)                         #input(64, 197) -> output(64, 197)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size + 2*input_size_feature + 1, 250), # 63 is the number of sequences
            nn.ReLU(),
            nn.Linear(250, output_size)
        )

    def forward(self, timestamps_codes, codes, sequences, timestamps, static_vars):
        # Embed ICD-10 codes
        embedded_codes = self.embedding(codes)

        # Concatenate embedded codes with the other sequences
        sequences_with_codes = torch.cat([sequences, timestamps, embedded_codes, timestamps_codes], dim=-1)
        
        feature_attention_weights, attended_values = self.feature_attention(sequences_with_codes)

        # Apply self-attention (temporal) with positional embeddings
        #timestamps = torch.cat([timestamps, timestamps_codes], dim=-1)

        attention_weights, attended_values_1 = self.temporal_attention(attended_values)

        # Aggregate attended values (e.g., mean)
        aggregated_values = attended_values_1.mean(dim=2)

        # Apply static 'attention' (more like a data transformation)
        static_attention_weights, static_attended_values = self.static_attention(static_vars)
        aggregated_values_static = torch.cat([aggregated_values, static_attended_values], dim=-1)

        # Pass through the final classification layer
        output = self.fc(aggregated_values_static)
        output = torch.sigmoid(output)
        return output, feature_attention_weights, attention_weights, static_attention_weights

batch_size = 64

input_size_code = 5460
hidden_size = 100
input_size_feature = sequences_train.shape[-1]
input_size_seq = sequences_train.shape[-2] 
static_size = static_train.shape[-1]
output_size = 1

# Step 3: Define Loss Function and Optimizer
model3 = SelfAttentionModel(input_size_code=input_size_code, hidden_size=hidden_size, output_size=output_size, static_size=static_size, input_size_feature = input_size_feature, input_size_seq = input_size_seq).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model3.parameters(), lr=0.0001)

model3.load_state_dict(torch.load(r'D:\cgon080\ML\Data\Features2\SelfAt_Normal_PE_6m.pth'))
model3.to(device)

correct = 0
total = len(labels_test)
model3.eval()
with torch.no_grad():
    all_predictions = []
    all_labels = []
    all_probs = []
    for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels_batch in test_loader:
        time_visit_batch = time_visit_batch.to(device)
        codes_batch = codes_batch.to(device)
        sequences_batch = sequences_batch.to(device)
        timestamps_batch = timestamps_batch.to(device)
        static_batch = static_batch.to(device)
        labels_batch = labels_batch.to(device)
        y_pred, feature_attention_weights, attention_weights, static_attended_values = model3(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
        predictions = y_pred.squeeze().round()
        correct += torch.sum((predictions == labels_batch).float())
        #print(f'Correct: {correct:.4f}')
        all_probs.extend(y_pred.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

print('Permutation importance started...')

def pred_test_at(test_loader, model):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels_batch in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels_batch = labels_batch.to(device)
            y_pred, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
            predictions = y_pred.squeeze().round()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    return all_predictions, all_labels


prediction, labels_pred_test = pred_test_at(test_loader=test_loader, model=model3) 

baseline_accuracy = metrics.accuracy_score(labels_pred_test, prediction)
print(f'Baseline accuracy: {baseline_accuracy:.4f}')
baseline_sensitivity = metrics.recall_score(labels_pred_test, prediction)
print(f'Baseline sens: {baseline_sensitivity:.4f}')
baseline_specificity = metrics.recall_score(labels_pred_test, prediction, pos_label=0)
print(f'Baseline spec: {baseline_specificity:.4f}')

print('Sequence features permutation importance started...')

# Permuration importance 

import numpy as np

# Initialize matrices to store results of each iteration
num_features = values_tensors.size(2)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

def permute_feature_and_evaluate_at(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences = sequences.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_sequences = sequences.clone()
            permuted_sequences[:, :, feature_index] = permuted_sequences[torch.randperm(permuted_sequences.size(0)), :, feature_index]
            
            #outputs = model(timevisits, codes, permuted_sequences, timestamps, static_vars)
            outputs, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch.float(), codes_batch, permuted_sequences.float(), timestamps_batch.float(), static_batch.float())
            preds = outputs.squeeze().round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_at(model3, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance)
#$print("Mean Specificity Importance per Feature:", mean_specificity_importance)

df = pd.DataFrame(mean_accuracy_importance)
df1 = pd.DataFrame(mean_sensitivity_importance)
df2 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Att2.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet1', index=False)
#    df1.to_excel(writer, sheet_name='Sheet2', index=False)
#    df2.to_excel(writer, sheet_name='Sheet3', index=False)

print('Timestamps sequence features permutation importance started...')

# Initialize matrices to store results of each iteration
num_features = timestamps_test.size(2)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

def permute_feature_and_evaluate_time_seq_at(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_timestamps = timestamps_batch.clone()
            permuted_timestamps[:, :, feature_index] = permuted_timestamps[torch.randperm(permuted_timestamps.size(0)), :, feature_index]
            
            outputs, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch.float(), codes_batch, sequences_batch.float(), permuted_timestamps.float(), static_batch.float())
            #outputs = model(timevisits, codes, sequences, permuted_timestamps, static_vars)
            preds = outputs.squeeze().round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_time_seq_at(model3, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"TIMESTAMP: Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_t3)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_t3)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_t3)

df3 = pd.DataFrame(mean_accuracy_importance)
df4 = pd.DataFrame(mean_sensitivity_importance)
df5 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Att2.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet4', index=False)
#    df1.to_excel(writer, sheet_name='Sheet5', index=False)
#    df2.to_excel(writer, sheet_name='Sheet6', index=False)

print('Static features permutation importance started...')

# Initialize matrices to store results of each iteration
num_features = static_test.size(1)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

def permute_feature_and_evaluate_stat_at(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_static_vars = static_batch.clone()
            permuted_static_vars[:, feature_index] = permuted_static_vars[torch.randperm(permuted_static_vars.size(0)), feature_index]
            
            outputs, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch.float(), codes_batch, sequences_batch.float(), timestamps_batch.float(), permuted_static_vars.float())
            #outputs = model(timevisits, codes, sequences, timestamps, permuted_static_vars)
            preds = outputs.squeeze().round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_stat_at(model3, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"STATIC: Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_s4)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_s4)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_s4)

df6 = pd.DataFrame(mean_accuracy_importance)
df7 = pd.DataFrame(mean_sensitivity_importance)
df8 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Att2.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet7', index=False)
#    df1.to_excel(writer, sheet_name='Sheet8', index=False)
#    df2.to_excel(writer, sheet_name='Sheet9', index=False)

print('ICD-10 features permutation importance started...')

num_features = 2
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))


def permute_feature_and_evaluate_codes_at(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_codes = codes_batch.clone()
            permuted_codes = permuted_codes[torch.randperm(permuted_codes.size(0)), :]
            
            outputs, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch.float(), permuted_codes, sequences_batch.float(), timestamps_batch.float(), static_batch.float())
            #outputs = model(timevisits, permuted_codes, sequences, timestamps, static_vars)
            preds = outputs.squeeze().round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

def permute_feature_and_evaluate_codes_time_at(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for timevisits, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            timevisits = timevisits.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_timevisits = timevisits.clone()
            permuted_timevisits = permuted_timevisits[torch.randperm(permuted_timevisits.size(0)), :]
            
            #outputs = model(permuted_timevisits, codes, sequences, timestamps, static_vars)
            outputs, feature_attention_weights, attention_weights, static_attended_values = model(permuted_timevisits.float(), codes_batch, sequences_batch.float(), timestamps_batch.float(), static_batch.float())
            preds = outputs.squeeze().round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Perform 50 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes_at(model3, test_loader)

    accuracy_importance = baseline_accuracy - permuted_accuracy
    sensitivity_importance = baseline_sensitivity - permuted_sensitivity
    specificity_importance = baseline_specificity - permuted_specificity

    accuracy_importances.append(accuracy_importance)
    sensitivity_importances.append(sensitivity_importance)
    specificity_importances.append(specificity_importance)

    #print(f"CODES: Iteration {iteration + 1}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes_time_at(model3, test_loader)

    accuracy_importance = baseline_accuracy - permuted_accuracy
    sensitivity_importance = baseline_sensitivity - permuted_sensitivity
    specificity_importance = baseline_specificity - permuted_specificity

    accuracy_importances.append(accuracy_importance)
    sensitivity_importances.append(sensitivity_importance)
    specificity_importances.append(specificity_importance)

    #print(f"TIMESTAMP: Iteration {iteration + 1}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_c4)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_c4)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_c4)

df9 = pd.DataFrame(mean_accuracy_importance)
df10 = pd.DataFrame(mean_sensitivity_importance)
df11 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Att2.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet10', index=False)
#    df1.to_excel(writer, sheet_name='Sheet11', index=False)
#    df2.to_excel(writer, sheet_name='Sheet12', index=False)       

with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Att2_6.xlsx') as writer:
    # Save the first set of sheets
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    df1.to_excel(writer, sheet_name='Sheet2', index=False)
    df2.to_excel(writer, sheet_name='Sheet3', index=False)
    
    # Save the second set of sheets
    df3.to_excel(writer, sheet_name='Sheet4', index=False)
    df4.to_excel(writer, sheet_name='Sheet5', index=False)
    df5.to_excel(writer, sheet_name='Sheet6', index=False)
    
    # Save the third set of sheets
    df6.to_excel(writer, sheet_name='Sheet7', index=False)
    df7.to_excel(writer, sheet_name='Sheet8', index=False)
    df8.to_excel(writer, sheet_name='Sheet9', index=False)
    
    # Save the fourth set of sheets
    df9.to_excel(writer, sheet_name='Sheet10', index=False)
    df10.to_excel(writer, sheet_name='Sheet11', index=False)
    df11.to_excel(writer, sheet_name='Sheet12', index=False)     

print('Permutation Importance for the Transf1 model...')       

class PositionalEncoding(nn.Module):
    def __init__(self, device):
        super(PositionalEncoding, self).__init__()
        self.device = device

    def forward(self, x):

        batch_size, seq_length, d_model = x.shape

        pos = torch.arange(seq_length, device = self.device).unsqueeze(1)  # Shape: [seq_length, 1]
        i = torch.arange(d_model, device = self.device).unsqueeze(0)       # Shape: [1, d_model]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(d_model))

        # Compute the positional encodings for one sequence
        pos_encoding = pos * angle_rates  # Shape: [seq_length, d_model]
    
        # Apply sin to even indices (2i) and cos to odd indices (2i+1)
        pos_encoding[:, 0::2] = torch.sin(pos_encoding[:, 0::2])  # Apply sin to even indices
        pos_encoding[:, 1::2] = torch.cos(pos_encoding[:, 1::2])  # Apply cos to odd indices
    
        # Add the batch dimension by repeating for the batch size
        pos_encoding_batch = pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: [batch_size, seq_length, d_model]
    
        x = x + pos_encoding_batch  # Ensure shape compatibility for broadcasting
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size_code, hidden_size, output_size, static_size, input_size_feature, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_size_code, hidden_size)
        self.positional_encoding = PositionalEncoding(device=device)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size + 2*input_size_feature + 1, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        #self.fc_static = nn.Linear(static_size, hidden_size)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size + 2*input_size_feature + static_size + 1, 500),
            nn.ReLU(),
            nn.Linear(500, output_size)
        )

    def forward(self, codes, sequences, timestamps, timestamps_codes, static_vars):
        embedded_codes = self.embedding(codes)
        #sequences_with_codes = torch.cat([sequences, embedded_codes], dim=-1)
        sequences_with_codes = torch.cat([sequences, timestamps, embedded_codes, timestamps_codes], dim=-1)
        
        #sequences_with_codes = sequences_with_codes.transpose(0, 1)  # Transformer expects (seq_len, batch_size, feature_dim)

        #timestamps = torch.cat([timestamps, timestamps_codes], dim=-1)

        sequences_with_codes1 = self.positional_encoding(sequences_with_codes)
        
        transformer_output = self.transformer_encoder(sequences_with_codes1)
        transformer_output = transformer_output.mean(dim=1)  # Aggregate over sequence length
        
        #static_output = self.fc_static(static_vars)
        combined_output = torch.cat([transformer_output, static_vars], dim=-1)
        
        output = self.fc_out(combined_output)
        output = torch.sigmoid(output)
        
        return output

batch_size = 64

input_size_code = 5460
hidden_size = 111 #449 for 0, 6 and 12. 451 for 36. 455 for 60. 467 for 96
input_size_feature = sequences_train.shape[-1]
#input_size_seq = sequences_train.shape[-2] 
static_size = static_train.shape[-1]
output_size = 1

num_heads = 8
num_layers = 3

# Step 3: Define Loss Function and Optimizer
model5 = TransformerModel(input_size_code=input_size_code, hidden_size=hidden_size, output_size=output_size, static_size=static_size, input_size_feature = input_size_feature,
                          num_heads=num_heads, num_layers=num_layers).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model5.parameters(), lr=0.0001)

model5.load_state_dict(torch.load(r'D:\cgon080\ML\Data\Features2\Transf_PE_6m.pth'))
model5.to(device)

print('Permutation importance started...')

def pred_test1(test_loader, model):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels_batch in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels_batch = labels_batch.to(device)
            y_pred = model(codes_batch, sequences_batch, timestamps_batch, time_visit_batch, static_batch)
            predictions = y_pred.squeeze().round()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    return all_predictions, all_labels

prediction, labels_pred_test = pred_test1(test_loader=test_loader, model=model5) 

baseline_accuracy = metrics.accuracy_score(labels_pred_test, prediction)
print(f'Baseline accuracy: {baseline_accuracy:.4f}')
baseline_sensitivity = metrics.recall_score(labels_pred_test, prediction)
print(f'Baseline sens: {baseline_sensitivity:.4f}')
baseline_specificity = metrics.recall_score(labels_pred_test, prediction, pos_label=0)
print(f'Baseline spec: {baseline_specificity:.4f}')

print('Sequence features permutation importance started...')

def permute_feature_and_evaluate_tr(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences = sequences.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_sequences = sequences.clone()
            permuted_sequences[:, :, feature_index] = permuted_sequences[torch.randperm(permuted_sequences.size(0)), :, feature_index]
            
            #outputs = model(timevisits, codes, permuted_sequences, timestamps, static_vars)
            outputs = model(codes_batch, permuted_sequences.float(), timestamps_batch.float(), time_visit_batch.float(), static_batch.float()).squeeze()
            preds = outputs.round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Permuration importance 

import numpy as np

# Initialize matrices to store results of each iteration
num_features = values_tensors.size(2)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_tr(model5, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance)
#$print("Mean Specificity Importance per Feature:", mean_specificity_importance)

df = pd.DataFrame(mean_accuracy_importance)
df1 = pd.DataFrame(mean_sensitivity_importance)
df2 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf1.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet1', index=False)
#    df1.to_excel(writer, sheet_name='Sheet2', index=False)
#    df2.to_excel(writer, sheet_name='Sheet3', index=False)

print('Timestamps sequence features permutation importance started...')

def permute_feature_and_evaluate_time_seq_tr(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_timestamps = timestamps_batch.clone()
            permuted_timestamps[:, :, feature_index] = permuted_timestamps[torch.randperm(permuted_timestamps.size(0)), :, feature_index]
            
            outputs = model(codes_batch, sequences_batch.float(), permuted_timestamps.float(), time_visit_batch.float(), static_batch.float())
            #outputs = model(timevisits, codes, sequences, permuted_timestamps, static_vars)
            preds = outputs.squeeze().round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Initialize matrices to store results of each iteration
num_features = timestamps_test.size(2)
num_iterations = 100

#For month 0, is t

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_time_seq_tr(model5, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"TIMESTAMP: Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_t3)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_t3)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_t3)

df3 = pd.DataFrame(mean_accuracy_importance)
df4 = pd.DataFrame(mean_sensitivity_importance)
df5 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf1.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet4', index=False)
#    df1.to_excel(writer, sheet_name='Sheet5', index=False)
#    df2.to_excel(writer, sheet_name='Sheet6', index=False)

print('Static features permutation importance started...')


def permute_feature_and_evaluate_stat_tr(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_static_vars = static_batch.clone()
            permuted_static_vars[:, feature_index] = permuted_static_vars[torch.randperm(permuted_static_vars.size(0)), feature_index]
            
            outputs = model(codes_batch, sequences_batch.float(), timestamps_batch.float(), time_visit_batch.float(), permuted_static_vars.float())
            #outputs = model(timevisits, codes, sequences, timestamps, permuted_static_vars)
            preds = outputs.squeeze().round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Initialize matrices to store results of each iteration
num_features = static_test.size(1)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_stat_tr(model5, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"STATIC: Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_s4)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_s4)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_s4)

df6 = pd.DataFrame(mean_accuracy_importance)
df7 = pd.DataFrame(mean_sensitivity_importance)
df8 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf1.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet7', index=False)
#    df1.to_excel(writer, sheet_name='Sheet8', index=False)
#    df2.to_excel(writer, sheet_name='Sheet9', index=False)

print('ICD-10 features permutation importance started...')


def permute_feature_and_evaluate_codes_tr(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_codes = codes_batch.clone()
            permuted_codes = permuted_codes[torch.randperm(permuted_codes.size(0)), :]
            
            outputs = model(permuted_codes, sequences_batch.float(), timestamps_batch.float(), time_visit_batch.float(), static_batch.float())
            #outputs = model(timevisits, permuted_codes, sequences, timestamps, static_vars)
            preds = outputs.squeeze().round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

def permute_feature_and_evaluate_codes_time_tr(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for timevisits, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            timevisits = timevisits.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels = labels.to(device)
            
            # Permute the specified feature
            permuted_timevisits = timevisits.clone()
            permuted_timevisits = permuted_timevisits[torch.randperm(permuted_timevisits.size(0)), :]
            
            #outputs = model(permuted_timevisits, codes, sequences, timestamps, static_vars)
            outputs = model(codes_batch, sequences_batch.float(), timestamps_batch.float(), permuted_timevisits.float(), static_batch.float())
            preds = outputs.squeeze().round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

num_features = 2
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 50 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes_tr(model5, test_loader)

    accuracy_importance = baseline_accuracy - permuted_accuracy
    sensitivity_importance = baseline_sensitivity - permuted_sensitivity
    specificity_importance = baseline_specificity - permuted_specificity

    accuracy_importances.append(accuracy_importance)
    sensitivity_importances.append(sensitivity_importance)
    specificity_importances.append(specificity_importance)

    #print(f"CODES: Iteration {iteration + 1}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes_time_tr(model5, test_loader)

    accuracy_importance = baseline_accuracy - permuted_accuracy
    sensitivity_importance = baseline_sensitivity - permuted_sensitivity
    specificity_importance = baseline_specificity - permuted_specificity

    accuracy_importances.append(accuracy_importance)
    sensitivity_importances.append(sensitivity_importance)
    specificity_importances.append(specificity_importance)

    #print(f"TIMESTAMP: Iteration {iteration + 1}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_c4)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_c4)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_c4)

df9 = pd.DataFrame(mean_accuracy_importance)
df10 = pd.DataFrame(mean_sensitivity_importance)
df11 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf1.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet10', index=False)
#    df1.to_excel(writer, sheet_name='Sheet11', index=False)
#    df2.to_excel(writer, sheet_name='Sheet12', index=False)    

with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf1_6.xlsx') as writer:
    # Save the first set of sheets
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    df1.to_excel(writer, sheet_name='Sheet2', index=False)
    df2.to_excel(writer, sheet_name='Sheet3', index=False)
    
    # Save the second set of sheets
    df3.to_excel(writer, sheet_name='Sheet4', index=False)
    df4.to_excel(writer, sheet_name='Sheet5', index=False)
    df5.to_excel(writer, sheet_name='Sheet6', index=False)
    
    # Save the third set of sheets
    df6.to_excel(writer, sheet_name='Sheet7', index=False)
    df7.to_excel(writer, sheet_name='Sheet8', index=False)
    df8.to_excel(writer, sheet_name='Sheet9', index=False)
    
    # Save the fourth set of sheets
    df9.to_excel(writer, sheet_name='Sheet10', index=False)
    df10.to_excel(writer, sheet_name='Sheet11', index=False)
    df11.to_excel(writer, sheet_name='Sheet12', index=False)         

print('Permutation Importance for the Transf2 model...')  


class PositionalEncoding(nn.Module):
    def __init__(self, device):
        super(PositionalEncoding, self).__init__()
        self.device = device

    def forward(self, x, timestamps):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
            timestamps: Tensor of shape (batch_size, seq_length, d_model)
        """

        position = timestamps

        last_feature = position[:, :, -1:]
        repeated_features = last_feature.repeat(1, 1, 107) #448 for 0, 6 and 12. 449 for 36. 451 for 60. 457 for 96
        position = torch.cat([position, repeated_features], dim=2)

        batch_size, seq_length, d_model = position.shape

        i = torch.arange(d_model, device = self.device).unsqueeze(0)       # Shape: [1, d_model]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(d_model))

        pos_encoding = position * angle_rates

        pos_encoding[:, :, 0::2] = torch.sin(pos_encoding[:, :, 0::2])  # Apply sin to even indices
        pos_encoding[:, :, 1::2] = torch.cos(pos_encoding[:, :, 1::2])  # Apply cos to odd indices

        # Add the positional encodings to the input tensor `x`
        x = x + pos_encoding  # Ensure shape compatibility for broadcasting
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size_code, hidden_size, output_size, static_size, input_size_feature, num_heads, num_layers, input_size_seq):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_size_code, hidden_size)
        self.positional_encoding = PositionalEncoding(device=device)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size + input_size_feature, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size + input_size_feature + static_size, 500),
            nn.ReLU(),
            nn.Linear(500, output_size)
        )

    def forward(self, codes, sequences, timestamps, timestamps_codes, static_vars):
        embedded_codes = self.embedding(codes)
        sequences_with_codes = torch.cat([sequences, embedded_codes], dim=-1)

        timestamps = torch.cat([timestamps, timestamps_codes], dim=-1)

        sequences_with_codes1 = self.positional_encoding(sequences_with_codes, timestamps)
        
        transformer_output = self.transformer_encoder(sequences_with_codes1)
        transformer_output = transformer_output.mean(dim=1)  # Aggregate over sequence length
        
        combined_output = torch.cat([transformer_output, static_vars], dim=-1)
        
        output = self.fc_out(combined_output)
        output = torch.sigmoid(output)
        
        return output

batch_size = 64

input_size_code = 5460
hidden_size = 108 #449 for 0, 6 and 12. 450 for 36. 452 for 60. 458 for 96
input_size_feature = sequences_train.shape[-1]
input_size_seq = sequences_train.shape[-2] 
static_size = static_train.shape[-1]
output_size = 1

num_heads = 8
num_layers = 3

# Step 3: Define Loss Function and Optimizer
model6 = TransformerModel(input_size_code=input_size_code, hidden_size=hidden_size, output_size=output_size, static_size=static_size, input_size_feature = input_size_feature, input_size_seq = input_size_seq,
                          num_heads=num_heads, num_layers=num_layers).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model6.parameters(), lr=0.0001)

model6.load_state_dict(torch.load(r'D:\cgon080\ML\Data\Features2\Transf_6m.pth'))
model6.to(device)


print('Permutation importance started...')

prediction, labels_pred_test = pred_test1(test_loader=test_loader, model=model6) 

baseline_accuracy = metrics.accuracy_score(labels_pred_test, prediction)
print(f'Baseline accuracy: {baseline_accuracy:.4f}')
baseline_sensitivity = metrics.recall_score(labels_pred_test, prediction)
print(f'Baseline sens: {baseline_sensitivity:.4f}')
baseline_specificity = metrics.recall_score(labels_pred_test, prediction, pos_label=0)
print(f'Baseline spec: {baseline_specificity:.4f}')

print('Sequence features permutation importance started...')

# Permuration importance 

import numpy as np

# Initialize matrices to store results of each iteration
num_features = values_tensors.size(2)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_tr(model6, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance)
#$print("Mean Specificity Importance per Feature:", mean_specificity_importance)

df = pd.DataFrame(mean_accuracy_importance)
df1 = pd.DataFrame(mean_sensitivity_importance)
df2 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf2.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet1', index=False)
#    df1.to_excel(writer, sheet_name='Sheet2', index=False)
#    df2.to_excel(writer, sheet_name='Sheet3', index=False)

print('Timestamps sequence features permutation importance started...')

# Initialize matrices to store results of each iteration
num_features = timestamps_test.size(2)
num_iterations = 100

#For month 0, is t

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_time_seq_tr(model6, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"TIMESTAMP: Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_t3)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_t3)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_t3)

df3 = pd.DataFrame(mean_accuracy_importance)
df4 = pd.DataFrame(mean_sensitivity_importance)
df5 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf2.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet4', index=False)
#    df1.to_excel(writer, sheet_name='Sheet5', index=False)
#    df2.to_excel(writer, sheet_name='Sheet6', index=False)

print('Static features permutation importance started...')

# Initialize matrices to store results of each iteration
num_features = static_test.size(1)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in tqdm(range(num_features), desc=f"Iteration {iteration + 1} Features", leave=False):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_stat_tr(model6, test_loader, i)

        accuracy_importance = baseline_accuracy - permuted_accuracy
        sensitivity_importance = baseline_sensitivity - permuted_sensitivity
        specificity_importance = baseline_specificity - permuted_specificity

        accuracy_importances.append(accuracy_importance)
        sensitivity_importances.append(sensitivity_importance)
        specificity_importances.append(specificity_importance)

        #print(f"STATIC: Iteration {iteration + 1}, Feature {i}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_s4)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_s4)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_s4)

df6 = pd.DataFrame(mean_accuracy_importance)
df7 = pd.DataFrame(mean_sensitivity_importance)
df8 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf2.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet7', index=False)
#    df1.to_excel(writer, sheet_name='Sheet8', index=False)
#    df2.to_excel(writer, sheet_name='Sheet9', index=False)

print('ICD-10 features permutation importance started...')

num_features = 2
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 50 iterations
for iteration in tqdm(range(num_iterations), desc="Iterations"):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes_tr(model6, test_loader)

    accuracy_importance = baseline_accuracy - permuted_accuracy
    sensitivity_importance = baseline_sensitivity - permuted_sensitivity
    specificity_importance = baseline_specificity - permuted_specificity

    accuracy_importances.append(accuracy_importance)
    sensitivity_importances.append(sensitivity_importance)
    specificity_importances.append(specificity_importance)

    #print(f"CODES: Iteration {iteration + 1}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes_time_tr(model6, test_loader)

    accuracy_importance = baseline_accuracy - permuted_accuracy
    sensitivity_importance = baseline_sensitivity - permuted_sensitivity
    specificity_importance = baseline_specificity - permuted_specificity

    accuracy_importances.append(accuracy_importance)
    sensitivity_importances.append(sensitivity_importance)
    specificity_importances.append(specificity_importance)

    #print(f"TIMESTAMP: Iteration {iteration + 1}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    # Store the results of the current iteration
    accuracy_matrix_att[iteration, :] = np.array(accuracy_importances)
    sensitivity_matrix_att[iteration, :] = np.array(sensitivity_importances)
    specificity_matrix_att[iteration, :] = np.array(specificity_importances)

# Compute the mean of accuracy, sensitivity, and specificity importances
mean_accuracy_importance = np.mean(accuracy_matrix_att, axis=0)
mean_sensitivity_importance = np.mean(sensitivity_matrix_att, axis=0)
mean_specificity_importance = np.mean(specificity_matrix_att, axis=0)

#print("Mean Accuracy Importance per Feature:", mean_accuracy_importance_c4)
#print("Mean Sensitivity Importance per Feature:", mean_sensitivity_importance_c4)
#print("Mean Specificity Importance per Feature:", mean_specificity_importance_c4)

df9 = pd.DataFrame(mean_accuracy_importance)
df10 = pd.DataFrame(mean_sensitivity_importance)
df11 = pd.DataFrame(mean_specificity_importance)

#with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf2.xlsx') as writer:
#    df.to_excel(writer, sheet_name='Sheet10', index=False)
#    df1.to_excel(writer, sheet_name='Sheet11', index=False)
#    df2.to_excel(writer, sheet_name='Sheet12', index=False)       

with pd.ExcelWriter(r'D:\cgon080\ML\Data\Features2\PI_Transf2_6.xlsx') as writer:
    # Save the first set of sheets
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    df1.to_excel(writer, sheet_name='Sheet2', index=False)
    df2.to_excel(writer, sheet_name='Sheet3', index=False)
    
    # Save the second set of sheets
    df3.to_excel(writer, sheet_name='Sheet4', index=False)
    df4.to_excel(writer, sheet_name='Sheet5', index=False)
    df5.to_excel(writer, sheet_name='Sheet6', index=False)
    
    # Save the third set of sheets
    df6.to_excel(writer, sheet_name='Sheet7', index=False)
    df7.to_excel(writer, sheet_name='Sheet8', index=False)
    df8.to_excel(writer, sheet_name='Sheet9', index=False)
    
    # Save the fourth set of sheets
    df9.to_excel(writer, sheet_name='Sheet10', index=False)
    df10.to_excel(writer, sheet_name='Sheet11', index=False)
    df11.to_excel(writer, sheet_name='Sheet12', index=False) 