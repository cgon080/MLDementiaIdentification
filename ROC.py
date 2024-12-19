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

def calculate_auc_ci(y_true, y_scores, n_bootstrap=1000, ci_level=0.95):
    # Ensure y_true and y_scores are NumPy arrays
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    y_scores = y_scores.cpu().numpy() if isinstance(y_scores, torch.Tensor) else np.array(y_scores)

    # Compute the original AUC
    auc_original = roc_auc_score(y_true, y_scores)

    # Bootstrap to calculate AUCs
    aucs = []
    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = resample(np.arange(len(y_true))).astype(int)  # Ensure indices are integers
        if len(np.unique(y_true[indices])) < 2:  # Avoid invalid splits
            continue
        auc_resample = roc_auc_score(y_true[indices], y_scores[indices])
        aucs.append(auc_resample)

    # Calculate the confidence interval
    lower_bound = np.percentile(aucs, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(aucs, (1 + ci_level) / 2 * 100)

    return auc_original, (lower_bound, upper_bound)


gpu = 2
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Visits
#base_path = r'C:\Users\cgon080\OneDrive - The University of Auckland\Documents\'
#base_path = r'D:\cgon080\ML\DifferentTest\Time0m_scaled\'
#visits_df = pd.read_csv(r'D:\cgon080\ML\DifferentTest\Time96m_retrain\Codes_visit1.csv')
visits_df = pd.read_csv(r'D:\cgon080\ML\Data\Features2\Time96m_retrain\Codes_visit1.csv')

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
data_folder = r"D:\cgon080\ML\Data\Features2\Time96m_retrain\DataPython"   
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
static_df = pd.read_csv(r'D:\cgon080\ML\Data\Features2\Time96m_retrain\Static_features.csv')
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

print(time_visit_test.shape)
print(codes_test.shape)
print(sequences_test.shape)
print(timestamps_test.shape)

roc_data = {}

#RNN

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

model.load_state_dict(torch.load(r'D:\cgon080\ML\Data\Features2\RNN_96.pth'))
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

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)
_, ci = calculate_auc_ci(all_labels, all_probs)

roc_data['RNN'] = (fpr, tpr, auc)

print('Performance RNN')
print('Test accuracy: {}'.format(metrics.accuracy_score(all_labels, all_predictions)))  
print('Sensitivity: {}'.format(metrics.recall_score(all_labels, all_predictions)))
print('Specificity: {}'.format(metrics.recall_score(all_labels, all_predictions, pos_label=0)))
print('F1_score: {}'.format(metrics.f1_score(all_labels, all_predictions)))
print(f"AUC: {auc:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
'''
#CNN

import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes, embedding_dim, input_size_data):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(5460, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.fc1 = nn.Linear(128 * 180, 256) # Check how to calculate 494 = final_seq_len #0: 455, 6: 414, 12: 388, 36: 329, 60: 268, 96: 180
        self.dropout = nn.Dropout(0.5)
        #self.fc2 = nn.Linear(input_size_data + 256, num_classes)
        self.nn1 = nn.Sequential(
            nn.Linear(input_size_data + 256, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Linear(100,num_classes)
            )

    def forward(self, codes, timevisits, sequences, timestamps, static_vars):
        embedded_codes = self.embedding(codes)
        embedded_codes = embedded_codes.transpose(2, 1)
        sequences_with_time = torch.cat([sequences, timestamps, embedded_codes, timevisits], dim=-2)
        x = F.relu(self.conv1(sequences_with_time))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, static_vars], dim=-1)
        #x = self.fc2(x)
        x = self.nn1(x)
        return torch.sigmoid(x)
        #return x

embedding_dim = 100
num_classes = 1    
input_channels = sequences_train.shape[-1] + timestamps_train.shape[-1] + embedding_dim + 1
input_size_data =  static_train.shape[-1]   

model1 = CNNModel(input_channels=input_channels, num_classes=num_classes, embedding_dim=embedding_dim, input_size_data=input_size_data).to(device)  
model1 = model1.double()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.00001)

model1.load_state_dict(torch.load(r'D:\cgon080\ML\DifferentTest\CNN_96.pth'))
model1.to(device)

test_dataset = TensorDataset(time_visit_test.transpose(1,2), codes_test, sequences_test.transpose(1,2), timestamps_test.transpose(1,2), static_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = len(labels_test)
model1.eval()
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
        y_pred = model1(codes_batch, time_visit_batch, sequences_batch, timestamps_batch, static_batch)
        predictions = y_pred.squeeze().round()
        correct += torch.sum((predictions == labels_batch).float())
        #print(f'Correct: {correct:.4f}')
        all_probs.extend(y_pred.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)
_, ci = calculate_auc_ci(all_labels, all_probs)

roc_data['CNN'] = (fpr, tpr, auc)

print('Performance CNN')
print('Test accuracy: {}'.format(metrics.accuracy_score(all_labels, all_predictions)))  
print('Sensitivity: {}'.format(metrics.recall_score(all_labels, all_predictions)))
print('Specificity: {}'.format(metrics.recall_score(all_labels, all_predictions, pos_label=0)))
print('F1_score: {}'.format(metrics.f1_score(all_labels, all_predictions)))
print(f"AUC: {auc:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

#Attention 1

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
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()
        self.device = device
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.input_size = input_size
        #self.max_length = max_length

    def forward(self, x):
        batch_size, seq_length, d_model = x.size()

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
        self.embedding = nn.Embedding(input_size_code, hidden_size)                    #input(64, 463) -> output(64, 463, 100)
        self.feature_attention = FeatureAttention(hidden_size + 2*input_size_feature + 1)  #input(64, 463, 100+63+63) -> output(64, 463, 100+63+63)
        self.temporal_attention = SelfAttention(input_size_seq)                      #input(64, 463, 100+63+63) -> output(64, 100+63+63, 463)
        self.static_attention = FeatureAttentionStats(static_size)                     #input(64, 197) -> output(64, 197)
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
model2 = SelfAttentionModel(input_size_code=input_size_code, hidden_size=hidden_size, output_size=output_size, static_size=static_size, input_size_feature = input_size_feature, input_size_seq = input_size_seq).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001)

model2.load_state_dict(torch.load(r'D:\cgon080\ML\DifferentTest\SelfAt_NoPE_96months.pth'))
model2.to(device)

test_dataset = TensorDataset(time_visit_test, codes_test, sequences_test, timestamps_test, static_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = len(labels_test)
model2.eval()
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
        y_pred, feature_attention_weights, attention_weights, static_attended_values = model2(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
        predictions = y_pred.squeeze().round()
        correct += torch.sum((predictions == labels_batch).float())
        #print(f'Correct: {correct:.4f}')
        all_probs.extend(y_pred.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)
_, ci = calculate_auc_ci(all_labels, all_probs)

roc_data['Att1'] = (fpr, tpr, auc)

print('Performance Att1')
print('Test accuracy: {}'.format(metrics.accuracy_score(all_labels, all_predictions)))  
print('Sensitivity: {}'.format(metrics.recall_score(all_labels, all_predictions)))
print('Specificity: {}'.format(metrics.recall_score(all_labels, all_predictions, pos_label=0)))
print('F1_score: {}'.format(metrics.f1_score(all_labels, all_predictions)))
print(f"AUC: {auc:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
'''
# Attention 2 

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

model3.load_state_dict(torch.load(r'D:\cgon080\ML\Data\Features2\SelfAt_Normal_PE_96m.pth'))
model3.to(device)

test_dataset = TensorDataset(time_visit_test, codes_test, sequences_test, timestamps_test, static_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)
_, ci = calculate_auc_ci(all_labels, all_probs)

roc_data['Att2'] = (fpr, tpr, auc)

print('Performance Att2')
print('Test accuracy: {}'.format(metrics.accuracy_score(all_labels, all_predictions)))  
print('Sensitivity: {}'.format(metrics.recall_score(all_labels, all_predictions)))
print('Specificity: {}'.format(metrics.recall_score(all_labels, all_predictions, pos_label=0)))
print('F1_score: {}'.format(metrics.f1_score(all_labels, all_predictions)))
print(f"AUC: {auc:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

'''
# Attention 3

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

    def forward(self, x, timestamps):

        # Add positional embeddings using the timestamps
        position_embeddings = self._get_position_embeddings(timestamps)
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
    
    def _get_position_embeddings(self, timestamps):
        # Calculate positional encodings using sine and cosine functions

        position = timestamps

        last_feature = position[:, :, -1:]
        repeated_features = last_feature.repeat(1, 1, 99)
        position = torch.cat([position, repeated_features], dim=2)

        batch_size, seq_length, d_model = position.shape

        i = torch.arange(d_model, device = self.device).unsqueeze(0)       # Shape: [1, d_model]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(d_model))

        pos_encoding = position * angle_rates

        pos_encoding[:, :, 0::2] = torch.sin(pos_encoding[:, :, 0::2])  # Apply sin to even indices
        pos_encoding[:, :, 1::2] = torch.cos(pos_encoding[:, :, 1::2])  # Apply cos to odd indices

        return pos_encoding

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
        self.embedding = nn.Embedding(input_size_code, hidden_size)                    #input(64, 463) -> output(64, 463, 100)
        self.feature_attention = FeatureAttention(hidden_size + input_size_feature)    #input(64, 463, 100+63) -> output(64, 463, 100+63)
        self.temporal_attention = SelfAttention(input_size_seq, device=device)                        #input(64, 463, 100+63) -> output(64, 100+63, 463)
        self.static_attention = FeatureAttentionStats(static_size)                     #input(64, 197) -> output(64, 197)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size + input_size_feature, 250), # 63 is the number of sequences
            nn.ReLU(),
            nn.Linear(250, output_size)
        )

    def forward(self, timestamps_codes, codes, sequences, timestamps, static_vars):
        # Embed ICD-10 codes
        embedded_codes = self.embedding(codes)

        # Concatenate embedded codes with the other sequences
        sequences_with_codes = torch.cat([sequences, embedded_codes], dim=-1)
        
        feature_attention_weights, attended_values = self.feature_attention(sequences_with_codes)

        # Apply self-attention (temporal) with positional embeddings
        timestamps = torch.cat([timestamps, timestamps_codes], dim=-1)

        attention_weights, attended_values_1 = self.temporal_attention(attended_values, timestamps)

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
model4 = SelfAttentionModel(input_size_code=input_size_code, hidden_size=hidden_size, output_size=output_size, static_size=static_size, input_size_feature = input_size_feature, input_size_seq = input_size_seq).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model4.parameters(), lr=0.00001)

model4.load_state_dict(torch.load(r'D:\cgon080\ML\DifferentTest\SelfAt_PE_TimeStamps96m_a.pth'))
model4.to(device)

test_dataset = TensorDataset(time_visit_test, codes_test, sequences_test, timestamps_test, static_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = len(labels_test)
model4.eval()
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
        y_pred, feature_attention_weights, attention_weights, static_attended_values = model4(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
        predictions = y_pred.squeeze().round()
        correct += torch.sum((predictions == labels_batch).float())
        #print(f'Correct: {correct:.4f}')
        all_probs.extend(y_pred.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)
_, ci = calculate_auc_ci(all_labels, all_probs)

roc_data['Att3'] = (fpr, tpr, auc)

print('Performance Att3')
print('Test accuracy: {}'.format(metrics.accuracy_score(all_labels, all_predictions)))  
print('Sensitivity: {}'.format(metrics.recall_score(all_labels, all_predictions)))
print('Specificity: {}'.format(metrics.recall_score(all_labels, all_predictions, pos_label=0)))
print('F1_score: {}'.format(metrics.f1_score(all_labels, all_predictions)))
print(f"AUC: {auc:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
'''
# Transformer 1

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
hidden_size = 137 #449 for 0, 6 and 12. 451 for 36. 455 for 60. 467 for 96
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

model5.load_state_dict(torch.load(r'D:\cgon080\ML\Data\Features2\Transf_PE_96m.pth'))
model5.to(device)

test_dataset = TensorDataset(time_visit_test, codes_test, sequences_test, timestamps_test, static_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = len(labels_test)
model5.eval()
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
        y_pred = model5(codes_batch, sequences_batch, timestamps_batch, time_visit_batch, static_batch)
        predictions = y_pred.squeeze().round()
        correct += torch.sum((predictions == labels_batch).float())
        #print(f'Correct: {correct:.4f}')
        all_probs.extend(y_pred.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)
_, ci = calculate_auc_ci(all_labels, all_probs)

roc_data['Transf1'] = (fpr, tpr, auc)

print('Performance Transf1')
print('Test accuracy: {}'.format(metrics.accuracy_score(all_labels, all_predictions)))  
print('Sensitivity: {}'.format(metrics.recall_score(all_labels, all_predictions)))
print('Specificity: {}'.format(metrics.recall_score(all_labels, all_predictions, pos_label=0)))
print('F1_score: {}'.format(metrics.f1_score(all_labels, all_predictions)))
print(f"AUC: {auc:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

# Transformer 2

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
        repeated_features = last_feature.repeat(1, 1, 120) #448 for 0, 6 and 12. 449 for 36. 451 for 60. 457 for 96
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
hidden_size = 121 #449 for 0, 6 and 12. 450 for 36. 452 for 60. 458 for 96
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

model6.load_state_dict(torch.load(r'D:\cgon080\ML\Data\Features2\Transf_96m.pth'))
model6.to(device)

test_dataset = TensorDataset(time_visit_test, codes_test, sequences_test, timestamps_test, static_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = len(labels_test)
model6.eval()
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
        y_pred = model6(codes_batch, sequences_batch, timestamps_batch, time_visit_batch, static_batch)
        predictions = y_pred.squeeze().round()
        correct += torch.sum((predictions == labels_batch).float())
        #print(f'Correct: {correct:.4f}')
        all_probs.extend(y_pred.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)
_, ci = calculate_auc_ci(all_labels, all_probs)

roc_data['Transf2'] = (fpr, tpr, auc)

print('Performance Transf2')
print('Test accuracy: {}'.format(metrics.accuracy_score(all_labels, all_predictions)))  
print('Sensitivity: {}'.format(metrics.recall_score(all_labels, all_predictions)))
print('Specificity: {}'.format(metrics.recall_score(all_labels, all_predictions, pos_label=0)))
print('F1_score: {}'.format(metrics.f1_score(all_labels, all_predictions)))
print(f"AUC: {auc:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

plt.figure(figsize=(10, 8))
for model_name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")

# Customize plot
plt.title("ROC Curves for Models - Time window 8")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig(r'D:\cgon080\ML\Data\Features2\ROC_96.png', dpi=300, bbox_inches='tight')
plt.close()

