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

# Step 1: Load Data

gpu = 2
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Visits
#base_path = r'C:\Users\cgon080\OneDrive - The University of Auckland\Documents\'
#base_path = r'D:\cgon080\ML\DifferentTest\Time0m_scaled\'
#visits_df = pd.read_csv(r'D:\cgon080\ML\DifferentTest\Time0m_scaled\Codes_visit.csv')
visits_df = pd.read_csv(r'D:\cgon080\ML\DifferentTest\Time36m_retrain\Codes_visit1.csv')

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

#data_folder = r"D:\cgon080\ML\DifferentTest\Time0m_scaled\DataPython"
data_folder = r"D:\cgon080\ML\DifferentTest\Time36m_retrain\DataPython"   
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
#static_df = pd.read_csv(r'D:\cgon080\ML\DifferentTest\Time0m_scaled\Static_features.csv')
static_df = pd.read_csv(r'D:\cgon080\ML\DifferentTest\Time36m_retrain\Static_features.csv')
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

codes_train.shape
time_visit_train.shape
sequences_train.shape
sequences_test.shape
sequences_val.shape
timestamps_train.shape
timestamps_test.shape
static_train.shape
static_test.shape
labels_train.shape
labels_test.shape

#######################################################################################################
#Using attention mechanism 
#######################################################################################################
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
        #batch_size, seq_length, feature_dim = x.size()

        #x = x.transpose(2,1)

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(-2, -1)) / math.sqrt(self.input_size)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, values)
        #attended_values = attended_values.transpose(-2, -1)

        # Transpose back to the original dimensions
        #attended_values = attended_values.transpose(1, 2)

        return attention_weights, attended_values

class SelfAttention(nn.Module):
    def __init__(self, input_size, device):
        super(SelfAttention, self).__init__()
        self.device = device
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.input_size = input_size
        #self.max_length = max_length

    def forward(self, x, timestamps):
        #batch_size, seq_length, _ = x.size()

        # Add positional embeddings using the timestamps
        position_embeddings = self._get_position_embeddings(timestamps)
        x = x + position_embeddings
        x = x.transpose(2,1)

        # Apply linear projections
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        #print(f"Queries: {queries.shape}, Keys: {keys.shape}, Values: {values.shape}")

        # Calculate attention scores and apply attention
        scores = torch.bmm(queries, keys.transpose(-2, -1)) / math.sqrt(self.input_size)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attended_values = torch.bmm(attention_weights, values)

        return attention_weights, attended_values
    
    def _get_position_embeddings(self, timestamps):
        # Calculate positional encodings using sine and cosine functions
        #batch_size, seq_length, d_model = timestamps.shape

        position = timestamps

        last_feature = position[:, :, -1:]
        #print(last_feature)
        repeated_features = last_feature.repeat(1, 1, 99)
        #print(repeated_features.shape)
        position = torch.cat([position, repeated_features], dim=2)

        batch_size, seq_length, d_model = position.shape

        i = torch.arange(d_model, device = self.device).unsqueeze(0)       # Shape: [1, d_model]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(d_model))

        pos_encoding = position * angle_rates

        pos_encoding[:, :, 0::2] = torch.sin(pos_encoding[:, :, 0::2])  # Apply sin to even indices
        pos_encoding[:, :, 1::2] = torch.cos(pos_encoding[:, :, 1::2])  # Apply cos to odd indices

        '''
        i = torch.arange(d_model, device = self.device).unsqueeze(0)       # Shape: [1, d_model]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(d_model))

        pos_encoding = pos * angle_rates

        last_feature = position[:, :, -1:]
        #print(last_feature)
        repeated_features = last_feature.repeat(1, 1, 99)
        #print(repeated_features.shape)
        position = torch.cat([position, repeated_features], dim=2)
        
        i = torch.arange(d_model, device = self.device).unsqueeze(0)       # Shape: [1, d_model]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(d_model))

        pos_encoding = position * angle_rates
        
        div_term = torch.exp(torch.arange(d_model, dtype=torch.float32, device=timestamps.device) * -(math.log(10000.0) / d_model))

        pe = torch.zeros(batch_size, seq_length, d_model, device=timestamps.device)

        sin_values = torch.sin(position * div_term)
        cos_values = torch.cos(position * div_term)

        pe[:, :, 0::2] = sin_values[:, :, 0::2]
        pe[:, :, 1::2] = cos_values[:, :, 1::2]

        last_feature = pe[:, :, -1:]
        #print(last_feature)
        repeated_features = last_feature.repeat(1, 1, 99)
        #print(repeated_features.shape)
        pe = torch.cat([pe, repeated_features], dim=2)
        '''
        return pos_encoding

class FeatureAttentionStats(nn.Module):
    def __init__(self, input_size):
        super(FeatureAttentionStats, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.input_size = input_size

    def forward(self, x):
        #batch_size, seq_length, feature_dim = x.size()

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.mm(queries, keys.transpose(-2, -1)) / math.sqrt(self.input_size)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        attended_values = torch.mm(attention_weights, values)
        #print(attention_weights.shape)

        # Transpose back to the original dimensions
        #attended_values = attended_values.transpose(1, 2)

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

num_epochs = 2000
batch_size = 64

input_size_code = 5460
hidden_size = 100
input_size_feature = sequences_train.shape[-1]
input_size_seq = sequences_train.shape[-2] 
static_size = static_train.shape[-1]
output_size = 1

train_dataset = TensorDataset(time_visit_train, codes_train, sequences_train, timestamps_train, static_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(time_visit_val, codes_val, sequences_val, timestamps_val, static_val, labels_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Step 3: Define Loss Function and Optimizer
model = SelfAttentionModel(input_size_code=input_size_code, hidden_size=hidden_size, output_size=output_size, static_size=static_size, input_size_feature = input_size_feature, input_size_seq = input_size_seq).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Step 4: Train the Model

train_losses = []
val_losses = []
train_accs = []
val_accs = []
at_weights_time = []
at_weights_feature = []
at_weights_static = []

model.train()
for epoch in range(num_epochs):
    correct_train = 0
    total_train = 0
    running_loss = 0.0
    for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels_batch in train_loader:
        time_visit_batch = time_visit_batch.to(device)
        codes_batch = codes_batch.to(device)
        sequences_batch = sequences_batch.to(device)
        timestamps_batch = timestamps_batch.to(device)
        static_batch = static_batch.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad()
        outputs, feature_attention_weights, attention_weights, static_attention_weights = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
        loss = criterion(outputs.squeeze(), labels_batch)
        loss.backward()
        optimizer.step()

        #_, predicted = torch.max(outputs.data, 1)
        total_train += labels_batch.size(0)
        correct_train += (outputs.squeeze().round() == labels_batch).sum().item()
        
        running_loss += loss.item() * sequences_batch.size(0)

        #if (epoch+1) % 2 == 0:
        #    print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")
    
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)
    at_weights_time = attention_weights
    at_weights_feature = feature_attention_weights
    #at_weights_static = static_attended_values

    #correct = 0
    #total = len(labels_val)
    model.eval()
    correct_val = 0
    total_val = 0
    val_running_loss = 0.0
    with torch.no_grad():
        #all_predictions = []
        #all_labels = []
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels_batch in val_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences_batch = sequences_batch.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels_batch = labels_batch.to(device)
            y_pred, feature_attention_weights, attention_weights, static_attention_weights = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
            val_loss = criterion(y_pred.squeeze(), labels_batch)
            
            total_val += labels_batch.size(0)
            correct_val += (y_pred.squeeze().round() == labels_batch).sum().item()

            val_running_loss += val_loss.item() * sequences_batch.size(0)

    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_acc = correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}")


# Plotting the training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(r'D:\cgon080\ML\DifferentTest\Time36m_retrain\loss_plot.png', dpi=300, bbox_inches='tight')
plt.close()

plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(r'D:\cgon080\ML\DifferentTest\Time36m_retrain\Acc_plot.png', dpi=300, bbox_inches='tight')
plt.close()

torch.save(model.state_dict(), r'D:\cgon080\ML\DifferentTest\SelfAt_PE_TimeStamps36m_a.pth')

test_dataset = TensorDataset(time_visit_test, codes_test, sequences_test, timestamps_test, static_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = len(labels_test)
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
        correct += torch.sum((predictions == labels_batch).float())
        #print(f'Correct: {correct:.4f}')
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

print('Test accuracy: {}'.format(correct/total))   
print('Test accuracy: {}'.format(metrics.accuracy_score(all_labels, all_predictions)))  
metrics.confusion_matrix(all_labels, all_predictions) 

print('Bootstrap part started...')

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
            y_pred, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
            predictions = y_pred.squeeze().round()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    return all_predictions, all_labels

prediction, labels_pred_test = pred_test(test_loader=test_loader, model=model) 

num_bootstraps = 1000

accuracy_scores_att = []
sensitivity_scores_att = []
specificity_scores_att = []
f1_scores_att = []

for _ in range(num_bootstraps):
    # Sample with replacement from the predictions and labels
    indices = np.random.choice(len(prediction), size=len(prediction), replace=True)
    bootstrap_predictions = [prediction[i] for i in indices]
    bootstrap_labels = [labels_pred_test[i] for i in indices]
    
    # Calculate accuracy for the bootstrap sample
    accuracy = metrics.accuracy_score(bootstrap_labels, bootstrap_predictions)
    accuracy_scores_att.append(accuracy)

    sensitivity = metrics.recall_score(bootstrap_labels, bootstrap_predictions)
    sensitivity_scores_att.append(sensitivity)

    specificity = metrics.recall_score(bootstrap_labels, bootstrap_predictions, pos_label=0)
    specificity_scores_att.append(specificity)

    F1_score = metrics.f1_score(bootstrap_labels, bootstrap_predictions)
    f1_scores_att.append(F1_score)


# Calculate mean and standard deviation of accuracy values
mean_accuracy = np.mean(accuracy_scores_att)
print(f'Accuracy: {mean_accuracy:.4f}')
std_accuracy = np.std(accuracy_scores_att)
print(f'SD Accuracy: {std_accuracy:.4f}')

mean_sensitivity = np.mean(sensitivity_scores_att)
print(f'Sensitivity: {mean_sensitivity:.4f}')
std_dev_sensitivity = np.std(sensitivity_scores_att)
print(f'SD Sensitivity: {std_dev_sensitivity:.4f}')

mean_specificity = np.mean(specificity_scores_att)
print(f'Specificity: {mean_specificity:.4f}')
std_dev_specificity = np.std(specificity_scores_att)
print(f'SD Specificity: {std_dev_specificity:.4f}')

mean_f1 = np.mean(f1_scores_att)
print(f'F1 score: {mean_f1:.4f}')
std_dev_f1 = np.std(f1_scores_att)
print(f'SD F1 score: {std_dev_f1:.4f}')
'''
print('Permutation importance started...')

prediction, labels_pred_test = pred_test(test_loader=test_loader, model=model) 

baseline_accuracy = metrics.accuracy_score(labels_pred_test, prediction)
print(f'Baseline accuracy: {baseline_accuracy:.4f}')
baseline_sensitivity = metrics.recall_score(labels_pred_test, prediction)
print(f'Baseline sens: {baseline_sensitivity:.4f}')
baseline_specificity = metrics.recall_score(labels_pred_test, prediction, pos_label=0)
print(f'Baseline spec: {baseline_specificity:.4f}')

print('Sequence features permutation importance started...')

def permute_feature_and_evaluate_att1(model, test_loader, feature_index):
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
            labels_batch = labels_batch.to(device)
            
            # Permute the specified feature
            permuted_sequences = sequences.clone()
            permuted_sequences[:, feature_index, :] = permuted_sequences[torch.randperm(permuted_sequences.size(0)), feature_index, :]
            
            #outputs = model(timevisits, codes, permuted_sequences, timestamps, static_vars)
            outputs, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch.double(), codes_batch, permuted_sequences.double(), timestamps_batch.double(), static_batch.double()).squeeze()
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
num_features = values_tensors.size(1)
num_iterations = 100

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in range(num_iterations):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in range(num_features):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_att(model, test_loader, i)

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

with pd.ExcelWriter(r'D:\cgon080\ML\DifferentTest\output_att.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    df1.to_excel(writer, sheet_name='Sheet2', index=False)
    df2.to_excel(writer, sheet_name='Sheet3', index=False)

print('Timestamps sequence features permutation importance started...')

def permute_feature_and_evaluate_time_seq_att(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences = sequences.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Permute the specified feature
            permuted_timestamps = timestamps_batch.clone()
            permuted_timestamps[:, feature_index, :] = permuted_timestamps[torch.randperm(permuted_timestamps.size(0)), feature_index, :]
            
            outputs, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch.double(), codes_batch, sequences_batch.double(), permuted_timestamps.double(), static_batch.double()).squeeze()
            #outputs = model(timevisits, codes, sequences, permuted_timestamps, static_vars)
            preds = outputs.round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

# Initialize matrices to store results of each iteration
num_features = timestamps_test.size(1)
num_iterations = 100

#For month 0, is t

accuracy_matrix_att = np.zeros((num_iterations, num_features))
sensitivity_matrix_att = np.zeros((num_iterations, num_features))
specificity_matrix_att = np.zeros((num_iterations, num_features))

# Perform 100 iterations
for iteration in range(num_iterations):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in range(num_features):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_time_seq_att(model, test_loader, i)

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

df = pd.DataFrame(mean_accuracy_importance)
df1 = pd.DataFrame(mean_sensitivity_importance)
df2 = pd.DataFrame(mean_specificity_importance)

with pd.ExcelWriter(r'D:\cgon080\ML\DifferentTest\output_att.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet4', index=False)
    df1.to_excel(writer, sheet_name='Sheet5', index=False)
    df2.to_excel(writer, sheet_name='Sheet6', index=False)

print('Static features permutation importance started...')

def permute_feature_and_evaluate_stat_att(model, test_loader, feature_index):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences = sequences.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Permute the specified feature
            permuted_static_vars = static_batch.clone()
            permuted_static_vars[:, feature_index] = permuted_static_vars[torch.randperm(permuted_static_vars.size(0)), feature_index]
            
            outputs, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch.double(), codes_batch, sequences_batch.double(), timestamps_batch.double(), permuted_static_vars.double()).squeeze()
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

# Perform 50 iterations
for iteration in range(num_iterations):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    for i in range(num_features):  # Assuming sequences is the feature tensor
        permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_stat_cnn(model, test_loader, i)

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

df = pd.DataFrame(mean_accuracy_importance)
df1 = pd.DataFrame(mean_sensitivity_importance)
df2 = pd.DataFrame(mean_specificity_importance)

with pd.ExcelWriter(r'D:\cgon080\ML\DifferentTest\output_att.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet7', index=False)
    df1.to_excel(writer, sheet_name='Sheet8', index=False)
    df2.to_excel(writer, sheet_name='Sheet9', index=False)

print('ICD-10 features permutation importance started...')

def permute_feature_and_evaluate_codes_att(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for time_visit_batch, codes, sequences_batch, timestamps_batch, static_vars, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences = sequences.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Permute the specified feature
            permuted_codes = codes.clone()
            permuted_codes = permuted_codes[torch.randperm(permuted_codes.size(0)), :]
            
            outputs, feature_attention_weights, attention_weights, static_attended_values = model(time_visit_batch.double(), permuted_codes, sequences_batch.double(), timestamps_batch.double(), static_vars.double()).squeeze()
            #outputs = model(timevisits, permuted_codes, sequences, timestamps, static_vars)
            preds = outputs.round()  # Assuming binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = metrics.accuracy_score(all_labels, all_preds)
            sensitivity = metrics.recall_score(all_labels, all_preds, zero_division=0)
            specificity = metrics.recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    
    return accuracy, sensitivity, specificity

def permute_feature_and_evaluate_codes_time_att(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for timevisits, codes_batch, sequences_batch, timestamps_batch, static_vars, labels in test_loader:
            time_visit_batch = time_visit_batch.to(device)
            codes_batch = codes_batch.to(device)
            sequences = sequences.to(device)
            timestamps_batch = timestamps_batch.to(device)
            static_batch = static_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Permute the specified feature
            permuted_timevisits = timevisits.clone()
            permuted_timevisits = permuted_timevisits[torch.randperm(permuted_timevisits.size(0)), :]
            
            #outputs = model(permuted_timevisits, codes, sequences, timestamps, static_vars)
            outputs = model(codes_batch, permuted_timevisits.double(), sequences_batch.double(), timestamps_batch.double(), static_vars.double()).squeeze()
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
for iteration in range(num_iterations):
    accuracy_importances = []
    sensitivity_importances = []
    specificity_importances = []

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes_cnn(model, test_loader)

    accuracy_importance = baseline_accuracy - permuted_accuracy
    sensitivity_importance = baseline_sensitivity - permuted_sensitivity
    specificity_importance = baseline_specificity - permuted_specificity

    accuracy_importances.append(accuracy_importance)
    sensitivity_importances.append(sensitivity_importance)
    specificity_importances.append(specificity_importance)

    #print(f"CODES: Iteration {iteration + 1}, Accuracy Importance: {accuracy_importance}, Sensitivity Importance: {sensitivity_importance}, Specificity Importance: {specificity_importance}")

    permuted_accuracy, permuted_sensitivity, permuted_specificity = permute_feature_and_evaluate_codes_time_cnn(model, test_loader)

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

df = pd.DataFrame(mean_accuracy_importance)
df1 = pd.DataFrame(mean_sensitivity_importance)
df2 = pd.DataFrame(mean_specificity_importance)

with pd.ExcelWriter(r'D:\cgon080\ML\DifferentTest\output_att.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet10', index=False)
    df1.to_excel(writer, sheet_name='Sheet11', index=False)
    df2.to_excel(writer, sheet_name='Sheet12', index=False)

'''    