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
visits_df = pd.read_csv(r'D:\cgon080\ML\Data\Features1\Time96m_retrain\Codes_visit1.csv')

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
data_folder = r"D:\cgon080\ML\Data\Features1\Time96m_retrain\DataPython"   
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
static_df = pd.read_csv(r'D:\cgon080\ML\Data\Features1\Time96m_retrain\Static_features.csv')
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
    
    #def initialize_parameters(self):
    #    for name, param in self.named_parameters():
    #        if param.requires_grad:
    #            if len(param.shape) >= 2:
    #                init.xavier_normal_(param.data)
    #            else:
    #                init.normal_(param.data)


num_epochs = 50
batch_size = 64
embedding_dim = 100
input_size = sequences_train.shape[-1] + timestamps_train.shape[-1] + embedding_dim + 1
static_size = static_train.shape[-1] 

train_dataset = TensorDataset(time_visit_train, codes_train, sequences_train, timestamps_train, static_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(time_visit_val, codes_val, sequences_val, timestamps_val, static_val, labels_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Step 3: Define Loss Function and Optimizer
model = GRUModel_1(embedding_dim=embedding_dim, input_size=input_size, hidden_size=64, static_size=static_size, output_size=1).to(device)  
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Step 4: Train the Model

train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    model.train()
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

        outputs = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
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

            y_pred = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch).squeeze()
            val_loss = criterion(y_pred, labels_batch)
            
            total_val += labels_batch.size(0)
            correct_val += (y_pred.round() == labels_batch).sum().item()
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
plt.savefig(r'D:\cgon080\ML\Data\Features1\Time96m_retrain\loss_plot.png', dpi=300, bbox_inches='tight')
plt.close()

plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(r'D:\cgon080\ML\Data\Features1\Time96m_retrain\Acc_plot.png', dpi=300, bbox_inches='tight')
plt.close()

torch.save(model.state_dict(), r'D:\cgon080\ML\Data\Features1\RNN_96.pth')


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
        y_pred = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
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
            y_pred = model(time_visit_batch, codes_batch, sequences_batch, timestamps_batch, static_batch)
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