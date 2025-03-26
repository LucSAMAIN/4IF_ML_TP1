import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Charger les données
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Séparer les caractéristiques et la cible
X_train = train_df.drop(['id', 'co2'], axis=1)
y_train = train_df['co2'].values
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id'].values

# Identifier les colonnes catégoriques et numériques
categorical_cols = ['brand', 'model', 'car_class', 'range', 'fuel_type', 'hybrid', 'grbx_type_ratios']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

# Prétraitement des colonnes catégoriques : mapping vers des indices
cat_mappings = {}
for col in categorical_cols:
    # Combiner train et test pour avoir un mapping cohérent
    unique_values = pd.concat([X_train[col], X_test[col]]).unique()
    cat_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
    X_train[col] = X_train[col].map(cat_mappings[col])
    X_test[col] = X_test[col].map(cat_mappings[col])

# Gérer les valeurs manquantes dans les colonnes numériques
medians = X_train[numerical_cols].median()
X_train[numerical_cols] = X_train[numerical_cols].fillna(medians)
X_test[numerical_cols] = X_test[numerical_cols].fillna(medians)

# Normaliser les colonnes numériques
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Convertir en tenseurs PyTorch
X_train_cat = torch.tensor(X_train[categorical_cols].values, dtype=torch.long)
X_train_num = torch.tensor(X_train[numerical_cols].values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_cat = torch.tensor(X_test[categorical_cols].values, dtype=torch.long)
X_test_num = torch.tensor(X_test[numerical_cols].values, dtype=torch.float32)

# Définir un Dataset personnalisé
class VehicleDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = X_cat
        self.X_num = X_num
        self.y = y

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_num[idx], self.y[idx]
        return self.X_cat[idx], self.X_num[idx]

# Créer les datasets et dataloaders
train_dataset = VehicleDataset(X_train_cat, X_train_num, y_train)
test_dataset = VehicleDataset(X_test_cat, X_test_num)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Définir le modèle avec embeddings
class CO2Predictor(nn.Module):
    def __init__(self, cat_dims, embedding_dim=10, hidden_dim=128):
        super(CO2Predictor, self).__init__()
        # Embeddings pour chaque colonne catégorique
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=embedding_dim) for dim in cat_dims
        ])
        # Calculer la taille d'entrée après embeddings et numériques
        total_embedding_dim = len(cat_dims) * embedding_dim
        input_dim = total_embedding_dim + len(numerical_cols)
        # Couches fully connected
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_cat, x_num):
        # Passer les colonnes catégoriques dans les embeddings
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        # Concaténer avec les colonnes numériques
        x = torch.cat([embedded, x_num], dim=1)
        # Passer dans les couches fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialiser le modèle
cat_dims = [len(cat_mappings[col]) for col in categorical_cols]
model = CO2Predictor(cat_dims=cat_dims)

# Définir la perte et l'optimiseur
criterion = nn.L1Loss()  # MAE
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_cat_batch, X_num_batch, y_batch in train_loader:
        X_cat_batch, X_num_batch, y_batch = X_cat_batch.to(device), X_num_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_cat_batch, X_num_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_cat_batch.size(0)
    avg_loss = total_loss / len(train_dataset)
    if (epoch + 1) % 10 == 0:
        print(f'Époque [{epoch+1}/{num_epochs}], Perte moyenne : {avg_loss:.4f}')

# Faire des prédictions sur l'ensemble de test
model.eval()
predictions = []
with torch.no_grad():
    for X_cat_batch, X_num_batch in test_loader:
        X_cat_batch, X_num_batch = X_cat_batch.to(device), X_num_batch.to(device)
        outputs = model(X_cat_batch, X_num_batch)
        predictions.extend(outputs.cpu().numpy().flatten())

# Créer le fichier de soumission
submission = pd.DataFrame({'id': test_ids, 'co2': predictions})
submission.to_csv('submission.csv', index=False)
print("Fichier de soumission 'submission.csv' créé avec succès !")