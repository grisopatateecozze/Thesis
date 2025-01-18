import pandas as pd

# caricamento dataset attacchi diversi
dataset_dos = pd.read_csv('../2. Datasets/DoS_dataset.csv')
dataset_fuzzy = pd.read_csv('../2. Datasets/Fuzzy_dataset.csv')
dataset_spoofing_gear = pd.read_csv('../2. Datasets/gear_dataset.csv')
dataset4_spoofing_RPM = pd.read_csv('../2. Datasets/RPM_dataset.csv')

# Funzione per mappare le etichette di classificazione binaria in multi-classe


def map_labels(df, attack_label):
    df['Flag'] = df['Flag'].map({'R': 0, 'T': attack_label})
    return df


# nuove etichette: 0 - normal run, 1 - DOS attack, 2 - fuzzy attack, 3 - spoofing_gear attack, 4 - spoofing_RPM attack
dataset_dos = map_labels(dataset_dos, 1)
dataset_fuzzy = map_labels(dataset_fuzzy, 2)
dataset_gear1 = map_labels(dataset_spoofing_gear, 3)
dataset_gear2 = map_labels(dataset4_spoofing_RPM, 4)

# Unire i dataset
combined_dataset = pd.concat([dataset_dos, dataset_fuzzy, dataset_gear1, dataset_gear2], ignore_index=True)

# Salvare il dataset combinato
combined_dataset.to_csv('combined_dataset.csv', index=False)

print('Dataset combinato creato con successo e salvato come combined_dataset.csv')

# Caricamento del dataset da un file CSV
dataset = pd.read_csv("combined_dataset.csv")
dataset = dataset.dropna()

# esplorazione del dataset
print(dataset.info())
