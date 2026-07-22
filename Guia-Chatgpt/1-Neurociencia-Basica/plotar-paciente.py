import json
import pandas as pd
import mne
import matplotlib.pyplot as plt

# Lendo o arquivo
with open("./ds004504/participants.json", "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)

# Lê o arquivo TSV direto para um DataFrame
df = pd.read_csv("./ds004504/participants.tsv", sep="\t")

# Acessando as informações
print("participants.json:", end="\n\n")
print(json.dumps(dados, indent=4, ensure_ascii=False), end="\n\n")
print("participants.tsv (head):", end="\n\n")
print(df.head())

# Acessando dataset do primeiro paciente
raw = mne.io.read_raw_eeglab("./ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set")

# Informações Gerais
print(raw.info)

# Plota os canais
raw.plot()
plt.show()
# Calcula e plota o espectro de frequências de todos os canais
raw.compute_psd(fmax=50).plot()
plt.show()
# Plota a posição dos eletrodos
raw.plot_sensors(show_names=True)
plt.show()