= Sobre o Dataset

Link do dataset: #link("https://openneuro.org/datasets/ds004504/versions/1.0.9")[openneuro]

Este conjunto de dados contém registros de EEG em repouso com olhos fechados de 88 indivíduos no total.

Participantes:

- 36 indivíduos foram diagnosticados com doença de Alzheimer (grupo AD)
- 23 com demência frontotemporal (grupo FTD)
- 29 eram indivíduos saudáveis ​​(grupo CN). 

O estado cognitivo e neuropsicológico foi avaliado pelo Mini Exame do Estado Mental (MMSE). A pontuação no MMSE varia de 0 a 30, sendo que pontuações mais baixas indicam declínio cognitivo mais grave. A duração da doença foi medida em meses, com mediana de 25 meses e intervalo interquartil (IQR) de 24 a 28,5 meses (Q1-Q3). Não foram relatadas comorbidades relacionadas à demência no grupo DA. A pontuação média no MMSE para o grupo com AD foi de 17,75 (DP = 4,5), para o grupo com FTD foi de 22,17 (DP = 8,22) e para o grupo CN foi de 30. A idade média do grupo com DA foi de 66,4 anos (DP = 7,9), para o grupo com DFT foi de 63,6 anos (DP = 8,2) e para o grupo CN foi de 67,9 anos (DP = 5,4).

Pré-processamento: As gravações de EEG foram exportadas no formato .eeg e transformadas para o formato .set, aceito pelo BIDS, para inclusão no conjunto de dados. As anotações automáticas do dispositivo de EEG Nihon Kohden, que marcam artefatos (atividade muscular, piscadas, deglutição), não foram incluídas para fins de compatibilidade linguística (caso isso seja um problema, utilize o conjunto de dados pré-processado na pasta "derivados"). As gravações de EEG não processadas estão incluídas em pastas com o nome "sub-0XX". As pastas com o nome "sub-0XX" na subpasta "derivados" contêm as gravações de EEG pré-processadas e com ruído reduzido. O fluxo de pré-processamento dos sinais de EEG é o seguinte: primeiro, foi aplicado um filtro passa-banda de Butterworth de 0,5 a 45 Hz e os sinais foram re-referenciados para A1-A2. Em seguida, a rotina de Reconstrução do Subespaço de Artefatos (ASR), um método de correção de artefatos de EEG incluído no software EEGLab para Matlab, foi aplicada aos sinais, removendo períodos de dados ruins que excediam o desvio padrão máximo aceitável de 0,5 segundo da janela, de 17, considerado um valor conservador. Na sequência, foi realizada a Análise de Componentes Independentes (ICA) (algoritmo RunICA), transformando os 19 sinais de EEG em 19 componentes ICA. Os componentes ICA classificados como "artefatos oculares" ou "artefatos da mandíbula" pela rotina de classificação automática "ICLabel" da plataforma EEGLAB foram automaticamente rejeitados. Cabe ressaltar que, mesmo com a gravação realizada em repouso e com os olhos fechados, artefatos de movimento ocular ainda foram encontrados em alguns registros de EEG.

= Baixando Dataset

Ao tentar baixar o dataset por git clone, dá errado. Isso acontece porque esse dataset é gerenciado via *git-annex/DataLad*: os arquivos `.set` que você vê após o git clone são apenas *ponteiros/placeholders (symlinks)* — o conteúdo real (os .fdt, que guardam os dados EEG em si) fica armazenado separadamente (geralmente em S3) e precisa ser baixado à parte.

```
sudo apt install git-annex
pip install datalad
datalad clone https://github.com/OpenNeuroDatasets/ds004504.git`
cd ds004504
datalad get . # Baixa tudo
datalad get sub-001 # Baixa algo específico
```

= Visualizando Dataset

O dataset está em arquivos `.set`. É o formato padrão de dados do EEGLAB (uma das ferramentas mais populares em MATLAB para análise de eletroencefalografia). Ele guarda tanto as informações sobre o experimento quanto o sinal do cérebro em si.

Para ler arquivos `.set`: `pip install mne`.

Lendo Continuous Data:

```py
import mne

# Carrega o arquivo .set
raw = mne.io.read_raw_eeglab("seu_arquivo.set", preload=True)

# Exibe informações gerais sobre o sinal
print(raw.info)

# Plota os canais para visualização
raw.plot()
```

= Paciente sub-001

As características desse paciente são:

- Gênero: Feminino
- Idade: 57
- Grupo: A (Alzheimer Disease Group)
- MMSE: 16

E as informações sobre o sinal:

- bads: [] - Nenhum canal marcado como "ruim" ou com defeito.
- ch_names: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, ... - Nomes dos eletrodos mapeados. Seguem o sistema internacional 10-20.
- chs: 19 EEG - 19 canais do tipo EEG
- custom_ref_applied: False
- dig: 22 items (3 Cardinal, 19 EEG) - Contém as coordenadas 3D dos 19 eletrodos mais 3 pontos de referência anatômicos no crânio (geralmente Nasion, LPA e RPA — nariz e orelhas).
- highpass: 0.0 Hz
- lowpass: 250.0 Hz - Mostra os filtros físicos/digitais aplicados na gravação. O sinal contém frequências de 0 Hz até 250 Hz (limite de Nyquist para 500 Hz).
- meas_date: unspecified
- nchan: 19
- projs: []
- sfreq: 500.0 Hz - Taxa de amostragem (Sampling Frequency). O sinal foi gravado coletando 500 pontos de dados por segundo para cada canal.


O resto é mostrado pelos gráficos.

= Informações importantes

== Sistema 10-20

O sistema internacional 10-20 utiliza 21 pontos que são marcados dividindo o crânio em proporções de 10% ou 20% do comprimento das distâncias entre os pontos de referência

A nomenclatura dos pontos é dada de acordo com a região em que estão localizados, Fp = frontal polar, F = frontal, T = temporal, C = central, P = parietal e O = occipital. Os pontos localizados sobre a linha média são indexados pela letra “z”, de “zero”, os pontos localizados do lado esquerdo da linha média por índices ímpares e à direita por índices pares.

Imagem: #link("https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/21_electrodes_of_International_10-20_system_for_EEG.svg/960px-21_electrodes_of_International_10-20_system_for_EEG.svg.png")[Sistema10-20]

== Frequência de amostragem

É o número de vezes por segundo que um sinal analógico contínuo (como uma onda sonora) é medido e convertido em formato digital.