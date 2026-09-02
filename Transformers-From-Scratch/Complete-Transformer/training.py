import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer, create_masks



# ------------------ DEFININDO DATASET ------------------
EN_FILE = "./Datasets/paracrawl-release1.en-pt.zipporah0-dedup-clean.en" # Dataset (Cada linha é uma frase em inglês)
PT_FILE = "./Datasets/paracrawl-release1.en-pt.zipporah0-dedup-clean.pt" # Dataset (Cada linha é a tradução para o português)

MAX_SEQ_LEN = 200 # Sequência máxima de tokens nas frases (letras/símbolos)

START_TOKEN = "<START>"
PADDING_TOKEN = "<PADDING>"
END_TOKEN = "<END>"

enVocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', '<', '=', '>', '?', '@', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
    'Y', 'Z',
    '[', '\\', ']', '^', '_', '`', 
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
    'y', 'z', 
    '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

ptVocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', '<', '=', '>', '?', '@', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
    'Y', 'Z',
    'Ã', 'Á', 'À', 'Â', 'É', 'Ê', 'Í', 'Ó', 'Ô', 'Õ', 'Ú', 'Ç',
    '[', '\\', ']', '^', '_', '`', 
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
    'y', 'z',
    'ã', 'á', 'à', 'â', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú', 'ç',
    '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

# Criando hash maps para acessar vocabulário
enToIdx = {v:k for k,v in enumerate(enVocabulary)}
idxToEn = {k:v for k,v in enumerate(enVocabulary)}
ptToIdx = {v:k for k,v in enumerate(ptVocabulary)}
idxToPt = {k:v for k,v in enumerate(ptVocabulary)}

# Lê arquivos
with open(EN_FILE, 'r') as file:
    enSentences = file.readlines()
with open(PT_FILE, 'r') as file:
    ptSentences = file.readlines()

# Iremos filtrar as sentenças válidas (tratar dataset)
def sentencesTreatment(enSentences, ptSentences, max_seq_len, TOTAL_SENTENCES=100000):
    # Se não tiver tokens válidos, tira
    def is_valid_tokens(sentence, vocab):
        for token in sentence:
            if token not in vocab:
                return False
        return True

    # Se tamanho for maior do que o definido, tira
    def is_valid_length(sentence, max_seq_len):
        return len(list(sentence)) < (max_seq_len - 1)
    
    # Limitar número de sentenças
    enSentences = enSentences[:TOTAL_SENTENCES] # Limita
    ptSentences = ptSentences[:TOTAL_SENTENCES] # Limita
    enSentences = [sentence.rstrip('\n') for sentence in enSentences] # Tira '\n'
    ptSentences = [sentence.rstrip('\n') for sentence in ptSentences] # Tira '\n'

    # Índices das sentenças válidas
    valid_sentence_idx = []

    for i in range(len(ptSentences)):
        ptSentence, enSentence = ptSentences[i], enSentences[i]
        if is_valid_length(ptSentence, max_seq_len) \
            and is_valid_length(enSentence, max_seq_len) \
            and is_valid_tokens(ptSentence, ptVocabulary) \
            and is_valid_tokens(enSentence, enVocabulary):
                valid_sentence_idx.append(i)

    enFilteredSentences = [enSentences[i] for i in valid_sentence_idx]
    ptFilteredSentences = [ptSentences[i] for i in valid_sentence_idx]

    return enFilteredSentences, ptFilteredSentences

enSentences, ptSentences = sentencesTreatment(enSentences, ptSentences, MAX_SEQ_LEN, 100000)

# Cria objeto Dataset para utilizar DataLoader do PyTorch
class TextDataset(Dataset):
    def __init__(self, enSentences, ptSentences):
        self.enSentences = enSentences
        self.ptSentences = ptSentences

    # len(dataset) retorna isso aqui
    def __len__(self):
        return len(self.enSentences)

    # dataset[i] vai retornar isso aqui
    def __getitem__(self, idx):
        return self.enSentences[idx], self.ptSentences[idx]

dataset = TextDataset(enSentences, ptSentences)

batch_size = 3
train_loader = DataLoader(dataset, batch_size) # Empacota dataset para iterar sobre ele

# ------------------ DEFINIR MODELO ------------------

transformer = Transformer(
    d_model=512, 
    ffn_hidden=2048, 
    num_heads=8, 
    drop_prob=0.1, 
    num_layers=1, 
    max_sequence_length=MAX_SEQ_LEN, 
    pt_vocab_size=len(ptVocabulary),
    english_to_index=enToIdx,
    portuguese_to_index=ptToIdx,
    START_TOKEN=START_TOKEN, 
    END_TOKEN=END_TOKEN, 
    PADDING_TOKEN=PADDING_TOKEN)

print(transformer) # Imprime definição do modelo

# CrossEntropyLoss -> Função de perda
#   ignore_index -> Cálculo da Loss e gradientes ignora posições com <PADDING>
#   reduction -> Em vez agregar o resultado em um único número escalar, retorna a perda individual de cada token no mesmo formato do tensor de entrada

criterian = nn.CrossEntropyLoss(ignore_index=ptToIdx[PADDING_TOKEN], reduction='none')

# Itera sobre todos os tensores de parâmetros treináveis
for params in transformer.parameters():
    # Filtra apenas tensores multidimensionais
    if params.dim() > 1:
        # Preenche pesos com valores amostrados de uma distribuição uniforme baseada no número de entradas e saídas da camada, mantendo a variância dos gradientes estável e evitando explosão/desaparecimento de gradientes no início do treino
        nn.init.xavier_uniform_(params)

# Instancia otimizador Adam para atualizar os pesos do modelo no treinamento
# Define taxa de aprendizado inicial como 1e-4
optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)

# Define dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

# ------------------ TREINAMENTO ------------------

transformer.train() # Coloca no modo treinamento, ativando Dropout e Batch/Layer Norm
transformer.to(device)
total_loss = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")

    # Iteramos sobre o dataset tratado
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        enBatch, ptBatch = batch
        encoderSelfAttentionMask, decoderSelfAttentionMask, decoderCrossAttentionMask = create_masks(enBatch, ptBatch, MAX_SEQ_LEN)
        optim.zero_grad() # Zera o gradiente para cada batch

        # Faz transformer processar as frases do batch e gerar predições
        ptPredictions = transformer(
            enBatch,
            ptBatch,
            encoderSelfAttentionMask.to(device),
            decoderSelfAttentionMask.to(device),
            decoderCrossAttentionMask.to(device),
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=True
        )

        # Transforma Labels em português em vetor numérico
        labels = transformer.decoder.sentence_embedding.batch_tokenize(ptBatch, start_token=False, end_token=True)

        # Loss compara predição e label
        loss = criterian(
            ptPredictions.view(-1, len(ptVocabulary)).to(device),
            labels.view(-1).to(device)
        ).to(device)

        # Se for <PADDING>, retorna False. Se não, retorna True.
        validIdx = torch.where(labels.view(-1) == ptToIdx[PADDING_TOKEN], False, True)

        # Soma losses de todos os elementos, dividido pelo n de elementos (loss de <PADDING> zerada por ignore_index)
        loss = loss.sum() / validIdx.sum()

        # Faz backpropagation, calculando gradientes
        loss.backward()

        # Executa passo do otimizador, atualizando pesos com base nos gradientes
        optim.step()

        if batch_num % 100 == 0:
            print(f"Iteration: {batch_num}: {loss.item()}")
            print(f"\nEnglish: {enBatch[0]}")
            print(f"\nPortuguese Translation: {ptBatch[0]}")

            # Pega previsões da primeira sequência do lote. Tensor 1D com tokens preditos para a frase.
            ptSentencePredicted = torch.argmax(ptPredictions[0], axis=1)
            predictedSentence = ""

            # Traduz tokens em letras
            for idx in ptSentencePredicted:
                if idx == ptToIdx[END_TOKEN]:
                    break
                predictedSentence += idxToPt[idx.item()]

            print(f"\nPortuguese Prediction: {predictedSentence}")

            # Coloca modelo no modo avaliação/inferência (dropout desativado, para de atualizar médias e variâncias no batch normalization)
            transformer.eval()

            ptSentence = ("",)
            enSentence = ("should we go to the mall?", )

            # Decodificação autoregressiva gulosa para gerar a tradução token por token
            for wordCounter in range(MAX_SEQ_LEN):
                encoderSelfAttentionMask, decoderSelfAttentionMask, decoderCrossAttentionMask = create_masks(enSentence, ptSentence, MAX_SEQ_LEN)

                predictions = transformer(enSentence,
                                        ptSentence,
                                        encoderSelfAttentionMask.to(device),
                                        decoderSelfAttentionMask.to(device),
                                        decoderCrossAttentionMask.to(device),
                                        enc_start_token=False,
                                        enc_end_token=False,
                                        dec_start_token=True,
                                        dec_end_token=False)
                
                nextTokenProbDistribution = predictions[0][wordCounter]
                nextTokenIdx = torch.argmax(nextTokenProbDistribution).item()
                nextToken = idxToPt[nextTokenIdx]
                ptSentence = (ptSentence[0] + nextToken, )
                if nextToken == END_TOKEN:
                    break

            print(f"\nEvaluation translation (should we go to the mall?) : {ptSentence}")
            print("--------------------------------")