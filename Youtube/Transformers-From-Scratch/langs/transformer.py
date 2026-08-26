import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

  
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model)
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask):
        batch_size, sequence_length, d_model = x.size() # in practice, this is the same for both languages...so we can technically combine with normal attention
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask) # We don't need the mask for cross attention, removing in outer function!
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                pt_vocab_size,
                english_to_index,
                portuguese_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, portuguese_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, pt_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out

NEG_INFTY = -1e9

def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([maxSeqLen, maxSeqLen] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, maxSeqLen, maxSeqLen] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, maxSeqLen, maxSeqLen] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, maxSeqLen, maxSeqLen] , False)

    for idx in range(num_sentences):
      eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, maxSeqLen)
      kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, maxSeqLen)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

# ------------------ DEFININDO DATASET ------------------

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

enFile = "./paracrawl-release1.en-pt.zipporah0-dedup-clean.en"
ptFile = "./paracrawl-release1.en-pt.zipporah0-dedup-clean.pt"

# Criando hash maps para acessar vocabulário
enToIdx = {v:k for k,v in enumerate(enVocabulary)}
idxToEn = {k:v for k,v in enumerate(enVocabulary)}
ptToIdx = {v:k for k,v in enumerate(ptVocabulary)}
idxToPt = {k:v for k,v in enumerate(ptVocabulary)}

# Lê arquivos
with open(enFile, 'r') as file:
    enSentences = file.readlines()
with open(ptFile, 'r') as file:
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

maxSeqLen = 200

enSentences, ptSentences = sentencesTreatment(enSentences, ptSentences, maxSeqLen, 100000)

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
    max_sequence_length=maxSeqLen, 
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
        encoderSelfAttentionMask, decoderSelfAttentionMask, decoderCrossAttentionMask = create_masks(enBatch, ptBatch)
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
            print(f"English: {enBatch[0]}")
            print(f"Portuguese Translation: {ptBatch[0]}")

            # Pega previsões da primeira sequência do lote. Tensor 1D com tokens preditos para a frase.
            ptSentencePredicted = torch.argmax(ptPredictions[0], axis=1)
            predictedSentence = ""

            # Traduz tokens em letras
            for idx in ptSentencePredicted:
                if idx == ptToIdx[END_TOKEN]:
                    break
                predictedSentence += idxToPt[idx.item()]

            print(f"Portuguese Prediction: {predictedSentence}")

            # Coloca modelo no modo avaliação/inferência (dropout desativado, para de atualizar médias e variâncias no batch normalization)
            transformer.eval()

            ptSentence = ("",)
            enSentence = ("should we go to the mall?", )

            # Decodificação autoregressiva gulosa para gerar a tradução token por token
            for wordCounter in range(maxSeqLen):
                encoderSelfAttentionMask, decoderSelfAttentionMask, decoderCrossAttentionMask = create_masks(enSentence, ptSentence)

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

            print(f"Evaluation translation (should we go to the mall?) : {ptSentence}")
            print("--------------------------------")