#set page(paper: "a4", margin: 2.5cm) // Configurações globais
#set text(font: "Libertinus Serif", size: 12pt) 
#set heading(numbering: "1.1") // Isso numera as seções (1, 1.1, etc.)
// Define que a numeração só aparece a partir do nível 3 (===)
#set heading(numbering: (..nums) => {
  let level = nums.pos().len()
  if level >= 3 {
    // Formata os números a partir do terceiro nível (ex: 1.1 ou apenas 1)
    numbering("1.1", ..nums.pos().slice(2))
  }
})
#show link: it => box(
  stroke: green + 1pt, // Define a borda verde de 1pt
  radius: 0pt,         // Arredonda levemente os cantos (opcional)
  inset: 0pt,          // Espaçamento entre o texto e a borda
  it                   // O conteúdo do link 
)
#show ref: it => box(
  stroke: orange + 1pt, // Define a borda verde de 1pt
  radius: 0pt,         // Arredonda levemente os cantos (opcional)
  inset: 0pt,          // Espaçamento entre o texto e a borda
  it                   // O conteúdo do link 
)
#show figure.where(kind: image): set figure(supplement: [Figura])

#show heading.where(level: 1): it => {
  block(
    fill: rgb("e6f2ff"),   // Cor azul-claro do fundo
    inset: (x: 10pt, y: 8pt), // Espaçamento interno (padding)
    radius: 4pt,           // Bordas levemente arredondadas
    width: 100%,           // Ocupa a largura total da página
    it.body
  )
}

#show heading.where(level: 2): it => {
  block(
    fill: rgb("#ffd5d3"),   // Cor azul-claro do fundo
    inset: (x: 5pt, y: 4pt), // Espaçamento interno (padding)
    radius: 4pt,           // Bordas levemente arredondadas
    width: 100%,           // Ocupa a largura total da página
    it.body
  )
}

#outline(title: "Sumário")

#pagebreak()

= Introdução

Notas pessoais sobre os artigos. A intenção é entender a teoria completa por trás de Foundation Models, para entender como implementar e avaliar.

A intenção é ser enxuto para posterior consulta rápida.

= Transformers

== Attention is All You Need

Nota rápida: Esse artigo juntou várias pesquisas para criar um modelo final.

- Attention Mechanisms - "Neural Machine Translation by Jointly Learning to Align and Translate", "Effective Approaches to Attention-based Neural Machine Translation"
- Arquitetura Encoder-Decoder (Seq2Seq) - "Sequence to Sequence Learning with Neural Networks", Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
- Conexões residuais (blocos `Add`) - "Deep Residual Learning for Image Recognition"
- Normalização de Camada - "Layer Normalization"
- Positional Embeddings - "End-To-End Memory Networks", "Convolutional Sequence to Sequence Learning"
- Multi-Head Self-Attention - "A Structured Self-Attentive Sentence Embedding"

=== Arquitetura do Modelo

A maioria dos modelos competitivos de transdução (conversão) de sequência neural possuem uma estrutura encoder-decoder.

- *Encoder* - Mapeia uma sequência de representação simbólica $(x_1,...,x_n)$ para uma sequência de representação contínua $z = (z_1,...,z_n)$, *vetores que capturam o significado e o contexto de cada input*.
- *Decoder* - Dado $z$, gera uma sequência output $(y_1,...,y_m)$ de símbolos um elemento por vez.

A cada passo o modelo é auto-regressivo, consumindo os símbolos gerados previamente como input adicional enquanto gera o próximo. 

(Ex: tradução automática - Pega palavras, caputra significado, gera output em outra língua)

O Transformer segue essa arquitetura geral usando self-attention empilhados e processamento de um único item/elemento por vez.

==== Pilhas de Encoder e Decoder

#figure(image("./Images/transformer-encoder-decoder.png", width: 60%))

*Encoder*

- Composto de uma pilha de 6 camadas idênticas com 2 Sub-Camadas cada. 
- Camadas:
  1. Mecanismo *Multi-head Self-Attention* - Múltiplos cabeçotes atuando em paralelo. Cada um aprende a focar em aspectos/relações diferentes entre as palavras. No final, os resultados são concatenados.
  2. *Position-wise Fully Connected Feed-Forward Network (FFN)* - Rede neural densa (FFN) aplicada de forma independente e idêntica a cada posição/token da sequência.
- *Output Sub-Camada*: $"LayerNorm"(x+"Sublayer"(x))$ (Add + Normaliza)
- Todas as Sub-Camadas possuem dimensionalidade $d=512$ 

O *Self-Attention decide onde olhar* (decide quais palavras são importantes) e *Position-Wise FFN processa e transforma o output em conceitos abstratos* (relações sintáticas complexas, papéis semânticos e tipos de entidades). É um *vetor enriquecido*.

*Decoder*

- Também composto de uma pilha de 6 camadas idênticas, com 3 Sub-Camadas cada
- Camadas:
  1. *Masked Multi-Head Attention* - Mecanismo Self-Attention modificado para prevenir posições de prestar atenção a posições subsequentes (posteriores). Força o modelo a ignorar o que vem depois. (se não tivesse isso, no treinamento o modelo "colaria" a resposta do futuro, ao invés de tentar aprendê-la)
  2. *Multi-Head Attention* - Atua sobre o output da pilha do Encoder
  3. *Position-wise Fully Connected Feed-Forward Network (FFN)* - Consolida a mistura (passado gerado + contexto do encoder) e faz a predição.

  Na entrada do decoder, temos o *histórico de tokens já gerados*.

*Linear*

Transforma o vetor de dimensão $d_"model"$ para um vetor de dimensão $d_"vocab"$.

*Softmax*

Gera as probabilidades de cada palavra do vocabulário ser o próximo token.

==== Attention

*Scaled Dot-Product Attention*

$ "Attention"(Q,K,V) = "softmax"((Q K^T)/sqrt(d_k))V $

- Q = Matriz Query (dimensão $d_k$)
- K = Matriz Key (dimensão $d_k$)
- V = Matriz Value (dimensão $d_v$)
- $sqrt(d_k)$ = Normalização

*Multi-Head Attention*

Permite ao modelo prestar atenção a informações de diferentes subespaços de representação em diferentes posições. Só um cabeçote de atenção inibe isso (média).

As Queries/Keys/Values são projetados linearmente $h$ vezes com diferentes projeções lineares aprendíveis. Os diferentes outputs são processados em paralelo, concatenados e projetados novamente.

$ "MultiHead"(Q,K,V) = "Concat"("head"_1,...,"head"_h) W^O $
$ "head"_i = "Attention"(Q W^Q_i, K W^K_i, V W^V_i) $

- Projeções são matrizes de parâmetros $W^Q_i in RR^(d_"model" times d_k)$, $W^K_i in RR^(d_"model" times d_k)$ , $W^V_i in RR^(d_"model" times d_v)$

No paper, $h=8$ camadas de atenção paralelas. $d_k=d_v=d_"model"/h=64$. (Assim, para um input de 512 itens, cada cabeçote vai analisar um recorte do input, com 64 itens)

===== Aplicações do Attention no Modelo

- Em camadas "encoder-decoder attention", Q (queries) vem do output do *decoder* anterior, e K/V (Key, Values) vêm do output do *encoder*. Isso permite que toda posição no decoder preste atenção a todas as posições da sequência de input. Ou seja, a representação inteira do texto de entrada (processada pelo Encoder) fica disponível como um mapa de referência (V,K). A camada anterior do decoder diz: "Estou tentando gerar a próxima palavra... O que preciso buscar no texto original?"

- O encoder contém camadas self-attention. Em uma camada, K,V,Q vêm do mesmo lugar: a saída da camada anterior. Cada posição pode prestar atenção a todas as posições na camada anterior.

- No decoder, da mesma forma. Previne-se vazamento de informação a ser adivinhada com uma máscara, definindo para $-infinity$ tudo que não se quer.

==== Position-wise Feed-Forward Networks

A FFN é aplicada a cada posição separadamente e identicamente. Consiste em duas transformações lineares com uma ativação ReLU entre elas.

$ "FFN"(x) = max(0,x W_1 + b_1)W_2 + b_2 $

(outra maneira de descrever isso é como duas convoluções de Kernel size 1).

= Foundation Models

== A Compreensive Review of Biosignal Foundation Models

=== Modelos de Fundação

==== Definição de Modelos de Fundação

*Definição*: Modelo Grande pré-treinado em dados de larga-escala para aprender representações ricas em informação que podem ser usadas para tarefas específicas. *Ao invés de modelos de deep-learning, que são específicos para uma tarefa, um modelo de fundação busca capturar representações universais entre domínios e/ou modalidades*, servindo como um modelo de propósito geral.

Um modelo de fundação aprende uma função: 

$ f_theta: cal(X) arrow.r cal(Z) $

- $cal(X)$ = Dataset
- $cal(Z)$ = Espaço Latente
- $theta$ = Parâmetros Treináveis

O objetivo de um modelo de fundação é encontrar os parâmetros que minimizam a função Loss:

$ cal(L)(f_theta (x), t(x)) $

- $t(x)$ = Objetivo Auto-Supervisionado (Ex: Masked Prediction, Contrastive Learning Prediction, Next-Step Prediction). É exatamente o que o modelo deveria prever.

==== Arquiteturas de Modelo

Transformers/Mamba são o esqueleto principal do Modelo de Fundação. Eles são precedidos/antecedidos de módulos menores.

===== Convoluções e Encoding Preliminar

Convoluções sobre a dimensão do tempo são normalmente aplicadas.

===== Transformers

Seu mecanismo-chave é o self-attention, que *permite ao modelo considerar relações entre todos os elementos/tokens da sequência de input*.

$ H = "Attention"(X Q, X K, X V) = "Softmax"((X Q(X K)^T)/sqrt(d_k)) $

- $H$ = Representação final
- $X$ = Sequência de tokens $[x_1,...,x_T]$
- $Q,K,V$ = Matrizes de projeção aprendíveis
- $d_k$ = Dimensionalidade chave (regulariza valores)

O Transformer é construído com blocos encoder/decoder empilhados.

- Blocos Encoder - Contém módulos *multi-head self-attention* e *position-wise feedfoward
networks*
- Blocos Decoder - Incorporam *cross-attention* para os outputs do encoder.

Modelos de fundação podem usar a estrutura encoder/decoder completa ou não.

- BERT style - Apenas Encoder
- GPT style - Apenas Decoder

Transformers possuem complexidade $cal(O)(T^2)$.

*Adicional*: Na última camada, o Transformer gera uma *distribuição de probabilidades sobre o vocabulário para determinar o próximo token*.

===== Vector Quantisation

Em alguns casos, modelos EEG são pré-treinados em dois estágios:

1. *Tokenizer é treinado* para gerar tokens significativos das amostras de input (que representem bem o input)
2. Tokenizer é usado para treinar Modelo de Fundação

Em todos os modelos de EEG de dois estágios, o tokenizador
contém um codebook de quantização vetorial (VQ). 

Depois que a entrada for codificada em alguma representação latente h, um
livro de códigos aprendível $V = {v_i,...,v_K}$ é usado para
buscar o código mais próximo de $h$.

===== State-Space Models (Mamba)

Alternativa aos transformers.

$ h'(t) = A h(t) + B x(t), y(t) = C h(t) $

- $x(t)$ = Input
- $h(t)$ = Estado oculto
- $y(t)$ = Output no tempo $t$
- $A,B,C$ = Parâmetros aprendíveis

Na prática, isso é discretizado.

==== Arquiteturas e Objetivos SSL de modelos conhecidos

#figure(image("./Images/model-architecture-ssl-objective.png"))

==== Objetivos de Treino Auto-Supervisionado

- *Masked Autoencoding* - Máscara aplicada aos Dados ou Espaço Latente, e o modelo é treinado para reconstruir porções mascaradas. *Reconstrução de componentes faltantes a partir de componentes visíveis*. A Loss deve medir a fidelidade da reconstrução.
- *Autoregressive Models* - Tenta *adivinhar o próximo token/segmento na sequência*. Usa a Cross-Entropy Loss.
- *Contrastive Learning* - Tenta identificar relações de similaridade entre pares positivos/negativos no espaço latente $cal(Z)$. Encoraja minimizar a distância entre pares positivos e maximizar a distância entre pares negativos. "Estas duas imagens são variações do mesmo objeto ou são objetos diferentes?"

==== Métodos de Avaliação

Modelos de Fundação são avaliados avaliando sua performance em tarefas downstream como: Classificação, Regressão, Previsão, Imputação.

O modelo pode ser adaptado com: 

- *Full-finetuning*: Todos os pesos são levemente treinados com dados downstream.
- *Linear Probing*: Modelo congelado, adiciona-se cabeça de classificação treinável

Para Previsão/Imputação, pode ser usado *Zero-shot Learning*: Modelo usado diretamente em dados nunca vistos.

Métricas de avaliação comuns:

- *Acurácia* - Proporção de acertos totais (positivos/negativos) em relação ao total de previsões.
- *Acurácia Balanceada* - Acurácia obtida em cada classe individualmente
- *AUPRC (Area Under Precision-Recall Curve)* - Mede área abaixo da curva que cruza a Precisão (verquantos dos previstos como positivos eram realmente positivos) e o Recall (quantos dos positivos reais o modelo encontrou) em diferentes limiares.
- *AUROC (Area Under the Receiver Operating Characteristic Curve)* - Mede capacidade do modelo de separar/distinguir classes. Plota a taxa de verdadeiros positivos x falsos positivos.
- *F1-Score* - Média harmônica entre Precisão e Recall
- *MSE (Mean Squared Error)* - Para tarefas de regressão