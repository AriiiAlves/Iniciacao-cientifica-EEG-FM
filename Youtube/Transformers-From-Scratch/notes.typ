#link("https://www.youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4")[Playlist]

= Aula 1 - Self Attention in Transformer Neural Networks

== RNNs

RNNs (Recurrent Neural Networks) costumavam ser o estado da arte para modelagem sequência-para-sequência. Muitas aplicações em processamento de linguagem natural.

Mas sofrem de 2 problemas.

1) São muito lentas.
2) O token `t` pode olhar apenas para o contexto `t-1` ou `t+1` separadamente, e depois concatenar. Isso tira qualidade da informação. Resolve-se isso com o mecanismo Attention.

== Attention Mechanism

Com Attention, podemos decidir em quais partes cada palavra precisa focar.

#figure(image("./images/attention.png"))

Quanto mais claro, mais atenção. (maior o número)

- My - Foca em "My" e "Name"
- "Name" - Foca muito em "Ajay"

== Transformer Overview

Vamos traduzir "My name is Ajay".

#figure(image("./images/encoder-decoder.png"))

1. Passamos todas as palavras pelo Encoder.
2. Isso vai gerar 4 vetores (um para cada palavra)
3. Na entrada do Decoder, vai o token `<start>`
4. No output, teremos "Meu"
5. Na entrada do Decoder, vai o token "Meu"
6. No output, teremos "nome"
7. Na entrada do Decoder, vai o token "nome"
8. No output, teremos "é"
9. Na entrada do Decoder, teremos "é"
10. No output, teremos "Ajay"

=== Encoder - Token Embedding + Positional Encoding

Entrada: "My Name is Ajay"

*1. Token Embedding* = Significado Semântico da palavra (palavra vira vetor)

$ e_"Name" = [0.25,-0.80, 0.15, 0.90] $

*2. Position Encoding* = Vetor de mesmo tamanho representando a posição do token na frase

$ p_1 = [0.00, 1.00, 0.50, -0.50] $

$ "Vetor de Entrada = Embedding do Token + Positional Encoding" $

$ "Vetor Final" = [0.25,0.20,0.65,0.40] $

No paper, cada vetor de entrada tem dimensão 512.

=== Encoder - Attention

- Q - O que estou procurando ($d_k$)
- K - O que posso oferecer ($d_k$)
- V - O que realmente ofereço ($d_v$)

$ "Self Attention" = "Softmax"((Q dot K^T)/sqrt(d_k) + M)V $

- $sqrt(d_k)$ minimiza a variância e estabiliza os valores.

= Aula 2 - Multi-Head Attention

1. Token Embedding + Positional Enconding gera um vetor $d_"model" = 512$.
2. Geramos 3 matrizes Q,K,V a partir desse vetor, com $d_k=d_v=512$
3. Para Q,K,V, quebramos em 8 peças de $d_i=64$. Cada uma vai para uma unidade de Self-Attention e vai gerar um vetor final.
4. Os vetores são concatenados (unidos). O resultado é um vetor com um contexto muito mais rico.

= Aula 3 - Positional Encoding

== Transformer Architecture

Primeiro, temos a arquitetura do transformer:

#figure(image("./images/transformer-architecture-1.png"))

1. Cada palavra/token é transformada em um vetor de tamanho $512$, o tamanho do vocabulário.
2. Depois, adiciona-se a cada vetor soma-se um `position encoding` de mesmo tamanho.
3. Com esse vetor final, cada palavra `w` será transformada em 3 vetores `w_q`, `w_k`, `w_v` (Query, Key, Value), através de uma projeção linear. Quando juntamos tudo, temos as matrizes `Q`, `K`, `V`.

#figure(image("./images/transformer-architecture-2.png"))

=== Nota: Projeções Lineares ($Q$, $K$, $V$)

Para obter as matrizes $Q$, $K$, $V$, uma rede neural linear foi treinada, de modo que, para uma matriz de tokens $x in RR^(n times d_"model")$:

$ Q = X W_Q + b_Q $
$ K = X W_K + b_K $
$ V = X W_V + b_V $

Os pesos $W_Q$, $W_K$, $W_V$ são os parâmetros treinados junto com o resto da rede via backpropagation.

== Positional Embedding

$ "PE"_("pos",2i) = sin("pos"/10000^((2i)/d_"model")) $
$ "PE"_("pos",2i+1) = cos("pos"/10000^((2i)/d_"model")) $

- $i$ = Componente do vetor de embedding (0-512)
- $d_"model"$ = Embedding Lenght
- $"pos"$ = Posição do token na sequência (0,1,2,3,...)

Por que essas fórmulas?

1. Periodicidade
2. Valores limitados (entre -1 e 1)
3. Fácil de extrapolar para sequências longas

Em uma matriz onde cada linha é um embedding:

- O `pos` varia na vertical.
- O `i` varia na horizontal e altera a frequência da onda (frequências altas nas primeiras dimensões, frequências baixas nas últimas)

Sem o positional embedding, as frases:

- "O cão comeu o gato"
- "O gato comeu o cão"

Produziriam o mesmo resultado, pois contém ambas as mesmas palavras. Com Positional Embedding, os vetores iniciais de cada frase serão diferentes e, portanto, serão tratados de maneiras diferentes.

= Aula 4 - Layer Normalization

== Transformer Architecture

1. Cada palavra/token é transformada em um vetor de tamanho $512$, o tamanho do vocabulário.
2. Depois, adiciona-se a cada vetor soma-se um `position encoding` de mesmo tamanho.
3. Com esse vetor final, cada palavra `w` será transformada em 3 vetores `w_q`, `w_k`, `w_v` (Query, Key, Value), através de uma projeção linear. Quando juntamos tudo, temos as matrizes `Q`, `K`, `V`.
4. Todos os vetores `w_q`, `w_k`, `w_v` são divididos em 8 partes ($512/8 = 64$). Cada parte será um vetor para um bloco de atenção (são 8 blocos)
5. Os vetores serão concatenados, em ordem. A saída terá tamanho $512$ novamente.
6. Esse vetor enriquecido será somado com o vetor vocabulário original (com PE), e normalizado. Essa soma é para evitar o vanishing gradiente (gradientes de valores pequenos tendem a zero, parando o aprendizado)

== Layer Normalization

$ "LayerNorm"[x' + "out"]$

(Add and Norm)

- $x'$ = Vetor Original + Positional Encoding
- $"out"$ = Saída concatenada dos blocos de atenção

== Layer Normalization: What?

A normalização encapsula os valores em um intervalo bem menor do que o original, com centro tipicamente em torno de zero. Isso permite um treino muito mais estável, com passos mais uniformes durante o backpropagation.

Suponha uma rede neural.

#figure(image("./images/layer-norm.png", width: 40%))

$ x' = f(W^T_1 x + b_1) $

A normalização para a próxima camada será:

$ y = gamma_1 [(x'-mu_1)/sigma_1] + beta_1 $

- $mu_1$ = Média dos valores de ativação da camada
- $sigma_1$ = Standard Deviation dos valores de ativação da camada
- $gamma_1$ (escala),$beta_1$ (deslocamento) = Parâmetros aprendíveis do próprio LayerNorm. O mesmo para toda a camada.

A normalização pura força os dados a terem estritamente média 0 e variância 1. No entanto, essa distribuição fixa nem sempre é a ideal para a camada seguinte aprender.

Os parâmetros $gamma$ e $beta$ dão à rede a liberdade de desfazer ou ajustar a normalização se necessário:

- Se a rede aprender que $gamma = 1$ e $beta = 0$, ela mantém a normalização perfeita ($mu = 0, sigma^2 = 1$).
- Se a rede aprender que $gamma = sigma_"original"$ e $beta = mu_"original"$, ela pode restaurar os dados exatamente ao seu estado original.

== Layer Normalization: How?

$ x' = mat(0.2,0.1,0.3;0.5,0.1,0.1) $

$ mu_(11) = 1/3[0.2+0.1+0.3] = 0.2 $
$ mu_(21) = 1/3[0.5+0.1+0.1] = 0.233 $
$ sigma_(11) = sqrt(1/3{[0.2-0.2]^2+[0.1-0.2]^2+[0.3-0.2]^2}) = 0.08164 $
$ sigma_(21) = sqrt(1/3{[0.5-0.233]^2+[0.1-0.233]^2+[0.1-0.233]^2}) = 0.1885 $
$ mu = mat(mu_(11);mu_(21)) = mat(0.2;0.233) $
$ sigma = mat(sigma_(11) + sigma(21)) = mat(0.08164;0.1885) $
$ y = (x'-mu)/sigma = mat(0, -1.2248, 1.2248; 1.414, -0.707, -0.707) $
$ "out" = gamma dot y + beta $

= Aula 5 - Blowing up the Transformer Encoder!

Obs: Linhas = max_sequence_lenght, Colunas = d_k

1. Input Sentence

TRATAMENTO DE INPUT

2. *Embedding Transform* -> Transformar palavras/tokens em vetores de dimensão $d=512$ (vocabulário = $2^(512)$ possibilidades de representação)
3. *Positional Encoding* -> Adiciona-se ao embedding valores periódicos que informam a ordem/posição das palavras na sequência 

$ "PE"_("pos",2i) = sin("pos"/10000^((2i)/d_"model")) $
$ "PE"_("pos",2i+1) = cos("pos"/10000^((2i)/d_"model")) $

ENCODER

4. *Generate Q,K,V* -> Cada vetor sofre uma *projeção linear* para gerar os vetores $w_q$,$w_k$,$w_v$. Juntando todos os vetores em uma matriz:

$ Q = X W_Q + b_Q $
$ K = X W_K + b_K $
$ V = X W_V + b_V $

(onde $W_Q$,$W_K$, $W_V$ são matrizes aprendíveis que serão ajustadas pelo backpropagation)

5. *Multi-Head* -> Cada vetor será quebrado em 8 blocos ($d=64$), para processamento paralelo e mais enriquecido em contexto.
6. *Multi-Head Attention* -> Aplica Attention a cada uma das Heads. (A 1° linha de Q vezes a 2° linha de K tranposta, etc)

$ "Self Attention" = "Softmax"((Q dot K^T)/sqrt(d_k) + M)V $

7. *Concatenate* -> Os 8 blocos são concatenados novamente
8. *Add* -> Resultado da Concatenação + (Input Embedding + Positional Encoding). Soma-se o Attention ao Input inicial, evitando Vanishing Gradient (valores pequenos demais).
9. *Layer Normalization* -> Estabiliza os valores, os deixando em torno de zero. Para cada `LayerNorm()`, existe um único tensor $gamma$ e $beta$.

$ y = gamma [(x-mu)/sigma] + beta $

(onde $gamma$,$beta$ são parâmetros aprendíveis)

10. *Linear + ReLU + Dropout* -> Linear é uma rede neural de camada única. Dropout é desligar neurônios aleatoriamente, zerando uma porcentagem das saídas deles, o que evita vício em padrões e permite generalizar. Entrada: $"max_seq" times 512$. Saída: $"max_seq" times 1024$

$ y = "ReLU"(W_1 x + b_1) $

11. *Linear* -> Outra rede neural de camada única, para comprimir de volta para $512$. Entrada: $"max_seq" times 1024$. Saída: $"max_seq" times 512$.

$ y = W_2 x + b_2 $

12. *Add* -> Resultado + Input da Rede neural
13. *Layer Normalization* -> Estabiliza os valores.

$ y = gamma [(x-mu)/sigma] + beta $

14. Os vetores finais são uma representação muito mais rica em contexto do que os vetores iniciais.
15. Fazemos tudo de novo (por ex, 12 vezes), para enriquecer ainda mais os vetores.

DECODER


= Nota Minha: Transformers é apenas sobre backpropagation.

O que faz o Transformer aprender é o *backpropagation*. Essa é a única coisa certeira disso tudo: *gradiente descendente em direção a mínimos locais para minimizar a função custo*.

O resto são técnicas para tentar melhorar esse aprendizado, "conceitos" que queremos que a rede tente aprender.

- O positional encoding faz permutações de uma mesma frase terem números diferentes.
- As matrizes Q,K,V não são nada no começo. Mas a lógica é uma $"softmax"(Q,K)V$ obriga Q e K a se comportarem como "query-key" e V a se comportar como algo a ser enriquecido por isso de "query-key". Isso de "prestar atenção" é mais algo intuitivo do que certeiro.
- Multi-Head é dividir em blocos menores para um processamento mais localizado, melhorando a qualidade da análise
- Add + LayerNorm é uma técnica comum para facilitar o aprendizado (melhora efeito do gradiente)
- No LayerNorm, $gamma,beta$ são otimizados para cumprir com a função custo. Dá-se a eles sentido de "escalar" e "deslocamento", mas é só uma intuição. Apenas estamos dando mais possibilidades de aprendizado pro backpropagation.

Isso se chama *Viés Indutivo* (Inductive Bias), suposições/facilidadeds que embutimos na arquitetura para forçar ou guiar a rede a aprender determinados padrões em vez de tentar aprender tudo do zero absoluto. Todas essas camadas, somas residuais, encodings e normalizações são o *nosso jeito de encurtar o caminho do gradiente* e viabilizar a otimização em tempo útil.

Dizemos à rede: "A estrutura do seu cálculo será matrizes $Q, K, V$ passando por um softmax". O Backpropagation encontra o conteúdo: O otimizador preenche esse molde com os valores exatos de parâmetros que resolvem o problema.