= Guia IC

== Proposta

Modelos de fundação têm se tornado centrais no avanço de aplicações de aprendizado de máquina em diversos domínios, incluindo a área da saúde. No entanto, apesar de seu amplo estudo em tarefas de visão computacional nesse domínio de conhecimento, sua aplicação a sinais fisiológicos ainda é incipiente. *Com exceção do eletrocardiograma, poucos esforços foram dedicados ao desenvolvimento e à avaliação desses modelos para outros sinais*. Este projeto propõe uma *avaliação técnica e aprofundada dos modelos de fundação voltados a sinais fisiológicos, considerando critérios como custo computacional e desempenho em múltiplas tarefas*. Particularmente, esta frente irá pesquisar por modelos de fundação para eletroencefalogramas (EEGs), exame não evasivo que mede a atividade elétrica espontânea do cérebro. Espera-se, com isso, contribuir para o avanço científico nessa interseção entre inteligência artificial e saúde, além de proporcionar ao bolsista uma formação qualificada em aprendizado de máquina com foco em aplicações interdisciplinares.

== Objetivo Final

- Agregar modelos de fundação voltados para EEG
- Avaliar esses modelos de fundação, considerando critérios como custo computacional e desempenho em múltiplas tarefas

== Focar em

- Entender a arquitetura do modelo (Transformers, Encoders, Decoders, Redes Neurais, CNNs)
- Entender como rodar um modelo (PyTorch)
- Entender como avaliar o modelo (métricas de desempenho)

== Ferramentas que preciso aprender

- matplotlib
- PyTorch
- Numpy

== Cronograma

1. Estudar parte matemática de Deep Learning (NNs, CNNs, RNNs), Transformers
  - Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow (Aurélien Géron)
  - Use a 2ª parte do livro para entender como funcionam as redes profundas, a matemática por trás da retropropagação (backpropagation) e a intuição do mecanismo de Self-Attention.
  - Não perca tempo decorando a sintaxe em Keras/TensorFlow. Foque apenas na intuição teórica e nos diagramas do livro. A parte prática você fará em PyTorch na etapa seguinte.
2. Estudar métricas para avaliar Modelos de Fundação
  - Engenharia de IA: Construindo aplicações com modelos de fundação
  - Aproveitar capítulos sobre avaliação de modelos
3. Estudar como usar PyTorch, MatPlotLib, NumPy
   - Pytorch: #link("https://www.learnpytorch.io/")[learnpytorch.io]
4. Implementar uma arquitetura de Transformer simples para dados numéricos contínuos (séries temporais)
5. Expandir para papers da IC (Estado da arte) 

== Métodos de Revisão Bibliográfica

- https://www.prisma-statement.org/

= Ideias (junto com Diego)

== 1. Verificar proximidade por umap

O que será verificado será a saída do modelo (antes de classificação/regressão). Consideramos que ali está a representação rica sobre o EEG do paciente.

Em pacientes do mesmo dataset, comparar similaridade entre eles. Após detectar similaridades, agrupá-los e verificar rótulos semelhantes. Usar UMAP.

== 2. Entender o que causa mudanças no sinal puro EEG

Podemos considerar o sinal EEG Uma função:

$ f(A,B,C,D,E,...,A A,A B,...) = y " , "y in RR$

Ou seja, minha ideia é que não adianta nada tentar generalizar a detecção por meio de um foundation model sendo que temos uma função $f:R^infinity arrow.r R$ onde a saída é o sinal em si. Inúmeras ações biológicas poderiam levar ao mesmo valor de sinal.

O segredo está em identificar variações no sinal e entender o que elas significam.

Após filtrar variações enviesadas, refinamos o sinal e mandamos ao processamento no modelo.

Ideia de experimento: mesma pessoa, mesmos estímulos e experimentos, mas sob condições diferentes:

- Equipamentos diferentes
- Taxas de amostragem diferentes
- Calor/Frio
- Ambiente ruidoso/não ruidoso
- Dia/Noite
- Animada/Cansada
- Com fome/Sem fome
- ????

Rotulação MÁXIMA.

Conversar com profissional de EEG para entender melhor. (inclusive coisas sobre localização geográfica)

Fazer isso com vários indivíduos, e buscar entender o que muda o EEG. Se entendermos o que causa as variações, podemos ajustar o modelo para generalizar com muito mais precisão e facilidade.

O objetivo final é o modelo se dar bem em datasets diferentes do dataset de treino.