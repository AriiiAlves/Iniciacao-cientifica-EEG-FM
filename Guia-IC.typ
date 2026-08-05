= Guia IC

A proposta da IC é:

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

== Ideias do que fazer

Vou tentar aprender a teoria de Transformers, implementar um modelo de Transformer na mão, e entender quais métricas posso usar para avaliar o que foi aprendido.

Posso fazer um aplicativo que acompanha passo a passo o funcionamento do transformer e vai mostrando as métricas que podem ser estudadas.

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

== Bons materiais bibliográficos

- Mãos à obra: aprendizado de máquina com Scikit-Learn, Keras & TensorFlow (Livro Prático)
- Deep Learning - Ian GoodFellow (Livro Teórico)
- Build a Large Language Model (from Scratch) - Sebastian Raschka (Livro Teórico + Prático)