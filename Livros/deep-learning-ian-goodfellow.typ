= 6 Deep Feedforward Networks / Multilayer Perceptrons (MLPs)

- Deve *aproximar uma função* $f^*$, de modo que $y=f(x;theta)$, onde $theta$ são parâmetros aprendíveis
- *Feedforward*, pois a informação flui diretamente de $x$ ate $y$. *Não há conexões de feedback*, exceto em RNNs (Recurrent Neural Networks)
- *Networks*, pois são a composição de muitas funções. $f(x) = f^((3))(f^((2)(f^((1)(x)))))$, onde $f^((1))$ = First Layer, e assim por diante. A largura da cadeia da a *profundidade* do modelo.
- *Neural*, pois são baseadas na neurosciência. (neurônio: input de vários neurônios para gerar um único output)
- Os dados de treinamento especificam que o último Layer deve produzir um valor próximo a $y approx f^*(x)$. O comportamento dos Layers internos não está diretamente relacionado aos dados de treinamento, por isso são chamados *hidden layers*.