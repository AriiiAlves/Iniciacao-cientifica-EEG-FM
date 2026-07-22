#set page(paper: "a4", margin: 2.5cm) // Configurações globais
#set text(font: "Libertinus Serif", size: 12pt) 
#set heading(numbering: "1.1") // Isso numera as seções (1, 1.1, etc.)
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
#set bibliography(title: "Referências", style: "ieee") // Define
#show figure.where(kind: image): set figure(supplement: [Figura])

// --- Capa ---
#align(center)[
  #v(2cm) // Espaço no topo
  
  #text(size: 24pt, weight: "bold")[Anotações]
  
  #v(1cm)
  
  #text(size: 14pt, style: "italic")[Livro Deep Learning - Ian Goodfellow]
]

#pagebreak() // Quebra para começar o conteúdo na página seguinte
// --- Fim da Capa ---

#outline(title: "Sumário")

#pagebreak()

= Introdução

Anotações sobre o livro Deep Learning de Ian Goodfellow.

//#figure(
//  image("./images/diagram.png", width: 100%),
//  caption: [Diagrama do Circuito de Fonte.],
//)
// 

Estou usando o seguinte editor de PDF: `sudo dnf install evince`. É o que mais me atende até agora. É fácil, leve, e permite fazer anotações em cima dos highlights.

= Chapter 5: Machine Learning Basics

"Diz-se a um programa de computador para aprender de ume experiência _E_ com respeito a uma classe de tarefas _T_ e medição de performance _P_, se sua performance nas tarefas em _T_, como medido em _P_, melhora com experiência _E_."

== Tipos de tarefa de Machine Learning (T)

Um exemplo é uma coleção de features que foram quantitativamente medidas de um objeto/evento que queremos que o sistema de machine learning processe. Representa-se um exemplo como um vetor $x in RR^n$ onde cada entrada $x_i$ do vetor é uma feature. Por exemplo, as features de uma imagem costumam ser o valor dos pixels da imagem.

As tarefas mais comuns de machine learning incluem:

- *Classificação*: O programa precisa especificar a qual das _k_ categorias a entrada pertence. Para resolver essa tarefa, o algoritmo de aprendizado deve produzir uma função $f:RR^n arrow.r {1,...,k}$. Ex: Reconhecer objetos.

- *Classificação com entradas faltando*: O algoritmo deve aprender um conjunto de funções ao invés de uma única função. Um modo eficiente de definir isso é aprender uma distribuição de probabilidade sobre todas as variáveis relevantes. Ex: diagnóstico médico.

- *Regressão*: O programa precisa predizer um valor numérico dada uma entrada. O algoritmo deve gerar uma função $f: RR^n arrow.r RR$. Ex: predizer quanto dinheiro um assegurado irá receber.

- *Transcrição*: O programa deve observar uma representação relativamente desestruturadda de algum tipo de dado e transcrever em uma forma discreta/textual. Ex: transcrição de voz.

- *Machine Translation*: A entrada consiste em uma sequência de símbolos de uma linguagem, e o programa deve converter isto em uma sequência de símbolos em outra linguagem. Ex: traduzir Inglês para Francês.

- *Structured Output*: Tarefa onde a saída é um vetor com importantes relações entre os diferentes elementos. Ex: analisar uma frase em uma árvore que descreve sua estrutura gramatical.

- *Detecção de anomalias*: O programa examina um conjunto de objetos/eventos e sinaliza quais são incomuns/atípícos. Ex: detecção de fraudes em cartão de crédito.

- *Síntese e amostragem*: O algoritmo deve gerar novos exemplos similares aos dados de treino.

- *Imputação de valores perdidos*: O algoritmo recebe um exemplo $x in RR^n$, mas algumas entradas $x_i$ de $x$ estão faltando, e deve prever estas.

- *Remoção de ruído*: O algoritmo recebe um exemplo corrompido $bar(x) in RR^n$ obtido de um processo desconhecido de corrupção, e deve prever o exemplo limpo $x$ da sua versão corrompida $bar(x)$.

== A medição de performance (P)

Normalmente estamos interessados em quão bem o algoritmo de machine learning performa em dados que nunca foram vistos antes, já que isso sinaliza o quanto ele vai funcionar no mundo real. Para isso, *utilizamos um conjunto de dados teste separados do conjunto de treinamento*.

A escolha de uma medida de performance deve corresponder ao comportamento esperado do sistema. Isto pode ser um pouco difícil.

== A experiência (E)

Algoritmos de machine learning podem ser categorizados em supervisionados e não-supervisionados.

Algoritmos de aprendizado não-supervisionado experienciam um dataset contendo muitas features, depois aprendem propriedades úteis da estrutura desse dataset.

Algoritmos de aprendizado supervisionado experienciam um dataset contendo features, mas cada exemplo é associado com um label/target.

== Capacidade, Overfitting e underfitting

Os fatores determinantes para quão bem o algoritmo de machine learning vai performar é sua habilidade para:

1. Fazer o treinamento de erro pequeno
2. Fazer o espaço entre o erro de treinamento e teste pequeno

Esses dois fatores correspondem a dois desafios centrais de machine learning: underfitting e overfitting.

Underfitting ocorre quando o modelo não é ábil para obter um valor de erro suficientemente pequeno no conjunto de treinamento.

Overfitting ocorre quando o gap entre o erro de treinamento e de teste é muito grande.

Podemos controlar isso alterando a capacidade do modelo. Modelos com baixa capacidade podem ter dificuldade para se adaptar ao conjunto de dados do treinamento. Modelos com alta capacidade podem ter overfitting, memorizando propriedadeds do conjunto de treinamento que não servem bem no conjunto de testes.

*Os algoritmos geralmente vão performar melhor quando sua capacidade é apropriada para a complexidade de tarefa* que têm de performar e o tanto de dados que são providos.

#image("./img-notas/underfitting-overfitting.png", width: 80%)

Acima, por exemplo, no primeiro quadro usamos um modelo de regressão linear:

$ hat(y) = b + w x $

Que possui pouca capacidade para aprender os pontos (underfitting). Já o segundo modelo de regressão linear: 

$ hat(y) = b + w_1 x + w_2 x^2 $

É perfeitamente adequado. Podemos continuar com um polinômio de grau 9, como no 3° quadro, e a função gerada passa por todos os pontos, mas não possui a estrutura correta esperada (overfitting).