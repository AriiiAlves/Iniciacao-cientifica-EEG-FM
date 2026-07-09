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

