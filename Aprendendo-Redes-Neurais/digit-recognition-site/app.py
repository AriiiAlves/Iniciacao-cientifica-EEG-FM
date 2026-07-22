from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import io
import base64
import networkx as nx # Gera gráficos de redes neurais!

matplotlib.use('Agg') # Evita abrir janelas no servidor

app = FastAPI()

# CARREGANDO MODELO DE REDE NEURAL PRÉ-TREINADO
modelo = nn.Sequential(
        nn.Linear(64, 64),   # camada de entrada: 64 dados -> 64 neurônios
        nn.ReLU(),           # ativação não-linear
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)     # saída: 10 valores (classifica)
        # Sem função de ativação, pois é responsabilidade da Loss.
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.load_state_dict(torch.load("digit-recognition-64px.pth"))
modelo.to(device)
modelo.eval() # desliga dropout/batchnorm em modo treino

def gerar_grafico_rede(ativacoes_camadas):
    """
    ativacoes_camadas: Lista contendo os arrays de ativações de cada camada:
                       [entrada (64), camada_1 (64), camada_2 (32), saida (10)]
    """
    G = nx.DiGraph()
    pos = {}
    cores_nos = []
    
    # Para o gráfico não ficar gigante e travado com centenas de bolinhas na tela,
    # nós vamos amostrar/resumir visualmente a estrutura (ex: mostrando os 8 neurônios mais ativos de cada camada)
    # ou desenhar apenas uma versão simplificada para a visualização ficar bonita e compreensível.
    limites_camadas = [64, 64, 32, 10] # Quantos neurônios exibir por camada na imagem
    nomes_camadas = ["Entrada", "Oculta 1", "Oculta 2", "Saída"]
    
    contador_no = 0
    camadas_nos = [] # Guarda os IDs dos nós de cada camada para criar as conexões depois
    
    for idx_camada, qtd_exibir in enumerate(limites_camadas):
        valores_reais = ativacoes_camadas[idx_camada]
        nos_da_camada = []
        
        # Seleciona os neurônios (se tiver mais que o limite, pega os de maior ativação para destacar)
        indices = np.argsort(valores_reais)[::-1][:qtd_exibir] if len(valores_reais) > qtd_exibir else range(len(valores_reais))
        
        for rank, idx_neuronio in enumerate(indices):
            id_no = contador_no
            nos_da_camada.append(id_no)
            
            # Posicionamento no plano (X = camada, Y = altura do neurônio)
            pos[id_no] = (idx_camada * 2, -rank + (qtd_exibir / 2))
            
            # Adiciona o nó e define a cor dele baseada no valor de ativação real
            ativacao = valores_reais[idx_neuronio]
            G.add_node(id_no, ativacao=ativacao)
            cores_nos.append(ativacao)
            
            contador_no += 1
            
        camadas_nos.append(nos_da_camada)

    # Cria conexões (arestas) apenas entre os neurônios exibidos das camadas adjacentes
    for i in range(len(camadas_nos) - 1):
        for no_origem in camadas_nos[i]:
            for no_destino in camadas_nos[i+1]:
                G.add_edge(no_origem, no_destino)

    # Desenha o gráfico
    fig, ax = plt.subplots(figsize=(25, 20))
    
    # Desenha as conexões bem finas e transparentes para não poluir
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, edge_color="gray", arrows=False)
    
    # Desenha as bolinhas dos neurônios brilhando conforme a ativação (cmap="YlOrRd" vai de amarelo para vermelho)
    nos = nx.draw_networkx_nodes(
        G, pos, ax=ax, 
        node_color=cores_nos, 
        cmap=plt.cm.YlOrRd, 
        node_size=150, 
        edgecolors="black"
    )
    
    # Adiciona rótulos simples nas colunas
    for idx_camada, nome in enumerate(nomes_camadas):
        ax.text(idx_camada * 2, 5, nome, horizontalalignment='center', fontsize=10, fontweight='bold')

    plt.colorbar(nos, ax=ax, label="Nível de Ativação (Output)")
    plt.axis('off')
    
    # Salva na memória para enviar ao HTML
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    string_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{string_base64}"

# Monta a pasta static para que o navegador consiga baixar o style.css
app.mount("/static", StaticFiles(directory="static"), name="static")

# Rota principal que entrega o arquivo HTML estático ao acessar http://localhost:8000/
@app.get("/")
async def root_route():
    return FileResponse("static/index.html")


class DadosDesenho(BaseModel):
    pixels: List[int]

@app.post("/api/process_drawing")
async def process_route(image: DadosDesenho):
    print(image.pixels)

    # 1. Processa a matriz para o plot
    pixelsarray = np.array(image.pixels)
    matriz_8x8 = pixelsarray.reshape(8, 8)

    # 2. Gera o gráfico de forma silenciosa na memória
    plt.figure(1, figsize=(3, 3))
    plt.imshow(matriz_8x8, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.axis('off')

    # Salva o gráfico em um "arquivo temporário" na memória RAM
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close() # Fecha a figura atual para liberar memória

    # 3. Converte os bytes da imagem para uma string Base64
    string_base64 = base64.b64encode(buf.read()).decode('utf-8')
    imagem_uri = f"data:image/png;base64,{string_base64}"

    # 4. Passa os pixels para a rede neural
    tensor_input = torch.tensor([pixelsarray], dtype=torch.float32).to(device)
    # Executa a previsão sem calcular gradientes (economiza memória)
    with torch.no_grad():
        # Entrada direta
        ativacao_entrada = tensor_input.cpu().numpy()[0]
        
        # Passa pela primeira sequência (Linear + ReLU)
        x1 = modelo[0](tensor_input)
        x1_act = modelo[1](x1)
        ativacao_c1 = x1_act.cpu().numpy()[0]
        
        # Passa pela segunda sequência (Linear + ReLU)
        x2 = modelo[2](x1_act)
        x2_act = modelo[3](x2)
        ativacao_c2 = x2_act.cpu().numpy()[0]
        
        # Passa pela camada final de saída
        saida = modelo[4](x2_act)
        ativacao_saida = saida.cpu().numpy()[0]

    # Pega o índice do maior valor (a classe prevista) e converte para int comum do Python
    resultado = torch.argmax(saida, dim=1).item()

    # 3. Lista contendo a ativação de todas as camadas
    todas_ativacoes = [ativacao_entrada, ativacao_c1, ativacao_c2, ativacao_saida]

    # 4. Gera o gráfico colorido de neurônios
    grafico_neuronios_base64 = gerar_grafico_rede(todas_ativacoes)

    # Retorna o número previsto E a imagem gerada pelo Matplotlib!
    return {
        "number": resultado,
        "graphic_image": imagem_uri,
        "nn_image": grafico_neuronios_base64
    }