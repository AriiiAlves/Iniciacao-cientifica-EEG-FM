# ------------------------ Inicialização mínima ------------------------
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "rodando"}

# Para rodar o servidor: uvicorn nome-arquivo:variavel-fast-api --reload
# Aqui, poderia ser app:app

# ------------------------ Rotas (Endpoints) e Parâmetros ------------------------

## Parâmetros de Rota (Valores na URL)
@app.get("/usuarios/{usuario_id}")
def obter_usuario(usuario_id: int): # Autocompleta e valida como int
    return {"id": usuario_id}

## Parâmetros de Busca (Parâmetros opcionais na URL, como ?limite=10)
@app.get("/produtos")
def listar_produtos(limite: int = 10, busca: str = None):
    return {"limite": limite, "busca": busca}

# ------------------------ Corpo da Requisição ------------------------
from pydantic import BaseModel
from typing import List

# 1. Defina o esquema de dados esperado
class DadosDesenho(BaseModel):
    pixels: List[int]
    nome: str = "Desenho Anônimo"  # Valor padrão/opcional

# 2. Use o esquema na rota POST
@app.post("/salvar")
def salvar(dados: DadosDesenho):
    # O FastAPI já valida se o body recebido respeita o modelo
    return {"mensagem": f"Salvo com sucesso {dados.nome}!", "tamanho": len(dados.pixels)}

# ------------------------ Arquivos Estáticos e Páginas HTML ------------------------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 1. Monta a pasta onde estão seus arquivos estáticos (CSS, JS, Imagens)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Entrega seu arquivo HTML principal na raiz "/"
@app.get("/")
def pagina_principal():
    return FileResponse("static/index.html")

# ------------------------ Tratamento de Erros ------------------------
from fastapi import HTTPException

@app.get("/itens/{item_id}")
def ler_item(item_id: int):
    if item_id != 42:
        raise HTTPException(status_code=404, detail="Item não encontrado")
    return {"item": "A resposta para tudo"}

# ------------------------ CORS (frontend rodando em porta diferente) ------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite qualquer origem (substitua pelo seu domínio em produção)
    allow_credentials=True,
    allow_methods=["*"], # Permite GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],
)