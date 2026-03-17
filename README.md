# Análise Bibliométrica com Semantic Scholar

App em Streamlit para repetir a análise bibliométrica que antes usava Scopus, agora com a API da Semantic Scholar.

## O que o app faz

- busca artigos pelo endpoint `GET /graph/v1/paper/search/bulk`
- pagina resultados usando `token`, no estilo do exemplo `search_bulk/get_dataset.py` do repositório `allenai/s2-folks`
- calcula indicadores bibliométricos simples
- mostra gráficos por ano, autores, venues, tipos de publicação e termos dos títulos
- exporta os dados tratados em CSV e Excel

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuração opcional

Se você tiver uma chave da Semantic Scholar, copie `.env.example` para `.env` e preencha:

```bash
cp .env.example .env
```

Para o deploy no Streamlit Community Cloud, a mesma chave pode ser colada como segredo:

```toml
S2_API_KEY = "sua-chave-aqui"
```

## Execução

```bash
streamlit run streamlit_app.py
```

## Consulta de exemplo

```text
"artificial intelligence" libraries
```

Você também pode usar filtros de ano no formato:

- `2020-2024`
- `2023-`
- `-2019`

## Publicação

No GitHub, publique esta pasta como um repositório próprio.

No Streamlit Community Cloud, selecione:

- repositório GitHub
- branch `main`
- arquivo de entrada `streamlit_app.py`
