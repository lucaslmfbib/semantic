# Bibliometria com Semantic Scholar

App Streamlit que refaz a análise bibliométrica do projeto `bibliometria-scopus`, agora com a API da Semantic Scholar (`GET /graph/v1/paper/search/bulk`, paginação por `token`).

## Estrutura
- `semantic_scholar_bibliometria.py`: acesso à API e cálculos (indicadores, coautoria, coocorrência de palavras-chave, termos de títulos e resumos).
- `streamlit_app.py`: interface (gráficos, tabelas, exportação CSV e Excel com várias abas).
- `notebooks/semantic_scholar_bibliometria_colab.ipynb`: versão para o Colab.

## Como rodar
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # opcional: preencher S2_API_KEY
streamlit run streamlit_app.py
```
Consulta de exemplo: `"artificial intelligence" libraries`. Filtro de ano: `2020-2024`, `2023-` ou `-2019`.

## Convenções
- A chave é opcional (sem ela, o limite de requisições é menor). No Streamlit Cloud, usar `S2_API_KEY` nos Secrets; aí o campo da chave some para os visitantes.
- Lógica nova vai em `semantic_scholar_bibliometria.py`; o app só monta a interface.
- Não commitar `.env` nem `.streamlit/secrets.toml`.
- Não há testes automatizados.
