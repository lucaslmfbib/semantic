#!/usr/bin/env python3
"""App Streamlit para busca e análise bibliométrica com a API da Semantic Scholar."""

from __future__ import annotations

import io
import os
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

SEMANTIC_SCHOLAR_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
DEFAULT_QUERY = '"artificial intelligence" libraries'
DEFAULT_FIELDS = ",".join(
    [
        "title",
        "year",
        "publicationDate",
        "citationCount",
        "influentialCitationCount",
        "referenceCount",
        "venue",
        "publicationTypes",
        "fieldsOfStudy",
        "authors",
        "openAccessPdf",
        "url",
        "externalIds",
    ]
)
STOPWORDS = {
    "a",
    "about",
    "and",
    "artificial",
    "as",
    "at",
    "biblioteca",
    "bibliotecas",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "education",
    "em",
    "for",
    "in",
    "intelligence",
    "library",
    "libraries",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "of",
    "on",
    "or",
    "para",
    "por",
    "the",
    "to",
    "um",
    "uma",
}


@dataclass
class SemanticScholarResult:
    papers: list[dict[str, Any]]
    total_results: int
    retrieved_results: int


def build_headers(api_key: str) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key.strip():
        headers["x-api-key"] = api_key.strip()
    return headers


def build_session() -> Session:
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    session = Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def safe_join(values: Iterable[Any]) -> str:
    cleaned = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return "; ".join(cleaned)


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_unique_parts(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass
    parts = []
    for part in str(value).split(";"):
        cleaned = part.strip()
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    return parts


def split_multivalue_column(series: pd.Series) -> pd.Series:
    values: list[str] = []
    for item in series.dropna().astype(str):
        values.extend(extract_unique_parts(item))
    return pd.Series(values, dtype="string")


def top_words(texts: pd.Series, n: int = 20) -> pd.DataFrame:
    tokens: Counter[str] = Counter()
    for text in texts.dropna().astype(str):
        for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", text.lower()):
            if token not in STOPWORDS:
                tokens[token] += 1
    return pd.DataFrame(tokens.most_common(n), columns=["termo", "frequencia"])


def fetch_bulk_page(
    session: Session,
    api_key: str,
    query: str,
    fields: str,
    page_size: int,
    year_filter: str,
    token: str | None = None,
    timeout: int = 45,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "query": query,
        "fields": fields,
        "limit": page_size,
    }
    if year_filter.strip():
        params["year"] = year_filter.strip()
    if token:
        params["token"] = token

    response = session.get(
        SEMANTIC_SCHOLAR_BULK_URL,
        headers=build_headers(api_key),
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_all_results(
    api_key: str,
    query: str,
    year_filter: str,
    page_size: int,
    max_results: int,
) -> SemanticScholarResult:
    session = build_session()
    papers: list[dict[str, Any]] = []
    token: str | None = None
    total_results = 0

    while len(papers) < max_results:
        payload = fetch_bulk_page(
            session=session,
            api_key=api_key,
            query=query,
            fields=DEFAULT_FIELDS,
            page_size=page_size,
            year_filter=year_filter,
            token=token,
        )
        if not total_results:
            total_results = safe_int(payload.get("total"))

        page_papers = payload.get("data") or []
        if not page_papers:
            break

        remaining = max_results - len(papers)
        papers.extend(page_papers[:remaining])
        if len(papers) >= max_results:
            break

        token = payload.get("token")
        if not token:
            break

    deduped: dict[str, dict[str, Any]] = {}
    for idx, paper in enumerate(papers):
        dedupe_key = str(paper.get("paperId") or f"row-{idx}")
        deduped[dedupe_key] = paper

    final_papers = list(deduped.values())[:max_results]
    return SemanticScholarResult(
        papers=final_papers,
        total_results=total_results,
        retrieved_results=len(final_papers),
    )


def papers_to_dataframe(papers: list[dict[str, Any]]) -> pd.DataFrame:
    if not papers:
        return pd.DataFrame()

    rows = []
    for paper in papers:
        authors = paper.get("authors") or []
        external_ids = paper.get("externalIds") or {}
        open_access_pdf = paper.get("openAccessPdf") or {}
        publication_types = paper.get("publicationTypes") or []
        fields_of_study = paper.get("fieldsOfStudy") or []

        rows.append(
            {
                "paper_id": paper.get("paperId"),
                "titulo": paper.get("title"),
                "primeiro_autor": authors[0].get("name") if authors else None,
                "autores": safe_join(author.get("name") for author in authors),
                "ids_autores": safe_join(author.get("authorId") for author in authors),
                "quantidade_autores": len(authors),
                "data_publicacao": paper.get("publicationDate"),
                "ano": paper.get("year"),
                "periodico_venue": paper.get("venue"),
                "tipos_publicacao": safe_join(publication_types),
                "areas_conhecimento": safe_join(fields_of_study),
                "citacoes": safe_int(paper.get("citationCount")),
                "citacoes_influentes": safe_int(paper.get("influentialCitationCount")),
                "referencias": safe_int(paper.get("referenceCount")),
                "doi": external_ids.get("DOI"),
                "corpus_id": external_ids.get("CorpusId"),
                "url": paper.get("url"),
                "status_acesso_aberto": open_access_pdf.get("status"),
                "licenca_acesso_aberto": open_access_pdf.get("license"),
                "pdf_acesso_aberto": open_access_pdf.get("url"),
            }
        )

    df = pd.DataFrame(rows)
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    if "data_publicacao" in df.columns:
        df["data_publicacao"] = pd.to_datetime(df["data_publicacao"], errors="coerce").dt.date.astype("string")
    if "pdf_acesso_aberto" in df.columns:
        df["tem_pdf_aberto"] = df["pdf_acesso_aberto"].fillna("").astype(str).str.strip().ne("")
    if "status_acesso_aberto" in df.columns:
        df["status_acesso_aberto"] = df["status_acesso_aberto"].fillna("UNKNOWN")

    sort_columns = [col for col in ["citacoes", "ano"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
    return df.reset_index(drop=True)


def filter_dataframe(
    df: pd.DataFrame,
    min_citations: int,
    only_open_access: bool,
    publication_types: list[str],
) -> pd.DataFrame:
    filtered = df.copy()
    if "citacoes" in filtered.columns and min_citations > 0:
        filtered = filtered[filtered["citacoes"] >= min_citations]
    if only_open_access:
        filtered = filtered[filtered["tem_pdf_aberto"] | filtered["status_acesso_aberto"].eq("OPEN")]
    if publication_types:
        selected = set(publication_types)
        filtered = filtered[
            filtered["tipos_publicacao"].fillna("").apply(
                lambda value: bool(selected.intersection({part.strip() for part in str(value).split(";") if part.strip()}))
            )
        ]
    return filtered.reset_index(drop=True)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    year_summary = build_year_summary(df)
    author_summary = build_entity_summary(df, "autores", "autor", top_n=50)
    venue_summary = build_entity_summary(df, "periodico_venue", "venue", top_n=50)
    field_summary = build_entity_summary(df, "areas_conhecimento", "area", top_n=50)
    edge_df, _ = build_coauthorship_network(df, max_nodes=50, max_edges=80)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="documentos", index=False)
        year_summary.to_excel(writer, sheet_name="anos", index=False)
        author_summary.to_excel(writer, sheet_name="autores", index=False)
        venue_summary.to_excel(writer, sheet_name="venues", index=False)
        field_summary.to_excel(writer, sheet_name="areas", index=False)
        edge_df.to_excel(writer, sheet_name="coautoria", index=False)
    return buffer.getvalue()


def build_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ano" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.dropna(subset=["ano"])
        .groupby("ano", as_index=False)
        .agg(
            documentos=("paper_id", "nunique"),
            citacoes=("citacoes", "sum"),
            citacoes_influentes=("citacoes_influentes", "sum"),
            referencias=("referencias", "sum"),
            pdf_aberto=("tem_pdf_aberto", "sum"),
        )
        .sort_values("ano", ascending=True)
    )
    if not summary.empty:
        summary["citacoes_medias"] = (summary["citacoes"] / summary["documentos"]).round(2)
        summary["taxa_pdf_aberto"] = ((summary["pdf_aberto"] / summary["documentos"]) * 100).round(1)
    return summary


def build_entity_summary(df: pd.DataFrame, column_name: str, label: str, top_n: int = 20) -> pd.DataFrame:
    if df.empty or column_name not in df.columns:
        return pd.DataFrame()

    rows = []
    for row in df.itertuples(index=False):
        for entity in extract_unique_parts(getattr(row, column_name)):
            rows.append(
                {
                    label: entity,
                    "paper_id": getattr(row, "paper_id"),
                    "citacoes": safe_int(getattr(row, "citacoes")),
                    "citacoes_influentes": safe_int(getattr(row, "citacoes_influentes")),
                    "tem_pdf_aberto": bool(getattr(row, "tem_pdf_aberto")),
                }
            )

    if not rows:
        return pd.DataFrame()

    exploded = pd.DataFrame(rows)
    summary = (
        exploded.groupby(label, as_index=False)
        .agg(
            documentos=("paper_id", "nunique"),
            citacoes=("citacoes", "sum"),
            citacoes_influentes=("citacoes_influentes", "sum"),
            pdf_aberto=("tem_pdf_aberto", "sum"),
        )
        .sort_values(["documentos", "citacoes"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    summary["citacoes_medias"] = (summary["citacoes"] / summary["documentos"]).round(2)
    summary["taxa_pdf_aberto"] = ((summary["pdf_aberto"] / summary["documentos"]) * 100).round(1)
    return summary


def build_open_access_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status_acesso_aberto" not in df.columns:
        return pd.DataFrame()
    summary = (
        df.groupby("status_acesso_aberto", as_index=False)
        .agg(
            documentos=("paper_id", "nunique"),
            citacoes=("citacoes", "sum"),
        )
        .sort_values(["documentos", "citacoes"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def build_coauthorship_network(
    df: pd.DataFrame,
    max_nodes: int = 30,
    max_edges: int = 40,
    max_authors_per_paper: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "autores" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    edge_weights: Counter[tuple[str, str]] = Counter()
    node_docs: Counter[str] = Counter()
    node_citations: Counter[str] = Counter()

    for row in df.itertuples(index=False):
        authors = extract_unique_parts(getattr(row, "autores"))[:max_authors_per_paper]
        citations = safe_int(getattr(row, "citacoes"))
        for author in authors:
            node_docs[author] += 1
            node_citations[author] += citations
        for source, target in combinations(authors, 2):
            edge_weights[tuple(sorted((source, target)))] += 1

    node_df = pd.DataFrame(
        [
            {"autor": author, "documentos": docs, "citacoes": node_citations[author]}
            for author, docs in node_docs.items()
        ]
    )
    if node_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    node_df = node_df.sort_values(["documentos", "citacoes"], ascending=[False, False]).head(max_nodes)
    allowed_authors = set(node_df["autor"])

    edge_df = pd.DataFrame(
        [
            {"source": source, "target": target, "peso": weight}
            for (source, target), weight in edge_weights.items()
            if source in allowed_authors and target in allowed_authors
        ]
    )
    if edge_df.empty:
        return edge_df, node_df.reset_index(drop=True)

    edge_df = edge_df.sort_values(["peso", "source", "target"], ascending=[False, True, True]).head(max_edges)
    connected_authors = set(edge_df["source"]).union(edge_df["target"])
    node_df = node_df[node_df["autor"].isin(connected_authors)].reset_index(drop=True)
    return edge_df.reset_index(drop=True), node_df


def build_coauthorship_dot(edge_df: pd.DataFrame, node_df: pd.DataFrame) -> str:
    if edge_df.empty or node_df.empty:
        return ""

    node_docs = {row["autor"]: safe_int(row["documentos"]) for _, row in node_df.iterrows()}
    node_citations = {row["autor"]: safe_int(row["citacoes"]) for _, row in node_df.iterrows()}
    max_docs = max(node_docs.values()) if node_docs else 1
    max_weight = max(edge_df["peso"]) if not edge_df.empty else 1

    lines = [
        "graph G {",
        'graph [layout="sfdp", overlap=false, splines=true, bgcolor="transparent"];',
        'node [shape=circle, style="filled", fillcolor="#E8F1FB", color="#1D4E89", fontname="Helvetica"];',
        'edge [color="#8FB3D9", fontname="Helvetica"];',
    ]

    for author in sorted(node_docs):
        docs = node_docs[author]
        citations = node_citations.get(author, 0)
        size = 0.8 + (docs / max_docs) * 1.6
        fontsize = 10 + int((docs / max_docs) * 8)
        label = f"{author}\\n{docs} docs | {citations} cit."
        lines.append(
            f'"{author}" [width={size:.2f}, height={size:.2f}, fontsize={fontsize}, label="{label}"];'
        )

    for _, row in edge_df.iterrows():
        weight = safe_int(row["peso"])
        penwidth = 1.0 + (weight / max_weight) * 4.0
        lines.append(
            f'"{row["source"]}" -- "{row["target"]}" [label="{weight}", penwidth={penwidth:.2f}];'
        )

    lines.append("}")
    return "\n".join(lines)


def show_metric_block(df: pd.DataFrame) -> None:
    total_docs = len(df)
    total_citations = int(df["citacoes"].sum()) if "citacoes" in df.columns else 0
    influential_citations = int(df["citacoes_influentes"].sum()) if "citacoes_influentes" in df.columns else 0
    open_access_docs = int(df["tem_pdf_aberto"].sum()) if "tem_pdf_aberto" in df.columns else 0
    earliest_year = int(df["ano"].dropna().min()) if "ano" in df.columns and df["ano"].notna().any() else None
    latest_year = int(df["ano"].dropna().max()) if "ano" in df.columns and df["ano"].notna().any() else None

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Documentos", f"{total_docs}")
    col2.metric("Citações", f"{total_citations}")
    col3.metric("Citações influentes", f"{influential_citations}")
    col4.metric("PDF aberto", f"{open_access_docs}")
    col5.metric("Período", f"{earliest_year or '-'} a {latest_year or '-'}")


def show_charts(df: pd.DataFrame) -> None:
    year_summary = build_year_summary(df)
    author_summary = build_entity_summary(df, "autores", "autor", top_n=20)
    venue_summary = build_entity_summary(df, "periodico_venue", "venue", top_n=20)
    field_summary = build_entity_summary(df, "areas_conhecimento", "area", top_n=20)
    type_summary = build_entity_summary(df, "tipos_publicacao", "tipo", top_n=20)
    open_access_summary = build_open_access_summary(df)
    edge_df, node_df = build_coauthorship_network(df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Produção", "Atores", "Grafos", "Tabelas", "Termos", "Dados"]
    )

    with tab1:
        if not year_summary.empty:
            st.subheader("Publicações por ano")
            st.bar_chart(year_summary.set_index("ano")[["documentos"]])

            st.subheader("Citações acumuladas por ano")
            st.line_chart(year_summary.set_index("ano")[["citacoes"]])

            st.subheader("Taxa de documentos com PDF aberto por ano")
            st.line_chart(year_summary.set_index("ano")[["taxa_pdf_aberto"]])

        scatter_df = df.dropna(subset=["citacoes", "referencias"]).copy()
        if not scatter_df.empty:
            st.subheader("Relação entre referências e citações")
            st.caption("Cada ponto representa um documento recuperado.")
            st.scatter_chart(scatter_df, x="referencias", y="citacoes", size="quantidade_autores", color="ano")

        if not open_access_summary.empty:
            st.subheader("Status de acesso aberto")
            st.bar_chart(open_access_summary.set_index("status_acesso_aberto")[["documentos"]])

        top_cited = df.nlargest(10, "citacoes")[["titulo", "primeiro_autor", "ano", "citacoes", "url"]]
        if not top_cited.empty:
            st.subheader("Artigos mais citados")
            st.dataframe(top_cited, use_container_width=True)

    with tab2:
        if not author_summary.empty:
            st.subheader("Top autores")
            st.bar_chart(author_summary.set_index("autor")[["documentos"]])
            st.dataframe(author_summary, use_container_width=True)

        if not venue_summary.empty:
            st.subheader("Top periódicos / venues")
            st.bar_chart(venue_summary.set_index("venue")[["documentos"]])

        if not type_summary.empty:
            st.subheader("Tipos de publicação")
            st.bar_chart(type_summary.set_index("tipo")[["documentos"]])

        if not field_summary.empty:
            st.subheader("Áreas do conhecimento")
            st.bar_chart(field_summary.set_index("area")[["documentos"]])

    with tab3:
        st.subheader("Grafo de coautoria")
        if edge_df.empty or node_df.empty:
            st.info("Não houve coautorias suficientes nos resultados filtrados para montar o grafo.")
        else:
            st.caption("Nós maiores representam autores com mais documentos. As arestas indicam coautorias.")
            dot_graph = build_coauthorship_dot(edge_df, node_df)
            st.graphviz_chart(dot_graph, use_container_width=True)
            st.dataframe(edge_df.rename(columns={"source": "autor_1", "target": "autor_2"}), use_container_width=True)

    with tab4:
        st.subheader("Resumo por ano")
        if not year_summary.empty:
            st.dataframe(year_summary, use_container_width=True)

        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.subheader("Resumo por periódico / venue")
            if not venue_summary.empty:
                st.dataframe(venue_summary, use_container_width=True)

            st.subheader("Resumo por tipo de publicação")
            if not type_summary.empty:
                st.dataframe(type_summary, use_container_width=True)

        with summary_col2:
            st.subheader("Resumo por área do conhecimento")
            if not field_summary.empty:
                st.dataframe(field_summary, use_container_width=True)

            st.subheader("Resumo por status de acesso aberto")
            if not open_access_summary.empty:
                st.dataframe(open_access_summary, use_container_width=True)

    with tab5:
        terms = top_words(df["titulo"], n=25)
        if not terms.empty:
            st.subheader("Termos mais frequentes nos títulos")
            st.bar_chart(terms.set_index("termo"))

        st.subheader("Amostra dos dados tratados")
        st.dataframe(
            df[
                [
                    "titulo",
                    "autores",
                    "periodico_venue",
                    "tipos_publicacao",
                    "areas_conhecimento",
                    "citacoes",
                ]
            ].head(15),
            use_container_width=True,
        )

    with tab6:
        st.subheader("Resultados completos")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        xlsx_bytes = dataframe_to_excel_bytes(df)
        down1, down2 = st.columns(2)
        down1.download_button(
            "Baixar CSV",
            data=csv_bytes,
            file_name="semantic_scholar_bibliometria.csv",
            mime="text/csv",
            use_container_width=True,
        )
        down2.download_button(
            "Baixar Excel",
            data=xlsx_bytes,
            file_name="semantic_scholar_bibliometria.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="Análise Bibliométrica Semantic Scholar", layout="wide")
    st.title("Análise Bibliométrica com Semantic Scholar")
    st.caption(
        "Busca baseada no endpoint bulk search da Semantic Scholar, com indicadores, gráficos e exportação."
    )

    load_dotenv()
    env_api_key = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    try:
        secrets_api_key = st.secrets.get("S2_API_KEY") or st.secrets.get("SEMANTIC_SCHOLAR_API_KEY")
    except Exception:
        secrets_api_key = ""
    default_api_key = env_api_key or secrets_api_key or ""
    has_managed_api_key = bool(default_api_key)

    with st.sidebar:
        st.header("Consulta")
        if has_managed_api_key:
            st.success("A chave da API ja esta configurada no servidor. Os visitantes nao precisam informar uma chave.")
            override_api_key = st.toggle("Usar outra chave nesta sessao", value=False)
            if override_api_key:
                api_key = st.text_input(
                    "Substituir chave da API",
                    type="password",
                    value=default_api_key,
                    help="Use isso apenas se quiser testar outra chave nesta sessao.",
                )
            else:
                api_key = default_api_key
        else:
            api_key = st.text_input(
                "Chave da API Semantic Scholar",
                type="password",
                value="",
                help="Opcional para testes leves, mas recomendada para uso recorrente.",
            )
        query = st.text_area("Consulta", value=DEFAULT_QUERY, height=110)
        year_filter = st.text_input(
            "Filtro de ano",
            value="2020-",
            help="Exemplos: 2020-2024, 2023- ou -2019.",
        )
        page_size = st.slider("Resultados por página", min_value=25, max_value=100, value=50, step=25)
        max_results = st.slider("Máximo de resultados para análise", min_value=50, max_value=2000, value=250, step=50)
        run = st.button("Buscar e analisar", type="primary", use_container_width=True)

    with st.expander("Exemplos de sintaxe para a busca", expanded=False):
        st.markdown(
            """
            - `"artificial intelligence" libraries`
            - `(library | libraries | biblioteca | bibliotecas) "large language model"`
            - `education -medical "machine learning"`
            """
        )

    if not run and "semantic_scholar_df" not in st.session_state:
        st.info("Ajuste a consulta e clique em 'Buscar e analisar'.")
        return

    if run:
        if not query.strip():
            st.error("Informe uma consulta válida.")
            return

        try:
            with st.spinner("Consultando a Semantic Scholar..."):
                result = fetch_all_results(
                    api_key=api_key.strip(),
                    query=query.strip(),
                    year_filter=year_filter.strip(),
                    page_size=page_size,
                    max_results=max_results,
                )
            df = papers_to_dataframe(result.papers)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            detail = ""
            if exc.response is not None:
                try:
                    detail = exc.response.json().get("error") or exc.response.text
                except ValueError:
                    detail = exc.response.text
            if status in {401, 403}:
                st.error("A consulta foi rejeitada. Revise a chave da API ou tente novamente sem autenticação.")
            elif status == 429:
                st.error("A API sinalizou limite de taxa. Tente novamente em instantes ou use uma chave da API.")
            else:
                st.error(f"Erro HTTP ao consultar a Semantic Scholar: status {status}. {detail[:200]}")
            return
        except requests.RequestException as exc:
            st.error(f"Erro de conexão com a API: {exc}")
            return
        except Exception as exc:
            st.error(f"Falha inesperada: {exc}")
            return

        if df.empty:
            st.warning("A busca foi concluída, mas não retornou documentos.")
            return

        st.session_state["semantic_scholar_df"] = df
        st.session_state["semantic_scholar_result"] = {
            "total_results": result.total_results,
            "retrieved_results": result.retrieved_results,
        }

    df = st.session_state["semantic_scholar_df"].copy()
    result_meta = st.session_state["semantic_scholar_result"]

    with st.container():
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
        min_citations = filter_col1.number_input("Citações mínimas", min_value=0, value=0, step=1)
        only_open_access = filter_col2.checkbox("Somente com PDF aberto", value=False)
        available_types = split_multivalue_column(df["tipos_publicacao"]).dropna().unique().tolist()
        selected_types = filter_col3.multiselect(
            "Filtrar tipos de publicação",
            options=sorted(available_types),
        )

    filtered_df = filter_dataframe(
        df=df,
        min_citations=int(min_citations),
        only_open_access=only_open_access,
        publication_types=selected_types,
    )

    if filtered_df.empty:
        st.warning("Os filtros locais removeram todos os resultados. Ajuste os critérios e tente novamente.")
        return

    st.success(
        "Busca concluída. "
        f"A API estimou {result_meta['total_results']} documentos, recuperamos {result_meta['retrieved_results']} e "
        f"{len(filtered_df)} permanecem após os filtros locais."
    )
    show_metric_block(filtered_df)
    show_charts(filtered_df)


if __name__ == "__main__":
    main()
