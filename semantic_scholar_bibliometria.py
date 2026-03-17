#!/usr/bin/env python3
"""App Streamlit para busca e análise bibliométrica com a API da Semantic Scholar."""

from __future__ import annotations

import io
import os
import re
from collections import Counter
from dataclasses import dataclass
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


def split_multivalue_column(series: pd.Series) -> pd.Series:
    values: list[str] = []
    for item in series.dropna().astype(str):
        for part in item.split(";"):
            cleaned = part.strip()
            if cleaned:
                values.append(cleaned)
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
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="semantic_scholar", index=False)
    return buffer.getvalue()


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
    tab1, tab2, tab3, tab4 = st.tabs(["Produção", "Atores", "Termos", "Dados"])

    with tab1:
        by_year = df.dropna(subset=["ano"]).groupby("ano").size().reset_index(name="publicacoes")
        if not by_year.empty:
            st.subheader("Publicações por ano")
            st.bar_chart(by_year.set_index("ano"))

        citations_by_year = (
            df.dropna(subset=["ano"])
            .groupby("ano", as_index=False)["citacoes"]
            .sum()
            .rename(columns={"citacoes": "citacoes_totais"})
        )
        if not citations_by_year.empty:
            st.subheader("Citações acumuladas por ano")
            st.line_chart(citations_by_year.set_index("ano"))

        top_cited = df.nlargest(10, "citacoes")[["titulo", "primeiro_autor", "ano", "citacoes", "url"]]
        if not top_cited.empty:
            st.subheader("Artigos mais citados")
            st.dataframe(top_cited, use_container_width=True)

    with tab2:
        authors = split_multivalue_column(df["autores"]).value_counts().head(20)
        if not authors.empty:
            st.subheader("Top autores")
            st.bar_chart(authors)

        venues = df["periodico_venue"].dropna().astype(str).str.strip()
        top_venues = venues[venues.ne("")].value_counts().head(20)
        if not top_venues.empty:
            st.subheader("Top periódicos / venues")
            st.bar_chart(top_venues)

        types = split_multivalue_column(df["tipos_publicacao"]).value_counts().head(15)
        if not types.empty:
            st.subheader("Tipos de publicação")
            st.bar_chart(types)

        fields = split_multivalue_column(df["areas_conhecimento"]).value_counts().head(15)
        if not fields.empty:
            st.subheader("Áreas do conhecimento")
            st.bar_chart(fields)

    with tab3:
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

    with tab4:
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

    with st.sidebar:
        st.header("Consulta")
        api_key = st.text_input(
            "Chave da API Semantic Scholar",
            type="password",
            value=default_api_key,
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
