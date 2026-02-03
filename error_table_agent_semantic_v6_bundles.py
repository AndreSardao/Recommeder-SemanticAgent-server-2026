"""
Created on Tue Nov 11 09:12:43 2025

@author: andresardao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Semantic Agent: Errors Tables Builder

"""

import argparse, ast, json, re, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import pandas as pd

try:
    import pdfplumber
except Exception:
    pdfplumber = None

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from Listing_maker import mk_listings

ALT_LETTERS = ['A','B','C','D']

@dataclass
class CaseText:
    """Minimal structure for representing a case (incorrect option)."""
    question_id: int
    alt_id: int
    text: str

# --- String and naming conventions utilities ---
def prereq_combo_code(prereqs: List[str]) -> str:
    """
    Ex.: ["EF04MA09","EF04MA10"] -> "EF04MA0910"
    - remove duplicatas
    - ordena
    - concatena sufixos numéricos (2 dígitos) preservando prefixo BNCC
    """
    if not prereqs:
        return ""

    # dedupe mantendo ordenação lexical (estável/replicável)
    prs = sorted(set(prereqs))

    # prefixo = tudo até os dois últimos dígitos (assume BNCC padrão)
    # EF04MA09 -> prefixo EF04MA, sufixo 09
    m0 = re.match(r"^(.+?)(\d{2})$", prs[0])
    if not m0:
        return "".join(prs)  # fallback feio, mas seguro

    prefix = m0.group(1)

    suffixes = []
    for p in prs:
        m = re.match(r"^(.+?)(\d{2})$", p)
        if m and m.group(1) == prefix:
            suffixes.append(m.group(2))
        else:
            # se tiver prefixos diferentes, cai para concatenação simples
            return "".join(prs)

    return prefix + "".join(suffixes)


# Extracts 'YYSS' from 'EFYYMASS' via simple slicing.
def code4(skill: str) -> str:
    s = skill.strip().upper()
    yy = s[2:4] if len(s)>=4 else '00'
    ss = s[-2:] if len(s)>=2 else '00'
    if not yy.isdigit(): yy='00'
    if not ss.isdigit():
        digits = ''.join(ch for ch in s if ch.isdigit())[-2:]
        ss = digits.zfill(2) if digits else '00'
    return yy+ss

# Common prefix between strings (lexicographical comparison).
def common_prefix(strings: List[str]) -> str:
    if not strings: return ''
    s1, s2 = min(strings), max(strings)
    for i,ch in enumerate(s1):
        if i>=len(s2) or ch!=s2[i]: return s1[:i]
    return s1

# Assembles 'EFxxMA060710' with common prefix + last 2 digits of each prefix
def prereq_combo_tag(prereqs: List[str]) -> str:
    if not prereqs: return ''
    prefix = common_prefix(prereqs)
    if len(prefix)<5: return ''.join(prereqs)
    tails=[]
    for s in sorted(prereqs):
        two = s[-2:]
        if not two.isdigit(): two = ''.join(ch for ch in s if ch.isdigit())[-2:]
        tails.append(two.zfill(2))
    return prefix + ''.join(tails)

""""
Extracts and reconstructs one or more Python-like blocks of questions even when there are internal line breaks introduced by PDF.

Steps:

1. Locate all sections between '[' and ']' that contain 'question_id'

2. Remove internal line breaks within strings

3. Apply ast.literal_eval to each block

4. Concatenate all the lists of questions found
"""
# --- Reading questions listings ---
def extract_questions_block(text: str) -> List[Dict[str, Any]]:
    
    # 1. Spotting all the raw blocks
    blocks = []
    # General pattern: list starting with "{'question_id'"
    for m in re.finditer(r"(\[\s*\{[\s\S]*?\}\s*\])", text):
        blocks.append(m.group(1))
    # fallback, whenever the general case is not the case
    if not blocks:
        for m in re.finditer(r"(\[\{\s*'question_id'.*?}])", text, flags=re.DOTALL):
            blocks.append(m.group(1))
    if not blocks:
        raise ValueError("Bloco(s) de questões não encontrado(s) no texto.")

    def fix_multiline_strings(s: str) -> str:
        """Remove internal line breaks within strings, replacing them with spaces. """ 
        new = []
        inside_string = False
        quote_char = None
        i = 0
        while i < len(s):
            ch = s[i]
            if not inside_string and ch in ("'", '"'):
                inside_string = True
                quote_char = ch
                new.append(ch)
            elif inside_string and ch == quote_char:
                inside_string = False
                quote_char = None
                new.append(ch)
            elif inside_string and ch in '\n\r':
                new.append(' ')
            else:
                new.append(ch)
            i += 1
        return ''.join(new)

    all_questions: List[Dict[str, Any]] = []
    for raw_block in blocks:
        fixed_block = fix_multiline_strings(raw_block)
        try:
            data = ast.literal_eval(fixed_block)
        except Exception as e:
            raise ValueError(
                f"Falha ao interpretar bloco de questões: {e}\nBloco reconstruído (início):\n{fixed_block[:1000]}"
            )
        if not isinstance(data, list):
            raise ValueError("Bloco de questões interpretado não é uma lista.")
        all_questions.extend(data)

    return all_questions


def read_question_listing(path: Path) -> List[Dict[str, Any]]:
    """
    Reads a questions listing from:
      - Structured CSV (columns like question_id, hab_bncc, Enunciado, A_alt_id, resposta_certa, ...)
      - Legacy CSV with a 'texto' column containing a Python-like list block
      - PDF (legacy)
    Returns: list[dict] compatible with build_cases_for_skill(...)
    """
    suf = path.suffix.lower()

    if suf == ".csv":
        df = pd.read_csv(path)

        # New structured format (produced by Listing_maker)
        if "question_id" in df.columns and "hab_bncc" in df.columns:
            # Normalize NaNs to None so downstream is stable
            df = df.where(pd.notna(df), None)
            return df.to_dict(orient="records")

        # Legacy format with 'texto' column
        if "texto" in df.columns:
            full = "\n".join(str(x) for x in df["texto"].tolist())
            return extract_questions_block(full)

        raise ValueError(
            "CSV listing must be either structured (question_id/hab_bncc/...) or legacy (column 'texto')."
        )

    if suf == ".pdf":
        if pdfplumber is None:
            raise RuntimeError("Install pdfplumber or use CSV for listings.")
        parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return extract_questions_block("\n".join(parts))

    raise ValueError(f"Unsupported listing format: {path}")


# --- Curated tables ---
CLASSIC_COLS = ["question_id","alternative_id","error_code","error_name","error_description"]

def normalize_curated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza colunas vindas de CSV curado (PT) ou DB (EN) para o padrão do DB (EN):
      question_id, alternative_id, error_code, error_name, error_description
    """
    cols_lower = {c.lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)

    # question_id
    if "question_id" in df.columns:
        out["question_id"] = df["question_id"]
    elif "id_da_questao" in df.columns:
        out["question_id"] = df["id_da_questao"]
    elif "id_da_questão" in df.columns:
        out["question_id"] = df["id_da_questão"]
    elif "id da questão" in df.columns:
        out["question_id"] = df["id da questão"]
    elif "id_da_questao " in df.columns:
        out["question_id"] = df["id_da_questao "]
    else:
        found = None
        for k, v in cols_lower.items():
            if (("quest" in k or "question" in k) and "id" in k):
                found = v
                break
        out["question_id"] = df[found] if found else pd.NA

    # alternative_id
    if "alternative_id" in df.columns:
        out["alternative_id"] = df["alternative_id"]
    elif "id_da_alternativa" in df.columns:
        out["alternative_id"] = df["id_da_alternativa"]
    elif "id da alternativa" in df.columns:
        out["alternative_id"] = df["id da alternativa"]
    else:
        found = None
        for k, v in cols_lower.items():
            if (("alternativ" in k or "alternativa" in k) and "id" in k):
                found = v
                break
        out["alternative_id"] = df[found] if found else pd.NA

    # error_code
    if "error_code" in df.columns:
        out["error_code"] = df["error_code"]
    elif "codigo_do_erro" in df.columns:
        out["error_code"] = df["codigo_do_erro"]
    elif "código do erro" in df.columns:
        out["error_code"] = df["código do erro"]
    elif "codigo do erro" in df.columns:
        out["error_code"] = df["codigo do erro"]
    else:
        found = None
        for k, v in cols_lower.items():
            if (("error" in k and "code" in k) or ("codigo" in k and "erro" in k)):
                found = v
                break
        out["error_code"] = df[found] if found else pd.NA

    # error_name
    if "error_name" in df.columns:
        out["error_name"] = df["error_name"]
    elif "nome_erro" in df.columns:
        out["error_name"] = df["nome_erro"]
    elif "nome do erro por extenso" in df.columns:
        out["error_name"] = df["nome do erro por extenso"]
    elif "nome do erro" in df.columns:
        out["error_name"] = df["nome do erro"]
    else:
        found = None
        for k, v in cols_lower.items():
            if (("error" in k and "name" in k) or ("nome" in k and "erro" in k)):
                found = v
                break
        out["error_name"] = df[found] if found else pd.NA

    # error_description
    if "error_description" in df.columns:
        out["error_description"] = df["error_description"]
    elif "descricao_erro" in df.columns:
        out["error_description"] = df["descricao_erro"]
    elif "descrição do erro" in df.columns:
        out["error_description"] = df["descrição do erro"]
    elif "Descrição do erro" in df.columns:
        out["error_description"] = df["Descrição do erro"]
    elif "descricao do erro" in df.columns:
        out["error_description"] = df["descricao do erro"]
    else:
        found = None
        for k, v in cols_lower.items():
            if (("error" in k and "description" in k) or ("descricao" in k and "erro" in k)):
                found = v
                break
        out["error_description"] = df[found] if found else pd.NA

    return out[CLASSIC_COLS]

def read_curated_table(path: Path) -> pd.DataFrame:
    """Lê CSV curado e normaliza colunas."""
    if path.suffix.lower()!='.csv':
        raise ValueError('Tabelas curadas devem ser CSV.')
    return normalize_curated_columns(pd.read_csv(path))


# --- Curated bundles reading (HAB + pré-requisitos) ---
def load_curated_from_bundles(config_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Lê um JSON de bundles curados e devolve:
      - lista de caminhos de CSVs de tabelas curadas
      - lista de caminhos de listagens de questões

    O JSON deve ser uma lista de objetos, cada um com, por exemplo:
      - "tables": dict ou lista de caminhos de CSV (tabelas de erros)
      - "listings": dict ou lista de caminhos de CSV/PDF (listagens de questões)
      - "hab", "prereqs" e "coupling_type" como metadados semânticos.

    O agente usa TODAS as tabelas/listagens descritas para construir um
    catálogo global de padrões de erro, preservando diferenças como
    EF04MA09_1 vs EF04MA09_2 em bundles distintos.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        bundles = json.load(f)

    curated_tables_files: List[Path] = []
    curated_listings_files: List[Path] = []

    for bundle in bundles:
        # Tabelas
        tables = bundle.get("tables", [])
        if isinstance(tables, dict):
            table_paths = tables.values()
        else:
            table_paths = tables
        for t in table_paths:
            curated_tables_files.append(Path(t))

        # Listagens
        listings = bundle.get("listings", [])
        if isinstance(listings, dict):
            listing_paths = listings.values()
        else:
            listing_paths = listings
        for l in listing_paths:
            curated_listings_files.append(Path(l))

    return curated_tables_files, curated_listings_files


# --- Semantic Catlog built from exemples ---
def build_semantic_catalog(curated_tables: List[pd.DataFrame], question_lists: List[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Creating a catalog: label → {description, examples} using (statement + alternative)."""
    alt_index: Dict[int, Tuple[str,str]] = {}
    for qlist in question_lists:
        for q in qlist:
            try:
                enun = str(q.get('Enunciado',''))
                for L in ALT_LETTERS:
                    aid = int(q.get(f'{L}_alt_id'))
                    atext = str(q.get(L,''))
                    alt_index[aid] = (enun, atext)
            except Exception:
                continue
    catalog: Dict[str, Dict[str, Any]] = {}
    print("[CATALOG] alt_index size:", len(alt_index))
    for df in curated_tables:
        for _, row in df.iterrows():
            try:
                alt_id = int(row["alternative_id"])
            except Exception:
                continue
    
            raw_label = row.get("error_name", "")
            raw_desc  = row.get("error_description", "")
    
            label = "" if pd.isna(raw_label) else str(raw_label).strip()
            desc  = "" if pd.isna(raw_desc)  else str(raw_desc).strip()
    
            if not label:
                continue
    
            enun, atext = alt_index.get(alt_id, ("", ""))
            blob = f"{enun}\nOPCAO_: {atext}".strip()
    
            if label not in catalog:
                catalog[label] = {"descricao": desc, "exemplos": []}
    
            if desc and not catalog[label]["descricao"]:
                catalog[label]["descricao"] = desc
    
            if blob:
                catalog[label]["exemplos"].append(blob)
    print("[CATALOG] n_labels:", len(catalog))
    if not catalog:
        print("[CATALOG] EMPTY -> check curated tables schema / loading")
    return catalog


# --- LLM Semantic Classification ---
def make_llm(model_name: str = 'gpt-4o-mini', temperature: float = 0.0) -> ChatOpenAI:
    """Instantiates the LLM client.(LangChain-OpenAI)."""
    return ChatOpenAI(model=model_name, temperature=temperature)

CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
   """ Você é um avaliador pedagógico especializado em Matemática do Ensino Fundamental.
Seu trabalho é classificar a alternativa errada abaixo em um dos TIPOS DE ERRO listados,
usando a taxonomia e alguns exemplos representativos.

REGRAS:
- Escolha APENAS um "label" existente (um dos nomes de erro por extenso).
- Se não houver encaixe razoável, responda com label = "NOVO_PADRAO".
- Responda em JSON puro com as chaves: "label", "rationale", "confidence" (entre 0 e 1).

TAXONOMIA:
{taxonomy}

EXEMPLOS (amostragem por rótulo):
{examples}

CASO PARA CLASSIFICAR:
\"\"\"{case_text}\"\"\"

"""
)

def fmt_taxonomy(catalog: Dict[str, Dict[str, Any]]) -> str:
    """Formating '- label: description' for the prompt."""
    lines=[]
    for label, meta in catalog.items():
        desc = str(meta.get('descricao','') or '(sem descrição)')
        lines.append(f"- {label}: {desc}")
    return '\n'.join(lines) if lines else '(vazia)'

def fmt_examples(catalog: Dict[str, Dict[str, Any]], max_per:int=3, max_chars:int=420) -> str:
    """Limita a 3 exemplos/label e trunca cada exemplo para ~420 chars."""
    blocks=[]
    for label, meta in catalog.items():
        exs = meta.get('exemplos', [])[:max_per]
        if not exs: continue
        lines=[]
        for t in exs:
            t2 = t if len(t)<=max_chars else t[:max_chars] + ' …'
            lines.append('• ' + t2)
        blocks.append(f'[{label}]\n' + '\n'.join(lines))
    return '\n\n'.join(blocks) if blocks else '(sem exemplos)'

def llm_classify(llm: ChatOpenAI, catalog: Dict[str, Dict[str, Any]], case_text: str) -> Tuple[str,str,float]:
    """Classifica um caso (retorna label, rationale, confidence)."""
    msg = CLASSIFY_PROMPT.format_messages(
        taxonomy=fmt_taxonomy(catalog),
        examples=fmt_examples(catalog),
        case_text=case_text,
    )
    resp = llm.invoke(msg)
    raw = (resp.content or '').strip()
    try:
        data = json.loads(raw)
        return str(data.get('label','NOVO_PADRAO')), str(data.get('rationale','')), float(data.get('confidence',0.0))
    except Exception:
        return 'NOVO_PADRAO', 'Falha ao interpretar JSON do LLM (revisar).', 0.0

# --- Construção dos casos por habilidade ---
def build_cases_for_skill(questions: List[Dict[str, Any]], skill_code: str) -> List[CaseText]:
    """Converte alternativas erradas das questões em objetos CaseText."""
    out: List[CaseText] = []
    for q in questions:
        if str(q.get('hab_bncc','')).strip().upper() != skill_code.strip().upper():
            continue
        enun = str(q.get('Enunciado',''))
        for L in ALT_LETTERS:
            try:
                alt_id = int(q.get(f'{L}_alt_id'))
                alt_text = str(q.get(L,''))
            except Exception:
                continue
            if str(q.get('resposta_certa','')).strip().upper() == L:
                continue
            out.append(CaseText(int(q['question_id']), alt_id, f"{enun}\nOPCAO_{L}: {alt_text}"))
    return out

# --- Acoplamento e convenção de nomes ---
def assign_codes(hab: str, prereqs: List[str],
                  hab_map: Dict[str, List[Tuple[int,int,str]]],
                  pre_map_all: Dict[str, Dict[str, List[Tuple[int,int,str]]]]
                  ) -> Tuple[Dict[str,str], Dict[str, Dict[str,str]]]:
    """Nomeia rótulos (acoplados e exclusivos) com base em code4 e convenção Erro_k/x."""
    hab_code = code4(hab)
    pre_codes = {pre: code4(pre) for pre in prereqs}
    hab_labels = set(k for k in hab_map if k!='NOVO_PADRAO')
    pre_labels = {pre: set(k for k in mp if k!='NOVO_PADRAO') for pre, mp in pre_map_all.items()}
    coupled = set(lbl for lbl in sorted(hab_labels) if any(lbl in s for s in pre_labels.values()))
    hab_names: Dict[str,str] = {}
    pre_names: Dict[str, Dict[str,str]] = {pre:{} for pre in prereqs}
    idx = 1
    for lbl in sorted(coupled):
        hab_names[lbl] = f"{hab_code}_Erro_{idx}"
        for pre in prereqs:
            if lbl in pre_labels.get(pre, set()):
                pre_names[pre][lbl] = f"{pre_codes[pre]}_Erro_{idx}"
        idx += 1
    for lbl in sorted(hab_labels - coupled):
        hab_names[lbl] = f"{hab_code}_Erro_{idx}x"; idx += 1
    for pre in prereqs:
        used=[]
        for name in pre_names[pre].values():
            m=re.search(r"_Erro_(\d+)$", name)
            if m: used.append(int(m.group(1)))
        start = max(used) if used else 0
        k = start + 1
        for lbl in sorted(pre_labels[pre] - coupled):
            pre_names[pre][lbl] = f"{pre_codes[pre]}_Erro_{k}x"; k += 1
    return hab_names, pre_names


##########################
# Colunas alvo (contrato)
##########################
CLASSIC_COLS = [
    "question_id",
    "alternative_id",
    "error_code",
    "error_name",
    "error_description",
]

EXTRA_COLS = [
    "error_table_title",
]

FINAL_COLS = CLASSIC_COLS + EXTRA_COLS


################################
# Renames mais comuns -> padrão
################################
CSV_TO_CLASSIC_RENAMES = {
    # seus nomes antigos
    "id_da_questao": "question_id",
    "id_da_alternativa": "alternative_id",
    "codigo_erro": "error_code",
    "nome_erro": "error_name",
    "Descrição do erro": "error_description",
    "descricao_do_erro": "error_description",
    "descricao erro": "error_description",
    "descricao": "error_description",

    # já no padrão
    "question_id": "question_id",
    "alternative_id": "alternative_id",
    "error_code": "error_code",
    "error_name": "error_name",
    "error_description": "error_description",
    "error_table_title": "error_table_title",
}


class ErrorTableSchemaError(Exception):
    """Erro de schema/colunas ao padronizar tabela de erros."""


def _rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: CSV_TO_CLASSIC_RENAMES.get(c, c) for c in df.columns})


def _ensure_columns_exist(df: pd.DataFrame, cols: list[str], fill_value="") -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def _coerce_text_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna("").astype(str)
    return df


def ensure_error_description_column(
    df: pd.DataFrame,
    *,
    use_llm: bool = False,
    run_agent_local=None,
    logger=None,
) -> pd.DataFrame:
    """
    Subagente opcional: garante 'error_description'.

    - Se já existir e tiver conteúdo, mantém.
    - Se não existir: cria.
    - Se use_llm=True: tenta gerar descrições curtas para pares (error_code, error_name)
      e preencher as linhas.
    """
    df = df.copy()

    if "error_description" not in df.columns:
        df["error_description"] = ""

    # se não for usar LLM, acabou
    if not use_llm:
        return df

    if run_agent_local is None:
        if logger:
            logger.warning("use_llm=True, mas run_agent_local=None. Mantendo descricoes vazias.")
        return df

    # Gera descrições apenas para pares únicos
    base = (
        df[["error_code", "error_name"]]
        .fillna("")
        .astype(str)
        .drop_duplicates()
        .to_dict(orient="records")
    )

    prompt = {
        "task": "Completar coluna error_description em tabela de classificacao de erros",
        "instructions": [
            "Para cada (error_code, error_name), produza uma descricao curta e objetiva em PT-BR.",
            "Uma frase curta, focada no erro conceitual do aluno.",
            "Nao use exemplos longos e nao inclua meta-explicacoes.",
        ],
        "items": base,
        "output_format": "json_list",
        "output_schema": [{"error_code": "str", "error_name": "str", "error_description": "str"}],
    }

    try:
        result = run_agent_local(prompt)
        items = result["items"] if isinstance(result, dict) and "items" in result else result

        mapping: dict[tuple[str, str], str] = {}
        for it in items:
            k = (str(it.get("error_code", "")).strip(), str(it.get("error_name", "")).strip())
            mapping[k] = str(it.get("error_description", "")).strip()

        def _fill(row):
            k = (str(row.get("error_code", "")).strip(), str(row.get("error_name", "")).strip())
            return mapping.get(k, "")

        # preenche apenas onde está vazio
        mask_empty = df["error_description"].fillna("").astype(str).str.strip().eq("")
        df.loc[mask_empty, "error_description"] = df.loc[mask_empty].apply(_fill, axis=1)

        if logger:
            filled = (df["error_description"].astype(str).str.strip() != "").sum()
            logger.info("error_description preenchida via LLM: %s/%s", filled, len(df))

    except Exception as e:
        if logger:
            logger.error("Falha ao gerar error_description via LLM. Mantendo vazio. Err=%s", str(e))

    return df


def finalize_error_table_df(
    df: pd.DataFrame,
    *,
    error_table_title: str,
    use_llm_description: bool = False,
    run_agent_local=None,
    logger=None,
) -> pd.DataFrame:
    """
    Versão definitiva: normaliza a tabela para o contrato final:

      question_id, alternative_id, error_code, error_name, error_description, error_table_title

    - Renomeia colunas conhecidas para o padrão.
    - Garante existência das colunas clássicas.
    - Garante error_table_title preenchida uniformemente.
    - Opcionalmente completa error_description via subagente (LLM local).
    - Retorna somente as colunas finais na ordem.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ErrorTableSchemaError("finalize_error_table_df: df inválido (não é DataFrame).")

    df2 = _rename_known_columns(df)
    df2 = _ensure_columns_exist(df2, CLASSIC_COLS, fill_value="")
    df2 = ensure_error_description_column(
        df2,
        use_llm=use_llm_description,
        run_agent_local=run_agent_local,
        logger=logger,
    )

    # coluna-chave para DB/loader
    df2["error_table_title"] = str(error_table_title)

    # coerções de texto (evita NaN e problemas de tipo)
    df2 = _coerce_text_cols(df2, ["error_code", "error_name", "error_description", "error_table_title"])

    # validação mínima: não deixar columns críticas faltando
    missing = [c for c in FINAL_COLS if c not in df2.columns]
    if missing:
        raise ErrorTableSchemaError(f"Colunas faltando após finalização: {missing}")

    return df2[FINAL_COLS].copy()


# --- Escrita dos CSVs de saída ---
def write_output_tables(hab: str,
                       cases_hab: list,
                       cases_pre: list,
                       outdir: str,
                       debug: bool = False,
                       prereq_combo: str | None = None):
    """
    Escreve as tabelas finais com schema EN (DB):
      question_id, alternative_id, error_code, error_name, error_description

    debug=True adiciona colunas auxiliares (não use para DB):
      llm_label_raw, llm_confidence
    """
    os.makedirs(outdir, exist_ok=True)

    def _to_rows(cases):
        rows = []
        for c in cases:
            # Expecting: c.qid, c.alt_id, c.code_err, c.label, c.rationale
            # Se no Case object os nomes forem diferentes, preciso lembrar de ajustar aqui.
            qid = getattr(c, "qid", getattr(c, "question_id", None))
            alt_id = getattr(c, "alt_id", getattr(c, "alternative_id", None))
            code_err = getattr(c, "code_err", getattr(c, "error_code", None))
            label = getattr(c, "label", getattr(c, "error_name", None))
            rat = getattr(c, "rationale", getattr(c, "error_description", ""))

            row = {
                "question_id": qid,
                "alternative_id": alt_id,
                "error_code": code_err,
                "error_name": label,
                "error_description": rat,
            }

            if debug:
                row["llm_label_raw"] = getattr(c, "llm_label_raw", label)
                row["llm_confidence"] = getattr(c, "llm_confidence", getattr(c, "confidence", pd.NA))

            rows.append(row)
        return rows

    # HAB
    hab_rows = _to_rows(cases_hab)
    df_hab = pd.DataFrame(hab_rows)
    # garante colunas e ordem
    for col in CLASSIC_COLS:
        if col not in df_hab.columns:
            df_hab[col] = pd.NA
    df_hab = df_hab[CLASSIC_COLS + ([c for c in df_hab.columns if c not in CLASSIC_COLS] if debug else [])]

    hab_path = os.path.join(outdir, f"classificacao_erros_{hab}.csv")
    df_hab.to_csv(hab_path, index=False)

    # PRE (se houver)
    pre_path = None
    if cases_pre:
        pre_rows = _to_rows(cases_pre)
        df_pre = pd.DataFrame(pre_rows)
        for col in CLASSIC_COLS:
            if col not in df_pre.columns:
                df_pre[col] = pd.NA
        df_pre = df_pre[CLASSIC_COLS + ([c for c in df_pre.columns if c not in CLASSIC_COLS] if debug else [])]

        # nome do combo prereq já usado em outro lugar; aqui segue padrão:
        #prereqs = sorted(set(prereqs)) if prereqs else []
        #combo = prereq_combo_code(prereqs) if prereqs else None

        combo = prereq_combo or f"{hab}_PREREQS"
        pre_path = os.path.join(outdir, f"classificacao_erros_{combo}.csv")
        df_pre.to_csv(pre_path, index=False)

    return hab_path, pre_path

# --- Main Pipeline (local) ---
def run_agent_local(hab: str, prereqs: List[str],
                     curated_tables_files: List[Path],
                     curated_listings_files: List[Path],
                     new_listings_files: List[Path],
                     outdir: Path,
                     model: str='gpt-4o-mini', temperature: float=0.0,
                     min_confidence: float=0.35) -> Tuple[Path, Optional[Path]]:
    """Executa o fluxo ponta-a-ponta local (sem integrar ao Recommender)."""
    prereqs = sorted(set(prereqs)) if prereqs else []
    combo = prereq_combo_code(prereqs) if prereqs else None
    
    curated_tables = [read_curated_table(p) for p in curated_tables_files]
    curated_lists = [read_question_listing(p) for p in curated_listings_files]
    catalog = build_semantic_catalog(curated_tables, curated_lists)
    # Debug nights"
    print("[CATALOG] labels:", len(catalog))
    
    new_questions=[]
    for p in new_listings_files:
        new_questions.extend(read_question_listing(p))
    llm = make_llm(model_name=model, temperature=temperature)
    def classify_cases(cases: List[CaseText]) -> Dict[str, List[Tuple[int,int,str]]]:
        m={}
        for c in cases:
            label, rat, conf = llm_classify(llm, catalog, c.text)
            if conf < min_confidence:
                label = 'NOVO_PADRAO'
            m.setdefault(label, []).append((c.question_id, c.alt_id, rat))
        return m
    hab_cases = build_cases_for_skill(new_questions, hab)
    pre_cases = {pre: build_cases_for_skill(new_questions, pre) for pre in prereqs}
    hab_map = classify_cases(hab_cases)
    # Dancing with the Bugs
    print("[HAB_MAP] labels:", list(hab_map.keys())[:10])
    pre_map_all = {pre: classify_cases(cs) for pre, cs in pre_cases.items()}
    hab_names, pre_names = assign_codes(hab, prereqs, hab_map, pre_map_all)
    print("[CODES] hab_codes:", len(hab_names), "pre_codes_total:", sum(len(d) for d in pre_names.values()) if isinstance(pre_names, dict) else "na")

    from types import SimpleNamespace

    def materialize_cases(map_by_label, names_by_label):
        """
        map_by_label: dict[label] -> list[(question_id, alt_id, rationale)]
        names_by_label: dict[label] -> error_code
        Retorna list de objetos com atributos esperados pelo write_output_tables:
          qid/alt_id/code_err/label/rationale
        """
        out = []
        for label, triples in (map_by_label or {}).items():
            code = (names_by_label or {}).get(label, None)
            for (qid, alt_id, rat) in triples:
                out.append(SimpleNamespace(
                    qid=qid,
                    alt_id=alt_id,
                    code_err=code,
                    label=label,
                    rationale=rat
                ))
        return out

    cases_hab_out = materialize_cases(hab_map, hab_names)

    # prereqs: UMA tabela agregada, então “achata” tudo numa lista só
    cases_pre_out = []
    for pre in prereqs:
        cases_pre_out.extend(materialize_cases(pre_map_all.get(pre, {}), pre_names.get(pre, {})))
  
    return write_output_tables(
    hab=hab,
    cases_hab=cases_hab_out,
    cases_pre=cases_pre_out,
    outdir=outdir,
    debug=False,
    prereq_combo=combo,
)


def run_agent_for_hab(hab: str, outdir: str = "./out_error_tables"):
    """
    Wrapper de alto nível para rodar o agente a partir de uma habilidade (hab),
    montando automaticamente prereqs + curated + listings e chamando run_agent_local().
    Ideal para testes (ii) e para o endpoint do SGP.
    """
    # 1) prereqs: reaproveite a mesma fonte do recommender, se existir
    try:
        from services_recommender import get_module  # ajuste se o import for outro
        module = get_module(hab)
        prereqs = module.get("prereqs", []) if isinstance(module, dict) else []
    except Exception:
        prereqs = []

    # 2) curated: carregar bundles (você já tem isso)
    curated_tables_files, curated_listings_files = load_curated_from_bundles()

    # 3) new_listings_files: gerar on-the-fly
    # A ideia é: produzir um dict/list com as listagens necessárias para hab e prereqs.
    new_listings_files = mk_listings(hab=hab, prereqs=prereqs)  # ajuste assinatura

    # 4) chamar a função de baixo nível
    return run_agent_local(
        hab=hab,
        prereqs=prereqs,
        curated_tables_files=curated_tables_files,
        curated_listings_files=curated_listings_files,
        new_listings_files=new_listings_files,
        outdir=outdir,
    )


# --- CLI ---
def parse_args() -> argparse.Namespace:
    """Argumentos da CLI para rodar localmente com seus arquivos."""
    ap = argparse.ArgumentParser(description='Agente Semântico de Tabelas de Erros (local, comentado)')
    ap.add_argument('--hab', required=True, help="Habilidade-alvo (ex.: EF05MA08)")
    ap.add_argument('--prereqs', default='', help="Pré-requisitos separados por vírgula (ex.: EF04MA06,EF04MA07)")
    ap.add_argument(
        '--curated-bundles',
        default='',
        help='Arquivo JSON com bundles curados (HAB + pré-requisitos) para montar o catálogo de exemplos'
    )
    ap.add_argument('--curated-tables', help='CSVs de tabelas curadas (separados por vírgula)', default='')
    ap.add_argument('--curated-listings', help='Listagens de origem (CSV/PDF) separadas por vírgula', default='')
    ap.add_argument('--new-listings', required=True, help='Listagens novas (CSV/PDF) separadas por vírgula (HAB + PRÉs)')
    ap.add_argument('--outdir', required=True, help='Diretório de saída dos CSVs gerados')
    ap.add_argument('--model', default='gpt-5.1', help='Modelo do LLM (via langchain_openai)')
    ap.add_argument('--temperature', type=float, default=0.0, help='Temperatura do LLM')
    ap.add_argument('--min-confidence', type=float, default=0.35, help='Confiança mínima (0..1) para aceitar um rótulo')
    return ap.parse_args()

def main() -> None:
    """Ponto de entrada principal quando usar via CLI."""
    args = parse_args()
    hab = args.hab.strip()
    prereqs = [s.strip() for s in args.prereqs.split(',') if s.strip()]

    # Escolha de fonte de exemplos curados
    if args.curated_bundles:
        curated_tables_files, curated_listings_files = load_curated_from_bundles(Path(args.curated_bundles))
    else:
        if not args.curated_tables or not args.curated_listings:
            raise ValueError(
                "Você deve fornecer OU --curated-bundles OU os dois: --curated-tables e --curated-listings."
            )
        curated_tables_files = [Path(s.strip()) for s in args.curated_tables.split(',') if s.strip()]
        curated_listings_files = [Path(s.strip()) for s in args.curated_listings.split(',') if s.strip()]

    new_listings_files = [Path(s.strip()) for s in args.new_listings.split(',') if s.strip()]
    outdir = Path(args.outdir)
    run_agent_local(
        hab=hab,
        prereqs=prereqs,
        curated_tables_files=curated_tables_files,
        curated_listings_files=curated_listings_files,
        new_listings_files=new_listings_files,
        outdir=outdir,
        model=args.model,
        temperature=args.temperature,
        min_confidence=args.min_confidence,
    )

if __name__=='__main__':
    main()
