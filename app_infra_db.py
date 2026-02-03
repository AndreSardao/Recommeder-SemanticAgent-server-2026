"""
Created on Wed Oct 15 02:27:57 2025

@author: andresardao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# app_infra_db.py
from __future__ import annotations

import os
from typing import Any, Optional

import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
import pandas as pd


###################
# DB config (env) #
###################
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "Mandy")
DB_USER = os.getenv("DB_USER", "root") 
DB_PASS = os.getenv("DB_PASS", "")  
DB_NAME_1 = os.getenv("DB_NAME_1", "perguntas_schema")

config = {
    "user": DB_USER,
    "password": DB_PASS,
    "host": DB_HOST,
    "database": DB_NAME,
}

config_1 = {
    "user": DB_USER,
    "password": DB_PASS,
    "host": DB_HOST,
    "database": DB_NAME_1,
}

def get_main_db_conn():
    """
    Conexão para o schema principal (DB_NAME).
    Retorna uma conexão mysql.connector (DB-API).
    """
    return mysql.connector.connect(**config)


def get_questions_db_conn():
    """
    Conexão para o schema de perguntas (DB_NAME_1).
    """
    return mysql.connector.connect(**config_1)


############################
# Bloco_1 helper: fetcher ##
############################
def fetcher(sql: str, params: Optional[tuple] = None, use_questions_db: bool = False) -> pd.DataFrame:
    """
    Helper do Bloco_1: executes SQL and returns a DataFrame.
    use_questions_db=False -> uses `config` (main schema)
    use_questions_db=True  -> uses `config_1` (questions schema)
    """
    conn = mysql.connector.connect(**(config_1 if use_questions_db else config))
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()



###################################################################
# BLOCO_1 (272..663): sub-dataframes para _gera_todos_os_objetos ##
###################################################################

def get_error_tables_relations():
    """
    Recupera a tabela error_tables_relations do DB_name_2
    e retorna no formato:
    
    {
        'EF05MA06': ['classificacao_erros_EF05MA06.csv', 'classificacao_erros_EF04MA0910.csv'],
        ...
    }
    """
    query = """
        SELECT
            bncc_code,
            current_table,
            pre_table
        FROM error_tables_relations
    """

    conn = mysql.connector.connect(**config_1)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    tabs = {}
    for r in rows:
        tabs[r["bncc_code"]] = [r["current_table"], r["pre_table"]]

    return tabs

def get_dc_grade_and_subject(data_challenge_id: int) -> tuple[int, int]:
    """
    Dado um data_challenge_id, retorna (grade_id, subject_id).

    subject_id: vem de data_challenge.subject_id
    grade_id:   vem da cadeia challenges.activity_id -> activity_chapters.chapter_id -> chapters.grade_id
    """
    data_challenge_id = int(data_challenge_id)  # evita numpy.int64 etc.

    conn = mysql.connector.connect(**config)
    try:
        sql = """
        SELECT
            dc.subject_id AS subject_id,
            ch.grade_id   AS grade_id
        FROM data_challenges dc
        JOIN challenges c
            ON c.id = dc.challenge_id
        JOIN activity_chapters ac
            ON ac.activity_id = c.activity_id
        JOIN chapters ch
            ON ch.id = ac.chapter_id
        WHERE dc.id = %s
        LIMIT 1
        """
        cur = conn.cursor()
        cur.execute(sql, (data_challenge_id,))
        row = cur.fetchone()
        cur.close()

        if not row:
            raise ValueError(f"data_challenge_id {data_challenge_id} não encontrado ou sem capítulo associado.")

        subject_id, grade_id = row[0], row[1]
        return int(grade_id), int(subject_id)
    finally:
        conn.close()
        


def get_data_challenge_answers(data_challenge_id) -> pd.DataFrame:
    """
    Retorna todas as respostas da tabela `data_challenge_answers`
    para um dado `data_challenge_id`, convertendo o parâmetro
    para int nativo antes de enviar ao MySQL.
    """
    # Converte numpy.int64 ou outros para int nativo
    data_challenge_id = int(data_challenge_id)

    
    conn = mysql.connector.connect(**config)

    sql = """
    SELECT *
    FROM data_challenge_answers
    WHERE data_challenge_id = %s
    """
    # params deve ser uma tupla de int nativo
    params = (data_challenge_id,)

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def get_videos(yr: int, sub: int) -> pd.DataFrame:
    """
    Retorna todos os vídeos da disciplina `sub` e dos anos `yr` e `yr-1`,
    usando as tabelas:
      videos            (activity_id →…)
      activity_chapters (activity_id, chapter_id)
      chapters          (id, subject_id, grade_id)

    Filtra:
      * chapters.subject_id IN (sub)
      * chapters.grade_id   IN (yr, yr-1)
      * videos.active = 1
    """
    
    
    conn = mysql.connector.connect(**config)

    sql = """
    SELECT v.*
    FROM videos v
    JOIN activity_chapters ac
      ON ac.activity_id = v.activity_id
    JOIN chapters ch
      ON ch.id = ac.chapter_id
    WHERE ch.subject_id = %(sub)s
      AND ch.grade_id   IN (%(yr)s, %(yr_minus)s)
      AND v.active = 1
    """
    params = {
        'sub': sub,
        'yr': yr,
        'yr_minus': yr - 1
    }

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def get_data_challenge(yr: int, sub: int) -> pd.DataFrame:
    """
    Retorna todos os registros de `data_challenge` filtrados por:
      * disciplina (data_challenge.subject_id = sub)
      * ano escolar (data_students.grade_id = yr)

    Faz JOIN em data_students para obter o `grade_id`.
    """
    
    
    conn = mysql.connector.connect(**config)

    sql = """
    SELECT dc.*
    FROM data_challenges dc
    JOIN data_students ds
      ON ds.id = dc.data_student_id
    WHERE dc.subject_id = %(sub)s
      AND ds.grade_id    = %(yr)s
    """
    params = {'sub': sub, 'yr': yr}

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def get_data_students(yr: int):
    """
    Retorna as linhas da tabela `data_students` para o ano escolar `yr`.

    Parâmetros:
      yr – código do ano escolar (ex: 5)

    Retorna:
      DataFrame com as colunas de `data_students` cujo `year_id` = yr.
    """
    
    
    conn = mysql.connector.connect(**config)

    sql = """
    SELECT *
    FROM data_students
    WHERE grade_id = %(yr)s
    """
    params = {'yr': yr}

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def get_mod(yr: int, sub: int):
    """
    Retorna registros da tabela `chapters` filtrados pelo ano (`grade_id`)
    e pela disciplina (`subject_id`).

    Parâmetros:
      yr  – ano escolar (ex: 5)
      sub – código da disciplina (ex: 2 para Matemática)

    Retorna:
      DataFrame com as colunas de `chapters` que atendem aos filtros.
    """
   
    conn = mysql.connector.connect(**config)

    sql = """
    SELECT *
    FROM chapters
    WHERE subject_id = %(sub)s
      AND grade_id   = %(yr)s
    """
    params = {'yr': yr, 'sub': sub}

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_activity_chapters(yr: int, sub: int) -> pd.DataFrame:
    """
    Retorna registros de activity_chapters para a disciplina `sub`
    e para os anos `yr` e `yr-1`, usando a relação:
      activity_chapters.chapter_id → chapters.id

    Filtra:
      * chapters.subject_id = sub
      * chapters.grade_id IN (yr, yr-1)
    """
    
    
    conn = mysql.connector.connect(**config)

    sql = """
    SELECT ac.*
    FROM activity_chapters ac
    JOIN chapters ch
      ON ch.id = ac.chapter_id
    WHERE ch.subject_id = %(sub)s
      AND ch.grade_id IN (%(yr)s, %(yr_minus)s)
    """
    params = {
        'sub': sub,
        'yr': yr,
        'yr_minus': yr - 1
    }

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def get_challenges(yr: int, sub: int):
    """
    Retorna os desafios (challenges) ativos da disciplina `sub` e dos anos
    `yr` e `yr-1`, a partir das relações:
      challenges.activity_id → activity_chapters.activity_id
      activity_chapters.chapter_id → chapters.id

    Filtra:
      * challenges.active = 1
      * chapters.subject_id = sub
      * chapters.grade_id IN (yr, yr-1)
    """
    
    
    conn = mysql.connector.connect(**config)

    sql = """
    SELECT c.*
    FROM challenges c
    WHERE c.active = 1
      AND EXISTS (
          SELECT 1
          FROM activity_chapters ac
          JOIN chapters ch
            ON ch.id = ac.chapter_id
          WHERE ac.activity_id = c.activity_id
            AND ch.subject_id = %(sub)s
            AND ch.grade_id IN (%(yr)s, %(yr_minus)s)
      )
    """
    params = {
        'sub': sub,
        'yr': yr,
        'yr_minus': yr - 1
    }

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def pega_perguntas_filtradas(ano, materia, sem_imagem=True):
    
    conn = mysql.connector.connect(**config_1)

    # Monta a query com JOIN para excluir as questões que têm imagem
    sql = """
        SELECT q.*
        FROM questions q
        {join_clause}
        WHERE q.year_id = %(ano)s
          AND q.subject_id = %(mat)s
          AND q.active = 1
          {no_img_clause}
    """.format(
        join_clause="LEFT JOIN image_question iq ON iq.question_id = q.id" if sem_imagem else "",
        no_img_clause="AND iq.question_id IS NULL" if sem_imagem else ""
    )

    params = {'ano': ano, 'mat': materia}
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_alternatives(ano: int, materia: int, sem_imagem: bool = True) -> pd.DataFrame:
    """
    Retorna as alternativas das questões do ano e matéria especificados,
    que estão ativas e, se solicitado, que não têm imagem.
    """
    
    
    conn = mysql.connector.connect(**config_1)

    no_img_clause = """
      AND NOT EXISTS (
          SELECT 1
          FROM image_question iq
          WHERE iq.question_id = q.id
      )
    """ if sem_imagem else ""

    sql = f"""
    WITH filtered_q AS (
        SELECT q.id
        FROM questions q
        WHERE q.year_id    = %(ano)s
          AND q.subject_id = %(mat)s
          AND q.active     = 1
          {no_img_clause}
    )
    SELECT a.*
    FROM alternatives a
    JOIN filtered_q fq ON fq.id = a.question_id
    """

    params = {'ano': ano, 'mat': materia}
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_questions_filtered(ano: int, materia: int, sem_imagem: bool = True) -> pd.DataFrame:
    """
    Retorna as questões do ano e matéria especificados, ativas e,
    opcionalmente, sem imagem.
    """

    conn = mysql.connector.connect(**config_1)

    if sem_imagem:
        sql = """
        SELECT q.*
        FROM questions q
        WHERE q.year_id      = %(ano)s
          AND q.subject_id   = %(mat)s
          AND q.active       = 1
          AND NOT EXISTS (
              SELECT 1
              FROM image_question iq
              WHERE iq.question_id = q.id
          )
        """
    else:
        sql = """
        SELECT q.*
        FROM questions q
        WHERE q.year_id    = %(ano)s
          AND q.subject_id = %(mat)s
          AND q.active     = 1
        """

    params = {'ano': ano, 'mat': materia}
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# puxa tabelas do banco básico
def get_data(tabela):
   

    conn = mysql.connector.connect(**config)

    query = "SELECT * FROM {}".format(tabela)
    df = pd.read_sql_query(query, conn)

    # Fechar a conexão com o banco de dados
    conn.close()
    
    return df 


def get_bncc_skills(ano: int) -> pd.DataFrame:
    """
    Retorna as habilidades da BNCC de um dado ano escolar, filtrando apenas:
      * códigos com 3º caractere '0' (para evitar códigos de simulado internos);
      * 4º caractere igual ao ano pedido (ex: '5' em EF05... para quinto ano);
      * prefixo de disciplina 'MA' nas posições 5 e 6 (Matemática).
    Ex: EF05MA23 passa (5º ano, matemática, 3º caractere zero).
    """
    if not (1 <= ano <= 9):
        raise ValueError("Ano deve ser um dígito de 1 a 9 correspondente ao 4º caractere.")

    
    conn = mysql.connector.connect(**config_1)

    sql = """
    SELECT *
    FROM skills
    WHERE CHAR_LENGTH(bncc_code) >= 6
      AND UPPER(SUBSTRING(bncc_code, 3, 1)) = '0'
      AND SUBSTRING(bncc_code, 4, 1) = %(ano_char)s
      AND UPPER(SUBSTRING(bncc_code, 5, 1)) = 'M'
      AND UPPER(SUBSTRING(bncc_code, 6, 1)) = 'A'
    """
    params = {'ano_char': str(ano)}

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


###########################################################
# BLOCO_2 (1021..1055): pré-requisitos (MANTÉM "puxa") ####
###########################################################

# A função que faz a requisição (legado bloco 2)
def puxa(id):
    event = get_module(id)
    return event


TBL_CHAPTERS = "chapters"
TBL_CHAPTER_LINK = "chapter_previous_chapter"


def get_module_primary(module_id: int):
    """Retorna {id, active, year:{id:grade_id}} para o capítulo."""
    module_id = int(module_id)  # evita numpy.int64 no driver
    conn = mysql.connector.connect(**config)
    try:
        sql = f"""
            SELECT m.id, m.grade_id, m.active
            FROM {TBL_CHAPTERS} AS m
            WHERE m.id = %s
            LIMIT 1
        """
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, (module_id,))
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    return {
        "id": int(row["id"]),
        "active": int(row["active"]),
        "year": {"id": int(row["grade_id"])},
    }


def get_module(module_id: int):
    """
    Retorna dict no formato legado:
    {
      "id": <module_id>,
      "active": <0/1>,
      "year": {"id": <grade_id>},
      "previous_modules": [ { "id":..., "active":..., "year": {"id":...} }, ...]
    }
    """
    base = get_module_primary(module_id)
    if not base:
        return None

    module_id = int(module_id)
    conn = mysql.connector.connect(**config)
    try:
        sql = f"""
            SELECT m.id, m.grade_id, m.active
            FROM {TBL_CHAPTER_LINK} AS mp
            JOIN {TBL_CHAPTERS} AS m
              ON m.id = mp.previous_chapter_id
            WHERE mp.chapter_id = %s
        """
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, (module_id,))
        rows = cur.fetchall()
    finally:
        conn.close()

    base["previous_modules"] = [
        {"id": int(r["id"]), "active": int(r["active"]), "year": {"id": int(r["grade_id"])}}
        for r in rows or []
    ]
    return base

