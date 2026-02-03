"""
Created on Mon Set 1 11:37:19 2025

@author: andresardao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# services_features.py

# services_recommender.py
from __future__ import annotations

import io
import os
from collections import Counter
from typing import Any, Dict, List, Tuple
from flask import current_app




import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd

# DB access functions that are intentionally kept in the DB module.
from app_infra_db import (
    get_dc_grade_and_subject,
    get_data_challenge_answers,
    puxa,  # prerequisite graph (legacy Block_2)
    get_main_db_conn,
)

# Feature bundle cache and TABS registry live in Redis.
from app_infra_redis import (
    get_features,  # read-through cache for the feature bundle (yr, sub)
    tabs_get,      # TABS[hab] -> [hab_table.csv, prereq_combo.csv]
)




CLASSIC_COLS = [
    "question_id",
    "alternative_id",
    "error_code",
    "error_name",
    "error_description",
]

CSV_TO_CLASSIC_RENAMES = {
    "id_da_questao": "question_id",
    "id_da_alternativa": "alternative_id",
    "codigo_erro": "error_code",
    "nome_erro": "error_name",
    "Descrição do erro": "error_description",
    # já no padrão
    "question_id": "question_id",
    "alternative_id": "alternative_id",
    "error_code": "error_code",
    "error_name": "error_name",
    "error_description": "error_description",
}


class ErrorTableNotReleased(Exception):
    """Tabela de erros ainda não liberada/ativa na curadoria."""


def _normalize_error_table_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: CSV_TO_CLASSIC_RENAMES.get(c, c) for c in df.columns})

    for c in CLASSIC_COLS:
        if c not in df.columns:
            df[c] = ""

    df = df[CLASSIC_COLS].copy()

    for c in ["error_code", "error_name", "error_description"]:
        df[c] = df[c].fillna("").astype(str)

    return df


def load_error_table_by_title(
    *,
    error_table_title: str,
    db_conn,
    logger=None,
    active_only: bool = True,
    allow_csv_fallback: bool = False,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Produção (recomendado):
      allow_csv_fallback=False  -> se DB não tiver (ou não tiver active=1), lança ErrorTableNotReleased.

    Local/dev:
      allow_csv_fallback=True + csv_path apontando para arquivo (pode ser só filename se estiver no CWD)
      -> tenta DB; se falhar ou vier vazio, tenta CSV.
    """

    # (i) DB
    try:
        where_active = " AND active = 1" if active_only else ""
        sql = f"""
            SELECT
                question_id,
                alternative_id,
                error_code,
                error_name,
                error_description
            FROM error_classifications
            WHERE error_table_title = %s
            {where_active}
        """

        df_db = pd.read_sql(sql, con=db_conn, params=(error_table_title,))
        if df_db is not None and len(df_db) > 0:
            if logger:
                logger.info(
                    "DB.errors OK (title=%s, rows=%s).", error_table_title, len(df_db)
                )
            return _normalize_error_table_df(df_db)

        # The database responded but with no rows => not released (or not active)
        raise LookupError(f"DB.errors 0 rows for title={error_table_title}")

    except Exception as e_db:
        if logger:
            logger.warning(
                "DB.errors indisponivel/ausente (title=%s). Err=%s",
                error_table_title, str(e_db)
            )

        # When in production doesn't drop to CSV: error with curation message.
        if not allow_csv_fallback:
            raise ErrorTableNotReleased(
                f"Tabela '{error_table_title}' ainda não foi liberada/ativa pela curadoria no DB.errors."
            ) from e_db

    # (ii) CSV fallback (somente se allow_csv_fallback=True)
    if allow_csv_fallback:
        if not csv_path:
            # fallback local “automático”: tenta usar o próprio título como path
            csv_path = error_table_title

        try:
            df_csv = pd.read_csv(csv_path)
            if logger:
                logger.info(
                    "CSV fallback OK (title=%s, csv=%s, rows=%s).",
                    error_table_title, csv_path, len(df_csv)
                )
            return _normalize_error_table_df(df_csv)

        except Exception as e_csv:
            if logger:
                logger.error(
                    "CSV fallback falhou (title=%s, csv=%s). Err=%s",
                    error_table_title, csv_path, str(e_csv)
                )
            raise ErrorTableNotReleased(
                f"Tabela '{error_table_title}' ainda não foi liberada/ativa pela curadoria "
                f"(DB.errors indisponível e CSV fallback falhou)."
            ) from e_csv

    # Teoricamente não chega aqui
    raise ErrorTableNotReleased(
        f"Tabela '{error_table_title}' ainda não foi liberada/ativa pela curadoria."
    )



TABS = {
        'EF05MA01': ['classificacao_erros_EF05MA01.csv', 'classificacao_erros_EF04MA01.csv'],
        'EF05MA03': ['classificacao_erros_EF05MA03.csv', 'classificacao_erros_EF04MA09_1.csv'],
        'EF05MA03A': ['classificacao_erros_EF05MA03.csv', 'classificacao_erros_EF04MA09_1.csv'],  
        'EF05MA04': ['classificacao_erros_EF05MA04.csv', 'classificacao_erros_EF04MA09_2.csv'],
        'EF05MA05': ['classificacao_erros_EF05MA05.csv', 'classificacao_erros_EF05MA04.csv'],
        'EF05MA06': ['classificacao_erros_EF05MA06.csv', 'classificacao_erros_EF04MA0910.csv'],
        'EF05MA07': ['classificacao_erros_EF05MA07.csv', 'classificacao_erros_EF04MA10_1.csv'],
        'EF05MA08': ['classificacao_erros_EF05MA08.csv', 'classificacao_erros_EF04MA060710_1.csv'],
        'EF05MA09': ['classificacao_erros_EF05MA09.csv', 'classificacao_erros_EF04MA08.csv'],
        'EF05MA10': ['classificacao_erros_EF05MA10.csv', 'classificacao_erros_EF04MA14.csv'],
        'EF05MA22': ['classificacao_erros_EF05MA22.csv', 'classificacao_erros_EF04MA26_2.csv'],
        'EF05MA23': ['classificacao_erros_EF05MA23.csv', 'classificacao_erros_EF04MA26.csv'],
    }

# MySQL connection configs used by legacy functions in this module.
# They mirror the original monolithic script to minimize behavioral drift.
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "Mandy")
DB_NAME_1 = os.getenv("DB_NAME_1", "perguntas_schema")

config = {"user": DB_USER, "password": DB_PASS, "host": DB_HOST, "database": DB_NAME}
config_1 = {"user": DB_USER, "password": DB_PASS, "host": DB_HOST, "database": DB_NAME_1}


def dict_maker(list1, list2):
    dct = {}
    for k, v in zip(list1, list2):
        dct[k] = v
    return dct


def dict_inverter(dct):
    dct_inv = {}
    for k, v in dct.items():
        dct_inv[v] = k
    return dct_inv


def double_value_inverter(dct):
    dct_inv = {}
    for k, v in dct.items():
        if isinstance(v, list) and len(v) == 2:
            dct_inv[v[0]] = k
            dct_inv[v[1]] = k
    return dct_inv


def check_unique(lista):
    return len(set(lista)) == len(lista)


# First function to act once a data_challenge_id
def get_student_and_chall(chall_dta_id, feats):
    df_challenges = feats['df_challenges']
    a = list(df_challenges[df_challenges['id'] == chall_dta_id]['data_student_id'].values)
    b = list(df_challenges[df_challenges['id'] == chall_dta_id]['challenge_id'].values)
    return a[0], b[0]

# Receives a data_student and returns related student id.
def devolve_student_id(dta_student, feats):
    df_data_students = feats['df_data_students']
    df_student_id = df_data_students.loc[df_data_students['id'] == dta_student]
    listenha = list(df_student_id["student_id"].values)
    return listenha[0]


def from_student_get_challenges(stu_dta_id, feats, sub):
    df_challenges = feats['df_challenges']
    df = df_challenges[df_challenges['data_student_id'] == stu_dta_id]
    df = df[df['subject_id'] == sub]
    lista_data_challenges = list(df['id'].values)
    lista_challenges = list(df['challenge_id'].values)
    lista_de_notas = list(df['performance'].values)
    return lista_data_challenges, lista_challenges, lista_de_notas 


def get_errors_completo(data_chall_id):
    erradas = {}
    certas = {} 
    tempo_erradas = {}
    tempo_certas = {}
    
    df_aux = get_data_challenge_answers(data_chall_id)
    qts = list(df_aux['question_id'].values)
    crt = list(df_aux['correct'].values)
    aws = list(df_aux['answer'].values)
    tme = list(df_aux['time'].values)
    for q,r,a,t in zip(qts,crt,aws,tme):
        
        if r == 0:
            erradas[q] = a
            tempo_erradas[q] = t
        else:
            certas[q] = a
            tempo_certas[q] = t
        
    return erradas, certas, tempo_erradas, tempo_certas


def get_errors_completo_aws(data_chall_id):
    erradas = {}
    certas = {} 
    tempo_erradas = {}
    tempo_certas = {}
    
    df_aux = get_data_challenge_answers(data_chall_id)
    qts = list(df_aux['question_id'].values)
    crt = list(df_aux['correct'].values)
    aws = list(df_aux['answer'].values)
    tme = list(df_aux['time'].values)
    for q,r,a,t in zip(qts,crt,aws,tme):
        
        if r == 0:
            erradas[q] = a
            tempo_erradas[q] = t
        else:
            certas[q] = a
            tempo_certas[q] = t
        
    return erradas



"""
    Classifica quais erros são sistemáticos com base na frequência dos erros.
    
    Critérios sugeridos:
    - (1) Se o total de erros for 3: Considera sistemático se houver apenas um tipo de erro.
    
    - (2) Se o total de erros for 4 ou 5: Considera sistemático se algum tipo de erro representar, por exemplo, 60% ou mais dos erros.
    - (3) Se o total de erros for maior que 5: Considera sistemático se algum tipo de erro ocorrer em 40% ou mais do total.
    
    Parâmetros:error_counts (dict): Dicionário com a distribuição dos erros. Ex.: {'Erro de composição': 5, 'Erro Posicional': 2, ...}
                             
    Retorna:
        systematic_errors (list): Lista dos tipos de erro classificados como sistemáticos.
"""
def classifica_erros_sistematicos(error_counts):

    total = sum(error_counts.values())
    systematic_errors = []
    
    if total == 3:
        # Se o aluno cometeu exatamente 3 erros,
        # o único erro sistemático ocorre somente se houver apenas um tipo de erro.
        if len(error_counts) == 1:
            systematic_errors = list(error_counts.keys())
    elif total in [4, 5]:
        # Se o total for 4 ou 5, usamos um limiar de 60%
        for error, count in error_counts.items():
            if count / total >= 0.6:
                systematic_errors.append(error)
    else:
        # Para total de erros maior que 5, usamos um limiar de 40%
        for error, count in error_counts.items():
            if count / total >= 0.4:
                systematic_errors.append(error)
                
    return systematic_errors


def gena_relatorios_template(*args, **kwargs):
    return {}


def plota_erros_pdf(error_counts) -> bytes:
    fig = plt.figure()
    plt.bar(list(error_counts.keys()), list(error_counts.values()))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def gena_relatorios_template_grafico(error_counts: dict,
                              erros_sist: list,
                              vid: bool,
                              anim: bool):
    """
    Gera três micro-relatórios (aluno, professor, pais) e um PDF do gráfico de erros.

    Parâmetros:
        error_counts (dict): contagens absolutas de erros.
        erros_sist (list): lista de erros dominantes.
        vid (bool): há vídeos disponíveis?
        anim (bool): há animações disponíveis?

    Retorna:
        tuple:
          - reports (dict): {'student': str, 'teacher': str, 'parent': str}
          - chart_pdf (bytes): bytes do PDF gerado pelo plota_erros.
    """
    # --- Step 1: to make a text ---
    total = sum(error_counts.values()) or 1
    distribution = ", ".join(
        f"{err}: {count/total:.0%}" for err, count in error_counts.items()
    )
    dominant = ", ".join(erros_sist) if erros_sist else "Nenhum"
    vid_str = "📺 Vídeos disponíveis." if vid else "Sem vídeos recomendados."
    anim_str = "🎞️ Animações disponíveis." if anim else "Sem animações recomendadas."

    # Micro-relatório para o aluno
    student = (
        f"📊 Progresso no tópico\n"
        f"Distribuição de erros: {distribution}\n"
        f"Erro principal: {dominant}\n"
        f"{vid_str} {anim_str}".strip()
    )

    # Micro-relatório para o professor
    teacher = (
        f"👩‍🏫 Relatório Turma\n"
        f"Distribuição de erros: {distribution}\n"
        f"Erros sistemáticos: {dominant}\n"
        f"{vid_str} {anim_str}".strip()
    )

    # Micro-relatório para os pais
    parent = (
        f"📌 Evolução do aluno\n"
        f"Distribuição de erros: {distribution}\n"
        f"Principal dificuldade: {dominant}\n"
        f"{vid_str} {anim_str}".strip()
    )

    reports = {
        "student": student,
        "teacher": teacher,
        "parent": parent
    }

    # --- Step 2: A PDF of the visuals ---
    #chart_pdf = plota_erros_pdf(error_counts)

    #return reports, chart_pdf
    return reports


def destaca_bncc_legado(erros_bncc: List[str]) -> List[str]:
    out = []
    for x in erros_bncc:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def from_chall_2_mod(data_challenge_id):
    conn = mysql.connector.connect(**config)
    sql = """
    SELECT DISTINCT m.id AS module_id
    FROM data_challenges dc
    JOIN challenges c ON c.data_challenge_id = dc.id
    JOIN activity_chapters ac ON ac.activity_id = c.activity_id
    JOIN chapters ch ON ch.id = ac.chapter_id
    JOIN modules m ON m.chapter_id = ch.id
    WHERE dc.id = %s
      AND m.deleted_at IS NULL
    """
    df = pd.read_sql_query(sql, conn, params=(int(data_challenge_id),))
    conn.close()
    if df.empty:
        return []
    return [int(x) for x in df["module_id"].tolist()]


def mod_to_chall(module_id):
    conn = mysql.connector.connect(**config)
    sql = """
    SELECT DISTINCT c.id AS challenge_id
    FROM challenges c
    JOIN activity_chapters ac ON ac.activity_id = c.activity_id
    JOIN chapters ch ON ch.id = ac.chapter_id
    JOIN modules m ON m.chapter_id = ch.id
    WHERE m.id = %s
    """
    df = pd.read_sql_query(sql, conn, params=(int(module_id),))
    conn.close()
    if df.empty:
        return []
    return [int(x) for x in df["challenge_id"].tolist()]


def from_chall_2_mod_4(data_challenge_id):
    return from_chall_2_mod(data_challenge_id)


def mod_to_chall_4(module_id):
    return mod_to_chall(module_id)


def skillid_2_videoid0(skill_id, vid_aux, hab_2_chapter):
    if skill_id not in hab_2_chapter:
        return None
    chapter_id = hab_2_chapter[skill_id]
    if chapter_id in vid_aux:
        return vid_aux[chapter_id]
    return None


def skillid_2_videoid(hb, feats):
    # Pega o capítulo correspondente à skill
    chap = feats['hab_2_chapter'].get(hb)
    if chap is None:
        return "Habilidade não mapeada em hab_2_chapter"

    # Pega a "inversa forçada" (lista ou NaN)
    raw = feats['chap_2_acttt'].get(chap, np.nan)

    # se for escalar (float) e NaN -> não há vídeos
    if isinstance(raw, float) and np.isnan(raw):
        return "Atualmente este capítulo não possui vídeos"

    # Montra a lista de activities
    # se raw for lista/array, usá-la direta;
    # se raw for um único número, colocá-lo numa lista
    if isinstance(raw, (list, np.ndarray)):
        lista_aux = list(raw)
    else:
        # pode ser um float (não-NaN) ou int
        lista_aux = [int(raw)]

    # Mapeamento cada activity ao vídeo
    video_s = []
    for x in lista_aux:
        if x in feats['activity_aux']:             # activity_aux é a lista de activities válidas
            video_s.append(feats['activity_2_vid'][x])  # activity_2_vid mapeia activity -> video

    # dando o resultado
    if video_s:
        return video_s
    else:
        return "Atualmente este capítulo não possui vídeos"




def back_bone(data_challenge, feats, yr, sub):

    dta_chall = data_challenge
    
    # Pega o aluno e o desafio correspondente
    dta_student, chall = get_student_and_chall(dta_chall, feats)
    
    std_id  = devolve_student_id(dta_student, feats)
    
    df_challenges = feats['df_challenges']
    
    # Filtra os data students relacionados ao desafio mãe
    df_chall = df_challenges[(df_challenges['challenge_id'] == chall)]
    
    # Pega as ações ligadas ao aluno: todos os challenges e notas daquele aluno 
    todos_dta_challs, todos_challs, todas_notas = from_student_get_challenges(dta_student, feats, sub)
    
    # Pega o id da habilidade principal (mãe)
    from_chall_2_skill = feats['from_chall_2_skill']
    hab_mae_id = from_chall_2_skill[chall]
    
    # Pega a bncc em si
    BNCC_5 = feats['BNCC_5']
    hab_mae = BNCC_5[hab_mae_id]
    #print(hab_mae, "HAB_MAE")
    
    # O nome da hbilidade mãe não é colocado no mesmo saco das outras habilidades
    csv_1 = hab_mae
    
    # O 'saco' onde colocamos o código BNCC das habilidades pré-requisito
    habs_para_csv = []
    
    # Peda o módulo em si
    hab_2_mod_5 = feats['hab_2_mod_5']
    if yr == 5:
        hab_2_mod_5['EF05MA01'] = 20169
        
    mod_mae = hab_2_mod_5[hab_mae]
    
    aux = puxa(mod_mae)
    
    aux_1 = aux['previous_modules']
    
    #print('ALCE: ',hab_2_mod_5)
    
    
    prev_mods_4 = []
    
    prev_mods_5 = []
    
    if len(aux_1) != 0:
        for dct in aux_1:
            if ((dct['year']['id'] == 5) and (dct['active'] == 1)):
                prev_mods_5.append(dct['id'])
            if ((dct['year']['id'] == 4) and (dct['active'] == 1)):
                prev_mods_4.append(dct['id'])
                
    else: 
        
        return {"response": "Este módulo não tem pré-requisito."},'',''
        
    
    aux_mod_id_5 = feats['aux_mod_id_5']
    mod_2_hab_5 = feats['mod_2_hab_5']
    BNCC_5 = feats['BNCC_5']
    
    aux_mod_id_4 = feats['aux_mod_id_4']
    mod_2_hab_4 = feats['mod_2_hab_4']
    BNCC_4 = feats['BNCC_4']
    aux_habs_ids = feats['aux_habs_ids']
    aux_habs_bncc = feats['aux_habs_bncc']
    

    
    prev_habs_5 = []
    
    prev_habs_4 = []
    
    ids_5 = []
    
    ids_4 = []
    
    if len(prev_mods_5) != 0:
        for mod in prev_mods_5:
            if (mod in aux_mod_id_5):
                ids_5.append(mod_2_hab_5[mod])
                prev_habs_5.append(BNCC_5[mod_2_hab_5[mod]])
                habs_para_csv.append(BNCC_5[mod_2_hab_5[mod]])
            
    if len(prev_mods_4) != 0:
        for mod in prev_mods_4:
            if (mod in aux_mod_id_4):
                ids_4.append(mod_2_hab_4[mod])
                prev_habs_4.append(BNCC_4[mod_2_hab_4[mod]])
                habs_para_csv.append(BNCC_4[mod_2_hab_4[mod]])
                            
    recommend = {}
    
    vds_5 = []
    vds_4 = []
    

    if hab_mae_id in feats['aux_skill_id']:
        vds_5.append(skillid_2_videoid(hab_mae_id,feats))
    
    
    if len(ids_5) != 0:
        for ide in ids_5:
            #if ide in aux_skill_id:
            vds_5.append(skillid_2_videoid(ide, feats)) 
    
    if len(ids_4) != 0:
        for ide in ids_4:
            #if ide in aux_skill_id:
            vds_4.append(skillid_2_videoid(ide, feats)) 
            
    recommend["vídeos_quinto_ano"] = vds_5
    recommend["vídeos_quarto_ano"] = vds_4
    
    challs_5 = []
    challs_4 = []
    
    from_skill_2_chall = feats['from_skill_2_chall']
    
    if len(ids_5) != 0:
        for hab in ids_5:
            challs_5.append(from_skill_2_chall[hab])
            
    if len(ids_4) != 0:
        for hab in ids_4:
            challs_4.append(from_skill_2_chall[hab]) 
    
    df_data_students = feats['df_data_students']
    
    # Descobrindo todos os data_student com aquele student_id
    df_student_id1 = df_data_students.loc[df_data_students['student_id'] == std_id]
    listra = list(df_student_id1['id'].values)
    
    dta_stu_ids = []
    
    # Achando os data_students correspondentes aos students id's encontrados acima.
    if len(challs_5) != 0:
        print("UEPA5")
        for ch in challs_5:
            df_ch = df_challenges[(df_challenges['challenge_id'] == ch)]
            lst_aux = list(df_ch['data_student_id'].values)
            for l in listra:
                if l in lst_aux:
                    if l != dta_student:
                        dta_stu_ids.append(l)
                
    if len(challs_4) != 0:
        for ch in challs_4:
            df_ch = df_challenges[(df_challenges['challenge_id'] == ch)]
            lst_aux = list(df_ch['data_student_id'].values)
            #print("LISTOMUI4: ",lst_aux)
            for l in listra:
                print("L")
                if l in lst_aux:
                    #print("UEPA4")
                    if l != dta_student:
                        dta_stu_ids.append(l)
                        

    # Using the data_student_id obtained from student_id, it is possible to retrieve the actions associated with the student: all the challenges and grades for that student.
    for ide in dta_stu_ids:
            todos_dta_challs_1, todos_challs_1, todas_notas_1 = from_student_get_challenges(ide, feats)
            todos_dta_challs = list(set(todos_dta_challs + todos_dta_challs_1))
            todos_challs = list(set(todos_challs + todos_challs_1))
            todas_notas = list(set(todas_notas + todas_notas_1))
            
     
    dcto = dict_maker(todos_challs, todos_dta_challs)
    
    erros_hab_mae = {}
    
    erros_hab_mae = get_errors_completo_aws(dcto[chall]) 
    
   
   
    
    resp_erradas_5 = {}
    resp_erradas_4 = {}
    
    if len(challs_5) != 0:
        for cha in challs_5:
            if cha in dcto.keys():
                resp_erradas_5[cha] = get_errors_completo_aws(dcto[cha])                
                
    if len(challs_4) != 0:       
        for cha in challs_4:
            if cha in dcto.keys():
                #print("foi2: ",dcto[cha])
                resp_erradas_4[cha] = get_errors_completo_aws(dcto[cha])

    erros_hab_filha, acertos_hab_filha, tempo_resp_err, tempo_resp_acerto = get_errors_completo(dcto[chall])
    
    re_er_5 = []
    re_er_4 = []
    if len(resp_erradas_5) != 0: 
        for dcio in resp_erradas_5.values():
            re_er_5.append(list(dcio.values()))
            
    if len(resp_erradas_4) != 0: 
        for dcio in resp_erradas_4.values():
            re_er_4.append(list(dcio.values()))
        
    soh_as_resp_erradas_4 = []
    soh_as_resp_erradas_5 = []
    
    if len(re_er_5) != 0:
        for lst in re_er_5:
            soh_as_resp_erradas_5 = soh_as_resp_erradas_5 + lst

    
    if len(re_er_4) != 0:
        for lst in re_er_4:
            soh_as_resp_erradas_4 = soh_as_resp_erradas_4 + lst
            
    x = list(erros_hab_mae.values())
    y = soh_as_resp_erradas_5
    z = soh_as_resp_erradas_4
    
    
   
    
    todas_resps_erradas = x + y + z
    
    recommend["questoes"] = todas_resps_erradas
    
    # Considerando passar as habs pra csv em uma lista NOPE
    return recommend,csv_1,habs_para_csv 


# Função recebe o id do desafio em que o aluno apresentou resultados insatisfatório junto com.../
# /... as tabelas de erros sistemáticos da habilidade atual e de seu pré-requisito direto.../
# /... identifica quais são os erros sistemáticos cometidos pelo aluno.../
# /... devolve uma recomendação de questões a serem feitas.
# Próximo passo imediato: adicionar um LLM que produza um relatório em português, baseado em um template.../
# /... que explique ao aluno, ou professor, o estatus atual das lacunas de conhecimento do aluno

def recommender(data_challenge):
    erros_cods = []
    erros = []

    # Derivates (yr, sub) from data_challenge
    yr, sub = get_dc_grade_and_subject(data_challenge)

    # Tries cache first, if missed, rebuilds it and saves
    feats = get_features(yr, sub)


    # Classification tables (fix: chave sem espaço)
    tabelas_nomes = TABS
    
    # ATTENTION: once the table "error_tables_relations" is ready to go, use this instead:
    #tabelas_nomes = feats['TABS']

    # Backbone gets feats explicitly 
    erros_1, nome_hab_mae, habs_pre = back_bone(data_challenge, feats, yr, sub)
    
    # É um dicionário que contém as tabelas de erros
    hab_mae = nome_hab_mae
    
    if hab_mae not in tabelas_nomes.keys():
        return {"KeyError": "Este sistema de recomendações ainda não pode ser usado para este capítulo {}".format(hab_mae)}
    
    title_hab, title_combo = tabelas_nomes[hab_mae]  # ["classificacao_erros_EF0XMAYZ.csv", "classificacao_erros_<COMBO_PRE>.csv"]

    
    df_tabela_de_erros_hab_atual = load_error_table_by_title(
        error_table_title=title_hab,
        db_conn=get_main_db_conn(),
        logger=current_app.logger,
        active_only=True,
        allow_csv_fallback=True,   # local
        csv_path=title_hab,        # filename in the same dir
            )

    '''
    for nm in habs_pre:
        print('DFDFDFDF')
        tabs_erros = tabelas_nomes[nm]
        DFs[tabs_erros] = pd.read_csv(tabs_erros)
    print("UBA: ", DFs)
    '''
    
    # pré-req error tables
    
    tabs_erros = load_error_table_by_title(
                error_table_title=title_combo,
                db_conn=get_main_db_conn(),
                logger=current_app.logger,
                active_only=True,
                allow_csv_fallback=True,   # local
                csv_path=title_combo,      # filename in the same dir
            )
    
    for x in erros_1["questoes"]:
        x = int(x)
        erros.append(x)
    
    aux_list_atual = list(df_tabela_de_erros_hab_atual["alternative_id"].values)
    aux_list_pre = list(tabs_erros["alternative_id"].values)

    
    for er in erros:
        if er in aux_list_atual:
            aux_1 = list(df_tabela_de_erros_hab_atual[df_tabela_de_erros_hab_atual["alternative_id"] == er]["error_name"].values)
            if len(aux_1) != 0:
                erros_cods.append(aux_1[0])
                
    for er in erros:
        if er in aux_list_pre:
            aux_2 = list(tabs_erros[tabs_erros["alternative_id"] == er]["error_name"].values)
            if len(aux_2) != 0:
                for e in aux_2:
                    erros_cods.append(e)
    
    
    erros_cont = Counter(erros_cods)
    
    # Function that tests frequency
    erros_sist = classifica_erros_sistematicos(erros_cont)
    
    #print("Erros Sistemáticos: ",erros_sist)
    
    recommendations_1 = []
    recommendations_2 = []
    
    for err in erros_sist:
    
        df_reco_1 = df_tabela_de_erros_hab_atual[df_tabela_de_erros_hab_atual["error_name"] == err]
        recommendations_1 = recommendations_1 + list(set(list(df_reco_1["question_id"].values)))
        
    for err in erros_sist:
        
        df_reco_2 = tabs_erros[tabs_erros["error_name"] == err]
        recommendations_2 = recommendations_2 + list(set(list(df_reco_2["question_id"].values)))
                
    recommendations = list(recommendations_1) + list(recommendations_2)               

    erros_1["questoes"] = recommendations
    
    vid = False
    anim = False
    
    if (len(erros_1['vídeos_quinto_ano']) != 0) or (len(erros_1['vídeos_quarto_ano']) != 0):
        vid = True
    
        
    #reports, pdf_bytes = gena_relatorios_template_grafico(erros_cont, erros_sist, vid, anim)
    reports = gena_relatorios_template_grafico(erros_cont, erros_sist, vid, anim)
    
    erros_1['relatório aluno'] = reports['student']
    erros_1['relatório professor'] = reports['teacher']
    erros_1['relatório pais'] = reports['parent']
    
    #erros_1['gráfico'] = pdf_bytes
    
    return erros_1

