"""
Created on Mon Set 1 11:32:11 2025

@author: andresardao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# services_features.py
from __future__ import annotations

import re
import copy
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


from app_infra_db import (
get_error_tables_relations,
get_dc_grade_and_subject,
get_data_challenge_answers,
get_videos,
get_data_challenge,
get_data_students,
get_mod,
get_activity_chapters,
get_challenges,
pega_perguntas_filtradas,
get_alternatives,
get_questions_filtered,
get_data,
get_bncc_skills,
)

# BAgaça para inverter dicionários que têm valores repetidos em chaves diferentes.
def double_value_inverter(dicto):
    lista_aux = copy.deepcopy(list(dicto.keys()))
    dictoo = {}
    for k in dicto.keys():
        if k in lista_aux:
            lista_aux.remove(k)
        for ke in lista_aux:
            if dicto[k] == dicto[ke]:
                dictoo[dicto[k]] = [k,ke]
                lista_aux.remove(ke)
        if dicto[k] not in dictoo.keys():
            dictoo[dicto[k]] = [k]
    return dictoo

def check_unique(lista):
    aux = list(set(lista))
    aux_remov = lista.copy()
    for x in aux:
        aux_remov.remove(x)
    return aux_remov

def limpa_html(html_string):
    if html_string is None or (hasattr(html_string, "__float__") and pd.isna(html_string)):
        return ""
    s = str(html_string).strip()
    if "<" not in s and ">" not in s:
        return s  # não parece HTML; devolve como texto puro
    soup = BeautifulSoup(s, "html.parser")
    texto_limpo = soup.get_text(separator=" ", strip=True)

    return texto_limpo


# Vira de LaTex em linguagem de gente
def decode_latex(texto):
    # Check if any LaTeX in the string (assumindo casos de LaTeX que tenham barras invertidas \\)
    if isinstance(texto, str) and '\\' in texto:
        # Padrão \frac{num}{den}
        pattern_frac = re.compile(r'(\\d*)\\frac\{(\\d+)\}\{(\\d+)\}')

        def converter_parte_latex(match):
            parte_inteira = match.group(1)
            num = match.group(2)
            den = match.group(3)
            if parte_inteira:
                return f"{parte_inteira}({num}/{den})"
            return f"({num}/{den})"

        # Replacing all LaTeX in the text
        texto_convertido = re.sub(pattern_frac, converter_parte_latex, texto)

        # Other LaTeX patterns can be added here as needed
        
        return texto_convertido
    return texto  

def dict_maker(list1, list2):
    dct = {}
    for k, v in zip(list1, list2):
        dct[k] = v
    return dct

def destaca_bncc(texto):
    # protects against None/NaN and guarantees string
    if texto is None:
        return ""
    try:
        import pandas as pd
        if hasattr(texto, "__float__") and pd.isna(texto):
            return ""
    except Exception:
        pass

    s = str(texto)
    if len(s) <= 8:
        # if it is too short: cant check s[8]; returns straight away
        return s
    # Safe ot access s[8]
    return s[:8] if s[8] in (" ", "\t") else s[:9]

def _gera_todos_os_objetos(yr, sub):

    """
    Here is the place where the bulk of all important objects is built.
    """

    
    # Collecting and preprocessing of base objects (dataframes and lists)
    


    # Optimized queries
    # Genarating objects
    
    TABS = get_error_tables_relations()
    
    df_questions_current = get_questions_filtered(yr, sub, sem_imagem=True)
    df_questions_pre = pega_perguntas_filtradas(yr-1, sub, sem_imagem=True)
    
    df_alternatives_current = get_alternatives(yr, sub,True)
    df_alternatives_pre = get_alternatives(yr-1, sub,True)
    
    # Current year questios DF cleanup
    df_questions_current['title'] = df_questions_current['title'].apply(limpa_html)
    df_questions_current['command'] = df_questions_current['command'].apply(limpa_html)
    df_questions_current['title'] = df_questions_current['title'].apply(decode_latex)
    df_questions_current['support'] = df_questions_current['support'].apply(decode_latex)
    df_questions_current['command'] = df_questions_current['command'].apply(decode_latex)
    
    # Previous year questios DF cleanup
    df_questions_pre['title'] = df_questions_pre['title'].apply(limpa_html)
    df_questions_pre['support'] = df_questions_pre['support'].apply(limpa_html)
    df_questions_pre['command'] = df_questions_pre['command'].apply(limpa_html)
    df_questions_pre['title'] = df_questions_pre['title'].apply(decode_latex)
    df_questions_pre['support'] = df_questions_pre['support'].apply(decode_latex)
    df_questions_pre['command'] = df_questions_pre['command'].apply(decode_latex)
    
    df_alternatives_current['content'] = df_alternatives_current['content'].apply(limpa_html)
    df_alternatives_current['content'] = df_alternatives_current['content'].apply(decode_latex)
    
    df_alternatives_pre['content'] = df_alternatives_pre['content'].apply(limpa_html)
    df_alternatives_pre['content'] = df_alternatives_pre['content'].apply(decode_latex)
    
    # List making: current year
    aux_alt_id_5 = list(df_alternatives_current['question_id'].values)
    aux_alt_5 = list(df_alternatives_current['content'].values)
    aux_alt_correct_5 = list(df_alternatives_current['correct'].values)
    aux_alt_id_das_alt_5 = list(df_alternatives_current['id'].values)

    aux_interess_5_id = list(df_questions_current['id'].values)
    aux_interess_5_title = list(df_questions_current['title'].values)
    aux_interess_5_support = list(df_questions_current['support'].values)
    aux_interess_5_command = list(df_questions_current['command'].values)
    aux_interess_5_ano = list(df_questions_current['year_id'].values)
    aux_interess_5_hab = list(df_questions_current['skill_id'].values)

    aux_alt_id_teste_5 = list(df_alternatives_current['id'].values)
    
    aux_interess_5_hab.append(200)
    
    # List making: previous year
    aux_alt_id_4 = list(df_alternatives_pre['question_id'].values)
    aux_alt_4 = list(df_alternatives_pre['content'].values)
    aux_alt_correct_4 = list(df_alternatives_pre['correct'].values)
    aux_alt_id_das_alt_4 = list(df_alternatives_pre['id'].values)


    aux_interess_4_id = list(df_questions_pre['id'].values)
    aux_interess_4_title = list(df_questions_pre['title'].values)
    aux_interess_4_support = list(df_questions_pre['support'].values)
    aux_interess_4_command = list(df_questions_pre['command'].values)
    aux_interess_4_ano = list(df_questions_pre['year_id'].values)
    aux_interess_4_hab = list(df_questions_pre['skill_id'].values)

    aux_alt_id_teste_4 = list(df_alternatives_pre['id'].values)
    
    # DF buildup (skills case)
    df_skills_current = get_bncc_skills(yr)
    df_skills_pre = get_bncc_skills(yr-1)
    df_skills = pd.concat([df_skills_current,df_skills_pre])
    df_skills = df_skills.loc[df_skills['active'] == 1]
    
    # lists auxiliaries to df_skills
    aux_habs_ids = list(df_skills['id'].values)
    aux_habs_bncc = list(df_skills['bncc_code'].values)

    BNCC_5 = {}
    for ide, hab in zip(aux_habs_ids, aux_habs_bncc):
        if ide in aux_interess_5_hab:
            BNCC_5[ide] = hab

    BNCC_5[193] = 'EF05MA02'

    BNCC_5_inv = {}
    for ide, hab in zip(aux_habs_ids, aux_habs_bncc):
        if ide in aux_interess_5_hab:
            BNCC_5_inv[hab] = ide

    BNCC_5_inv['EF05MA02'] = 193

    bncc_5 = list(BNCC_5.values())
    
    BNCC_4 = {}
    for ide, hab in zip(aux_habs_ids,aux_habs_bncc):
        if ide in aux_interess_4_hab:
            BNCC_4[ide] = hab

    BNCC_4[155] = 'EF04MA09'

    bncc_4 = list(BNCC_4.values())
    BNCC_4_inv = {}
    for ide, hab in zip(aux_habs_ids,aux_habs_bncc):
        if ide in aux_interess_4_hab:
            BNCC_4_inv[hab] = ide
    BNCC_4_inv['EF04MA09'] = 155

   
   
   # DFs concerning pure challenges
    df_challs = get_challenges(yr, sub)
    df_challs = df_challs.loc[df_challs['active'] == 1]
    
    # Aux list
    challs_act = list(df_challs['id'].values)
    
    df_activity_chapters = get_data('activity_chapters')
    
    # Chapters
    df_mod_current = get_mod(yr, sub)
    #df_mod_current = df_mod_current.loc[df_mod_current['active'] == 1]
    df_mod_pre = get_mod(yr-1, sub)
    #df_mod_pre = df_mod_pre.loc[df_mod_pre['active'] == 1]
    df_mod = pd.concat([df_mod_current,df_mod_pre])
    df_mod_mat_5 = df_mod_current
    df_mod_mat_4 = df_mod_pre
    
    df_mod_act = df_mod.loc[df_mod['active'] == 1]
    df_mod_mat_5_act = df_mod_mat_5.loc[df_mod_mat_5['active'] == 1]
    df_mod_mat_4_act = df_mod_mat_4.loc[df_mod_mat_4['active'] == 1]
    
    # Generating aux objects relating challenges and chapters
    aux_chall = list(df_challs['id'].values)
    aux_acttiv = list(df_challs['activity_id'].values)

    chall_2_activity = dict_maker(aux_chall, aux_acttiv)

    aux_actt = list(df_activity_chapters['activity_id'].values)
    aux_chap = list(df_activity_chapters['chapter_id'].values)

    activity_2_chapter = dict_maker(aux_actt, aux_chap)

    aux_chap_1 = list(df_mod['id'].values)
    aux_hab = list(df_mod['skill_id'].values)

    aux_chap_1_5 = list(df_mod_mat_5['id'].values)
    aux_hab_mat_5 = list(df_mod_mat_5['skill_id'].values)

    aux_chap_1_4 = list(df_mod_mat_4['id'].values)
    aux_hab_mat_4 = list(df_mod_mat_4['skill_id'].values)
    
    aux_chap_1_act_5 = list(df_mod_mat_5_act['id'].values)
    aux_hab_act_5 = list(df_mod_mat_5_act['skill_id'].values)
    
    aux_chap_1_act_4 = list(df_mod_mat_4_act['id'].values)
    aux_hab_act_4 = list(df_mod_mat_4_act['skill_id'].values)

    chapter_2_hab = dict_maker(aux_chap_1, aux_hab)

    from_chall_2_skill = {}

    for ch in aux_chall:
        if not np.isnan(chapter_2_hab[activity_2_chapter[chall_2_activity[ch]]]):
            from_chall_2_skill[ch] = int(chapter_2_hab[activity_2_chapter[chall_2_activity[ch]]])

    # Creating more aux mapping dictonaries        
    mod_2_hab_5_ = dict_maker(aux_chap_1_5, aux_hab_mat_5)
    mod_2_hab_4_ = dict_maker(aux_chap_1_4, aux_hab_mat_4)

    hab_2_mod_5_ = dict_maker(aux_hab_mat_5, aux_chap_1_5)
    hab_2_mod_4_ = dict_maker(aux_hab_mat_4, aux_chap_1_4)

    mod_2_hab_5 = {k: v for k, v in mod_2_hab_5_.items() if not np.isnan(v)}
    mod_2_hab_4 = {k: v for k, v in mod_2_hab_4_.items() if not np.isnan(v)}
    

    hab_2_mod_5_ = {k: v for k, v in hab_2_mod_5_.items() if not np.isnan(v)}
    hab_2_mod_4_ = {k: v for k, v in hab_2_mod_4_.items() if not np.isnan(v)}
    
    hab_2_mod_5_act = dict_maker(aux_hab_act_5, aux_chap_1_act_5)
    hab_2_mod_4_act = dict_maker(aux_hab_act_4, aux_chap_1_act_4)

    # Due cleanup followed by invetion of the from_chall_2_skill dict  
    oie = list(from_chall_2_skill.values())
    opa = check_unique(oie)

    retem = []
    for ide in opa:
        dfe = df_skills[df_skills['id']== ide]
        listy = list(dfe['bncc_code'].values)
        if len(listy) != 0:
            retem.append(listy[0])

    cimiterio = []
    for k in from_chall_2_skill.keys():
        a = from_chall_2_skill[k]
        if a in opa:
            cimiterio.append(k)

    for k in cimiterio:
        del from_chall_2_skill[k]   

    from_skill_2_chall = dict_maker(from_chall_2_skill.values(),from_chall_2_skill.keys())

    from_skill_2_chall[155] = 201068
    
    
    # Inicializing a few more DFs (with due minimal queries in each case)
    df_data_students_current = get_data_students(yr)

    df_data_students_pre = get_data_students(yr-1)

    df_data_students = pd.concat([df_data_students_current,df_data_students_pre])

    df_challenges = get_data_challenge(yr, sub)
    

    aux_data_challenge_id = list(df_challenges['id'].values)

    aux_challenge_id = list(df_challenges['challenge_id'].values)
    
    # Maps from data_challenge_id to challenge_id
    from_data_2_challs = dict_maker(aux_data_challenge_id, aux_challenge_id)
    
    
    df_mod_math_5_act = df_mod_current
    df_mod_math_4_act = df_mod_pre

    
    # Getting rid of the "simulados" 
    df_mod_math_5_act = df_mod_math_5_act.drop(df_mod_math_5_act[df_mod_math_5_act.id == 20729].index)
    df_mod_math_5_act = df_mod_math_5_act.drop(df_mod_math_5_act[df_mod_math_5_act.id == 20747].index)
    ds = list(df_mod_math_5_act['description'].values)
    idex = list(df_mod_math_5_act.index)

    ds_4 = list(df_mod_math_4_act['description'].values)
    idex_4 = list(df_mod_math_4_act.index)

    #df_mod_math_5_act['description'] = df_mod_math_5_act['description'].apply(destaca_bncc)
    df_mod_math_5_act['description'] = (df_mod_math_5_act['description'].fillna('').astype(str).map(destaca_bncc))
    
    #df_mod_math_4_act['description'] = df_mod_math_4_act['description'].apply(destaca_bncc)
    df_mod_math_4_act['description'] = (df_mod_math_4_act['description'].fillna('').astype(str).map(destaca_bncc))

    aux_mod_description_5 = list(df_mod_math_5_act['description'])
    aux_mod_id_5 = list(df_mod_math_5_act['id'])

    aux_mod_description_4 = list(df_mod_math_4_act['description'])
    aux_mod_id_4 = list(df_mod_math_4_act['id'])
    
    df_skills_act = df_skills.loc[df_skills['active'] == 1]
    
    aux_skill_id = list(df_skills_act['id'].values)
    aux_bncc = list(df_skills_act['bncc_code'].values)
    bncc_2_skill_id = dict_maker(aux_bncc, aux_skill_id)
    
    aux_chap_1_5 = list(df_mod_math_5_act['id'].values)
    aux_hab_5 = list(df_mod_math_5_act['skill_id'].values)
    skill_2_mod_5 = dict_maker(aux_hab_5, aux_chap_1_5)
    mod_2_skill_5 = dict_maker(aux_chap_1_5, aux_hab_5)

    aux_chap_1_4 = list(df_mod_math_4_act['id'].values)
    aux_hab_4 = list(df_mod_math_4_act['skill_id'].values)
    skill_2_mod_4 = dict_maker(aux_hab_4, aux_chap_1_4)
    mod_2_skill_4 = dict_maker(aux_chap_1_4, aux_hab_4)

    skill_2_modchap = {**skill_2_mod_5, **skill_2_mod_4}
    
    hab_2_mod_5 = {}
    for code in aux_bncc:    
        if bncc_2_skill_id[code] in aux_hab_5: 
            hab_2_mod_5[code] = skill_2_mod_5[bncc_2_skill_id[code]]

    hab_2_mod_4 = {}
    for code in aux_bncc:    
        if bncc_2_skill_id[code] in aux_hab_4: 
            hab_2_mod_4[code] = skill_2_mod_4[bncc_2_skill_id[code]]
            
    aux_mod_description_5 = list(df_mod_math_5_act['description'])
    aux_mod_id_5 = list(df_mod_math_5_act['id'])

    aux_mod_description_4 = list(df_mod_math_4_act['description'])
    aux_mod_id_4 = list(df_mod_math_4_act['id'])
    
    df_videos = get_videos(yr, sub)
    
    # valores bizarros
    dict_perdido = {851.0: 4,193.0: 3,664.0: 3,220.0: 3,867.0: 3, 871.0: 3,865.0: 3,866.0: 3,624.0: 3,632.0: 3, 638.0: 3, 
                                822.0: 3,825.0: 3,837.0: 3, 957.0: 3, 66.0: 2, 242.0: 2, 533.0: 2, 245.0: 2, 36.0: 2, 492.0: 2, 
                                1991.0: 2, 240.0: 2, 75.0: 2, 1243.0: 2, 169.0: 2, 173.0: 2, 1352.0: 2, 190.0: 2, 356.0: 2, 
                                272.0: 2, 689.0: 2, 852.0: 2, 973.0: 2, 853.0: 2, 974.0: 2, 854.0: 2, 857.0: 2, 874.0: 2, 978.0: 2, 
                                491.0: 2, 637.0: 2, 634.0: 2, 219.0: 2, 640.0: 2, 631.0: 2, 651.0: 2, 652.0: 2, 656.0: 2, 659.0: 2, 
                                526.0: 2, 927.0: 2, 943.0: 2, 946.0: 2, 363.0: 2, 384.0: 2, 389.0: 2, 406.0: 2, 407.0: 2, 409.0: 2,
                                980.0: 2, 841.0: 2, 870.0: 2, 855.0: 2, 849.0: 2, 972.0: 2, 882.0: 2, 433.0: 2, 307.0: 2, 312.0: 2, 
                                313.0: 2, 316.0: 2, 320.0: 2, 394.0: 2, 1220.0: 2, 1410.0: 2, 2000.0: 2}
        
        
    chapter_2_hab_II = dict(chapter_2_hab)
    repetidos = list(dict_perdido.keys())
    chapter_2_hab_III = {k: int(v) for k, v in chapter_2_hab_II.items() if (not np.isnan(v)) and (v not in repetidos)}

    activity_aux = list(df_videos['activity_id'].values)
    vid_aux = list(df_videos['id'].values)
    activity_2_vid = dict_maker(activity_aux, vid_aux)

    acttt_doest_matters = []
    for ativ in aux_actt:
        if ativ not in activity_aux:
            acttt_doest_matters.append(ativ)


    for k in acttt_doest_matters:
        del activity_2_chapter[k]


    chap_2_acttt = double_value_inverter(activity_2_chapter)

    hab_2_chapter = {v:k for k,v in chapter_2_hab_III.items()}

    
    return {
        
        'TABS': TABS,
        
        'df_questions_current': df_questions_current,
        'df_questions_pre': df_questions_pre,
        
        'df_alternatives_current': df_alternatives_current,
        'df_alternatives_pre': df_alternatives_pre,
        
        'df_challenges': df_challenges,
        'df_data_students': df_data_students,
        
        'aux_alt_id_5': aux_alt_id_5,
        'aux_alt_5': aux_alt_5,
        'aux_alt_correct_5': aux_alt_correct_5,
        'aux_alt_id_das_alt_5': aux_alt_id_das_alt_5,
        
        'aux_interess_5_id': aux_interess_5_id,
        'aux_interess_5_title': aux_interess_5_title,
        'aux_interess_5_support': aux_interess_5_support,
        'aux_interess_5_command': aux_interess_5_command,
        'aux_interess_5_ano': aux_interess_5_ano,
        'aux_interess_5_hab': aux_interess_5_hab,

        'aux_alt_id_teste_5': aux_alt_id_teste_5,
        
        'aux_alt_id_4': aux_alt_id_4,
        'aux_alt_4': aux_alt_4,
        'aux_alt_correct_4': aux_alt_correct_4,
        'aux_alt_id_das_alt_4': aux_alt_id_das_alt_4,
        
        'aux_interess_4_id': aux_interess_4_id,
        'aux_interess_4_title': aux_interess_4_title,
        'aux_interess_4_support': aux_interess_4_support,
        'aux_interess_4_command': aux_interess_4_command,
        'aux_interess_4_ano': aux_interess_4_ano,
        'aux_interess_4_hab': aux_interess_4_hab,

        'aux_alt_id_teste_4': aux_alt_id_teste_4,
        
        'aux_habs_ids': aux_habs_ids,
        'aux_habs_bncc': aux_habs_bncc,
        
        'BNCC_5': BNCC_5,
        'BNCC_5_inv': BNCC_5_inv,
        
        'BNCC_4': BNCC_4,
        'BNCC_4_inv': BNCC_4_inv,\
        
        'bncc_5': bncc_5,
        'bncc_4': bncc_4,
        
        'challs_act': challs_act,
        
        'df_mod_current': df_mod_current,
        'df_mod_pre': df_mod_pre,
        'df_mod': df_mod,
        'df_mod_mat_5': df_mod_current,
        'df_mod_mat_4': df_mod_pre,
        'df_mod_act': df_mod_act,
        
        'chall_2_activity': chall_2_activity,
        
        'activity_2_chapter': activity_2_chapter,
        
        'chapter_2_hab': chapter_2_hab,
        
        'from_chall_2_skill': from_chall_2_skill,
        
        'mod_2_hab_5_': mod_2_hab_5_,
        'mod_2_hab_4_': mod_2_hab_4_,
        
        'hab_2_mod_5_': hab_2_mod_5_,
        'hab_2_mod_4_': hab_2_mod_4_,

        'mod_2_hab_5': mod_2_hab_5,
        'mod_2_hab_4': mod_2_hab_4,
        
        'hab_2_mod_5_act': hab_2_mod_5_act,
        'hab_2_mod_4_act': hab_2_mod_4_act,
        
        'from_skill_2_chall': from_skill_2_chall,
        
        'from_data_2_challs': from_data_2_challs,

        'df_mod_math_5_act': df_mod_math_5_act,
        'df_mod_math_4_act': df_mod_math_4_act,
        
        'ds': ds,
        'idex': idex,
        
        'ds_4': ds_4,
        'idex_4': idex_4,
        
        'aux_mod_description_5': aux_mod_description_5,
        'aux_mod_id_5': aux_mod_id_5,
        
        'aux_mod_description_4': aux_mod_description_4,
        'aux_mod_id_4': aux_mod_id_4,

        'bncc_2_skill_id': bncc_2_skill_id,
        'aux_skill_id': aux_skill_id,

        'skill_2_mod_5': skill_2_mod_5,
        'mod_2_skill_5': mod_2_skill_5,
        
        'skill_2_mod_4': skill_2_mod_4,
        'mod_2_skill_4': mod_2_skill_4,
        
        'skill_2_modchap': skill_2_modchap,
        
        'hab_2_mod_5': hab_2_mod_5,
        'hab_2_mod_4': hab_2_mod_4,
        
        'aux_mod_description_5': aux_mod_description_5,
        'aux_mod_id_5': aux_mod_id_5,

        'aux_mod_description_4': aux_mod_description_4,
        'aux_mod_id_4':  aux_mod_id_4,
        
        'chapter_2_hab_II': chapter_2_hab_II,
        'repetidos': repetidos,
        'chapter_2_hab_III': chapter_2_hab_III,

        'activity_aux': activity_aux,
        'vid_aux': vid_aux,
        'activity_2_vid': activity_2_vid,

        'chap_2_acttt': chap_2_acttt,

        'hab_2_chapter': hab_2_chapter

    }

