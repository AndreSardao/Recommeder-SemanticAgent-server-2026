"""
Created on Tue Dec 16 10:32:11 2025

@author: andresardao
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from app_infra_db import puxa
from app_infra_redis import get_features
from pathlib import Path
from datetime import datetime
import pandas as pd


# Function that separates questions by 'skills'
# Variável df_1: df_de_interess_4 ou df_de_interess_5
# Variável df_2: df_de_interess_alternativas_4 ou df_de_interess_alternativas_5
# VAriável lista_1: aux_interess_4_id ou aux_interess_5_id
# Variável lista_2: BNCC_4 ou BNCC_5

# Flitering relevant questions and putting to a dictionarical format /...
# .../that fits the necessities of the requested new error tabre.
def sep_questoes(hab, ano, df_1, df_2, lista_1, lista_2):    
    questoes_hab = []
    for q_id in lista_1:

        questao = {}

        questao['question_id'] = q_id
        
        df_de_interess = lista_1

        df = df_1[df_1['id'] == q_id]
        
        aux_hab = list(df['skill_id'].values)
        
        if aux_hab[0] in lista_2.keys(): 
            skill = lista_2[aux_hab[0]]
        
            if skill == hab:

                
                skill = lista_2[aux_hab[0]]

                questao['hab_bncc'] = skill 

                aux_title = list(df['title'].values)

                aux_support = list(df['support'].values)

                aux_command = list(df['command'].values)

                if (aux_title[0] != None) and (aux_support[0] != None) and (aux_command[0] != None):

                    questao['Enunciado'] = aux_title[0] + ' ' + aux_support[0] + ' ' + aux_command[0]

                if (aux_support[0] == None) and (aux_command[0] == None):

                    questao['Enunciado'] = aux_title[0]

                if (aux_support[0] == None) and (aux_command[0] != None):

                    questao['Enunciado'] = aux_title[0] + ' ' + aux_command[0]

                if (aux_support[0] != None) and (aux_command[0] == None):

                    questao['Enunciado'] = aux_title[0] + ' ' + aux_support[0]

                #questao['ano'] = ano

                df_alt = df_2[df_2['question_id'] == q_id]


                alt_content = list(df_alt['content'].values)
                alt_id = list(df_alt['id'].values)

                questao['A'] = alt_content[0]
                questao['B'] = alt_content[1]
                questao['C'] = alt_content[2]
                questao['D'] = alt_content[3]

                questao['A_alt_id'] = alt_id[0]
                questao['B_alt_id'] = alt_id[1]
                questao['C_alt_id'] = alt_id[2]
                questao['D_alt_id'] = alt_id[3]


                alt_correct = list(df_alt['correct'].values)

                for i in range(4):

                        if alt_correct[0] == 1:

                            questao['resposta_certa'] = 'A'

                        elif alt_correct[1] == 1:

                            questao['resposta_certa'] = 'B'

                        elif alt_correct[2] == 1:

                            questao['resposta_certa'] = 'C'

                        elif alt_correct[3] == 1:

                            questao['resposta_certa'] = 'D'

                questoes_hab.append(questao)


    return questoes_hab

# Loading the bundle with useful objects stored on Redis. Obs: type(feats): dictionary
#feats = get_features(5,2)

# listings maker
def mk_listings(HAB: str, feats: dict) -> dict:
    
    """
    Receives the BNCC code for the main Disciple (HAB).
    - Search for its pre-requesites (if any).
    - Generates respectives listing with the relevant questions.
    - if HAB doesn't admit pre-requites, this bagaça returns an empty list for the correspondent key 
    """
    hab_2_mod_5 = feats['hab_2_mod_5']
    hab_2_mod_4 = feats['hab_2_mod_4']

    mod_2_hab_5 = feats['mod_2_hab_5']
    mod_2_hab_4 = feats['mod_2_hab_4']

    BNCC_5 = feats['BNCC_5']
    BNCC_4 = feats['BNCC_4']

    aux_mod_id_5 = feats['aux_mod_id_5']
    aux_mod_id_4 = feats['aux_mod_id_4']

    aux_interess_5_id = feats['aux_interess_5_id']
    aux_interess_4_id = feats['aux_interess_4_id']

    df_questions_current = feats['df_questions_current']
    df_questions_pre = feats['df_questions_pre']
    df_alternatives_current = feats['df_alternatives_current']
    df_alternatives_pre = feats['df_alternatives_pre']
    
    mod_mae = hab_2_mod_5[HAB]
    aux = puxa(mod_mae)
    aux_1 = (aux or {}).get('previous_modules') or []
    
    prev_mods_4 = []
    
    prev_mods_5 = []
    
    if len(aux_1) != 0:
        for dct in aux_1:
            if ((dct['year']['id'] == 5) and (dct['active'] == 1)):
                prev_mods_5.append(dct['id'])
            if ((dct['year']['id'] == 4) and (dct['active'] == 1)):
                prev_mods_4.append(dct['id'])
                    
    habs_para_csv = []
    
    prev_habs_5 = []
    prev_habs_4 = []
    
    ids_5 = []
    ids_4 = []
    
    if len(prev_mods_5) != 0:
        for mod in prev_mods_5:
            if (mod in aux_mod_id_5):
                ids_5.append(mod_2_hab_5[mod])
                prev_habs_5.append(BNCC_5[mod_2_hab_5[mod]])
                habs_para_csv.append([BNCC_5[mod_2_hab_5[mod]],5])
            
    if len(prev_mods_4) != 0:
        for mod in prev_mods_4:
            if (mod in aux_mod_id_4):
                ids_4.append(mod_2_hab_4[mod])
                prev_habs_4.append(BNCC_4[mod_2_hab_4[mod]])
                habs_para_csv.append([BNCC_4[mod_2_hab_4[mod]],4])
                
    listing_HAB = sep_questoes(HAB, 5, df_questions_current, df_alternatives_current,aux_interess_5_id, BNCC_5)
    
    listings_pre = []
    
    for h in habs_para_csv:
        if h[1] == 5:
            aux_l = sep_questoes(h[0], 5, df_questions_current, df_alternatives_current,aux_interess_5_id, BNCC_5)
            listings_pre.append(aux_l)
        if h[1] == 4:
            aux_l = sep_questoes(h[0], 4, df_questions_pre, df_alternatives_pre,aux_interess_4_id, BNCC_4)
            listings_pre.append(aux_l)
    
    
    if len(listings_pre) >= 1:
        prim = listings_pre[0]
        if len(listings_pre) > 1:
            for l in listings_pre[1:]:
                for d in l:
                    prim.append(d)
    
    else: 
        prim = []
        
                
    prereqs_codes = sorted({h[0] for h in habs_para_csv})  # BNCC of pre-reqs (sem duplicar)

    Listings = {
        "HAB": listing_HAB,
        "prereq_combined": prim,
        "prereqs_codes": prereqs_codes,
    }
    
    
    return Listings  

def mk_listings_by_year_subject(HAB: str, yr: int, sub: int) -> dict:
    feats = get_features(int(yr), int(sub))
    return mk_listings(HAB, feats=feats)
    

# Cannonical order
LISTING_COLS = [
    "question_id", "hab_bncc", "Enunciado",
    "A", "B", "C", "D",
    "A_alt_id", "B_alt_id", "C_alt_id", "D_alt_id",
    "resposta_certa",
]

def listings_to_df(listing: list[dict], cols: list[str] = LISTING_COLS) -> pd.DataFrame:
    """
    Converts list[dict] into a  DataFrame.
    - Makes sure that missing fields gets a NaN.
    - Meneges the columns.
    """
    df = pd.DataFrame.from_records(listing)

    # Makes sure all expected columns
    for c in cols:
        if c not in df.columns:
            df[c] = None

    # Resorts and (if it's the case) cuts extras
    df = df[cols]

    # Sorts by question_id
    if "question_id" in df.columns:
        df = df.sort_values("question_id", kind="stable").reset_index(drop=True)

    return df

def maestro(
    hab: str,
    yr: int = 5,
    sub: int = 2,
    feats: dict | None = None,
    out_dir: str = "novas",
) -> dict:
    """
    Orchestrates the generation of new listings CSVs for a given HAB.

    - If feats is None, loads via get_features(yr, sub)
    - out_dir: output folder for generated listing CSVs.
    """

    HAB = hab

    if feats is None:
        feats = get_features(int(yr), int(sub))

    listings = mk_listings(HAB, feats=feats)
    
    

    aux_HAB = (listings or {}).get("HAB") or []
    aux_pre = (listings or {}).get("prereq_combined") or []

    df_HAB = listings_to_df(aux_HAB, LISTING_COLS)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    f1 = out_path / f"classificacao_questoes_{HAB}.csv"
    df_HAB.to_csv(f1, index=False, encoding="utf-8-sig", lineterminator="\n")

    if not aux_pre:
        return {
            "response": "Listagem gerada apenas para a habilidade (sem pré-requisitos).",
            "hab_csv": str(f1),
            "prereqs_csv": None,
            "hab_rows": int(len(df_HAB)),
            "prereqs_rows": 0,
            "prereqs_codes": [],
        }

    df_pre = listings_to_df(aux_pre, LISTING_COLS)
    f2 = out_path / f"classificacao_questoes_{HAB}_prereqs.csv"
    df_pre.to_csv(f2, index=False, encoding="utf-8-sig", lineterminator="\n")

    return {
        "response": "Listagens geradas para habilidade e pré-requisitos.",
        "hab_csv": str(f1),
        "prereqs_csv": str(f2),
        "hab_rows": int(len(df_HAB)),
        "prereqs_rows": int(len(df_pre)),
        "prereqs_codes": (listings or {}).get("prereqs_codes") or [],
    }



def maestro_meu(hab: str, feats: dict | None = None, out_dir: str = "novas") -> dict:
    """
    Orchestrates the generation of new listings CSVs for a given HAB.

    - feats: optional features bundle (so the caller can decide yr/sub via BNCC parsing).
      If not provided, defaults to the pilot bundle (5th grade, Math).
    - out_dir: output folder for generated listing CSVs.

    Returns:
      {"hab_csv": "...", "prereqs_csv": "...|None", "hab_rows": int, "prereqs_rows": int, "response": "..."}
    """
    HAB = hab

    if feats is None:
        # Pilot default (5th grade, Math). Keeps current behavior.
        from app_infra_redis import get_features
        feats = get_features(5, 2)

    listings = mk_listings(HAB, feats=feats)

    aux_HAB = (listings or {}).get("HAB") or []
    aux_pre = (listings or {}).get("prereq_combined") or []

    df_HAB = listings_to_df(aux_HAB, LISTING_COLS)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    f1 = out_path / f"classificacao_questoes_{HAB}.csv"
    df_HAB.to_csv(f1, index=False, encoding="utf-8-sig", lineterminator="\n")

    if not aux_pre:
        return {
            "response": "Listagem gerada apenas para a habilidade (sem pré-requisitos).",
            "hab_csv": str(f1),
            "prereqs_csv": None,
            "hab_rows": int(len(df_HAB)),
            "prereqs_rows": 0,
        }

    df_pre = listings_to_df(aux_pre, LISTING_COLS)
    f2 = out_path / f"classificacao_questoes_{HAB}_prereqs.csv"
    df_pre.to_csv(f2, index=False, encoding="utf-8-sig", lineterminator="\n")

    return {
        "response": "Listagens geradas para habilidade e pré-requisitos.",
        "hab_csv": str(f1),
        "prereqs_csv": str(f2),
        "hab_rows": int(len(df_HAB)),
        "prereqs_rows": int(len(df_pre)),
    }
'''
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Gerador de listings para tabela de erros.")
    parser.add_argument("--hab", required=True, help="Código BNCC da habilidade (ex: EF05MA06)")
    parser.add_argument("--yr", type=int, default=5, help="Ano (ex: 4 ou 5). Default=5")
    parser.add_argument("--sub", type=int, default=2, help="Disciplina (ex: 2=Mat). Default=2")
    parser.add_argument("--outdir", default="novas", help="Pasta de saída dos CSVs")

    args = parser.parse_args()

    out = maestro(hab=args.hab, yr=args.yr, sub=args.sub, out_dir=args.outdir)
    print(json.dumps(out, ensure_ascii=False, indent=2))
   
'''    
   
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--hab", required=True)
    parser.add_argument("--yr", type=int, required=True)
    parser.add_argument("--sub", type=int, required=True)
    args = parser.parse_args()

    out = mk_listings_by_year_subject(args.hab, args.yr, args.sub)
    print(json.dumps(out, ensure_ascii=False, indent=2))

