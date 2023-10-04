import itertools
import pandas as pd
from PrettyColorPrinter import add_printer
import numpy as np
import numexpr  # install!!!
from time import perf_counter

add_printer(1)
col_team1 = "bet365_nome1"
col_team2 = "bet365_nome_2"
df = pd.read_pickle(r"c:\arquivoraspagem2.pkl")

df = df.sample(37, replace=True).reset_index(drop=True)
df.bet365_nome1 = df.bet365_nome1 + np.random.uniform(1, 20000, len(df)).astype("U")
df.bet365_nome_2 = df.bet365_nome_2 + np.random.uniform(1, 20000, len(df)).astype("U")
valorparaapostar = 1000
maxadd = 1.051
casas_adicionais = 8

df = df.rename(
    columns={
        "betfair_0": "betfair_team1",
        "betfair_4": "betfair_empate",
        "betfair_8": "betfair_team2",
    }
)

for b in range(casas_adicionais):
    df[f"casa_team1{b}"] = df.bet365_time1 * np.random.uniform(
        1, high=maxadd, size=len(df)
    )
    df[f"casa_empate{b}"] = df.bet365_empate * np.random.uniform(
        1, high=maxadd, size=len(df)
    )
    df[f"casa_team2{b}"] = df.bet365_time2 * np.random.uniform(
        1, high=maxadd, size=len(df)
    )

start = perf_counter()

colunas_para_comparar = np.array(
    [
        ["bet365_time1", "bet365_empate", "bet365_time2"],
        ["betfair_team1", "betfair_empate", "betfair_team2"],
        *[
            [f"casa_team1{b}", f"casa_empate{b}", f"casa_team2{b}"]
            for b in range(casas_adicionais)
        ],
    ]
)

divcols = []
suffixo = "_div1"
for d in colunas_para_comparar.flatten():
    novo_column = f"{d}{suffixo}"
    df[novo_column] = (1 / df[d]).astype(np.float64)
    divcols.append(novo_column)

colunas_para_comparar_div = np.array(divcols).reshape(colunas_para_comparar.shape)

itercols = itertools.product(*colunas_para_comparar_div.T)

tabela_inteira = pd.concat(
    [df[col] for col in map(list, itercols)], ignore_index=True, copy=False
).fillna(0)

maisque1 = numexpr.evaluate(
    "sum(tabela_inteira, 1)",
    global_dict={},
    local_dict={"tabela_inteira": tabela_inteira},
)

menos_de_um = numexpr.evaluate(
    "maisque1<1", global_dict={}, local_dict={"maisque1": maisque1}
)

bons_resultados = tabela_inteira[menos_de_um]

tabela_colunas = pd.DataFrame(
    np.tile(tabela_inteira.columns, ((len(bons_resultados)), 1))
)

nao_zero = numexpr.evaluate(
    "bons_resultados!=0",
    global_dict={},
    local_dict={"bons_resultados": bons_resultados},
)

bons_ind = np.where(nao_zero)

bons_resultados_np_array = np.array(
    [bons_resultados.iat[*h] for h in zip(*bons_ind)]
).reshape((-1, 3))

bons_resultados_np_array_col = np.array(
    [tabela_colunas.iat[*h] for h in zip(*bons_ind)]
).reshape((-1, 3))
allresults = []
for colu, numero in zip(bons_resultados_np_array_col, bons_resultados_np_array):
    original = [x[: -len(suffixo)] for x in colu]

    vence = df.loc[
        (df[colu[0]] == numero[0])
        & (df[colu[1]] == numero[1])
        & (df[colu[2]] == numero[2])
    ][list((col_team1, col_team2, *colu, *original))]
    vence.columns = [
        "team1",
        "team2",
        "odd0",
        "odd1",
        "odd2",
        "odd_casa0",
        "odd_casa1",
        "odd_casa2",
    ]
    vence["percentagem"] = vence[["odd0", "odd1", "odd2"]].sum(axis=1)
    for i in range(len(colu)):
        vence[f"casa{i}"] = colu[i]
        vence[f"div{i}"] = vence[f"odd{i}"] / vence[f"percentagem"]
        vence[f"aposta{i}"] = valorparaapostar * vence[f"div{i}"]
        vence[f"lucro{i}"] = vence[f"aposta{i}"] * vence[f"odd_casa{i}"]
    allresults.append(vence)
try:
    dffinal = (
        pd.concat(allresults).sort_values(by=f"percentagem").reset_index(drop=True)
    )
except Exception:
    dffinal = pd.DataFrame()
print(dffinal)
unicos = dffinal.drop_duplicates(subset=["team1", "team2"])

print(perf_counter() - start)
