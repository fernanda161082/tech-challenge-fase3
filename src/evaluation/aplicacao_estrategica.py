"""
Aplicacao estrategica do modelo.

Traduz as previsoes em respostas para as perguntas de negocio do
desafio, produzindo material que um gestor publico consegue usar.

PERGUNTAS RESPONDIDAS AQUI
--------------------------
1. Quais municipios apresentam maior risco educacional?
   -> ranking por probabilidade de nao atingir a meta

2. Quais regioes possuem padroes semelhantes?
   -> agrupamento nao supervisionado em perfis municipais

3. Como prever municipios que podem nao atingir metas futuras?
   -> score de risco aplicado a toda a base, exportado como lista

As outras duas perguntas do desafio - quais fatores mais impactam e
quais variaveis tem maior influencia - sao respondidas pelo script de
interpretabilidade.

SOBRE O SCORE DE RISCO
----------------------
O modelo devolve a probabilidade de ATINGIR a meta. O risco e o
complemento: 1 menos essa probabilidade. Municipios sao ordenados do
maior para o menor risco, o que produz uma fila de priorizacao.

Um cuidado importante: o modelo foi treinado para prever 2024 com
dados de 2023. Aplica-lo aos mesmos municipios do treino serve para
demonstrar o uso, nao para avaliar desempenho - a avaliacao honesta
esta no conjunto de teste, no script de treinamento.

Execucao:
    python src/evaluation/aplicacao_estrategica.py
    python src/evaluation/aplicacao_estrategica.py --perfis 5
    python src/evaluation/aplicacao_estrategica.py --top 50
"""

from pathlib import Path
import argparse
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import joblib

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_PROCESSED = RAIZ_PROJETO / "data" / "processed"
PASTA_MODELOS = RAIZ_PROJETO / "models"
PASTA_REPORTS = RAIZ_PROJETO / "reports"
PASTA_IMAGES = RAIZ_PROJETO / "images"

ALVO = "atingiu_meta"

# Variaveis que descrevem o perfil do municipio, usadas no agrupamento.
# Sao poucas e interpretaveis de proposito: um perfil que ninguem
# consegue explicar nao vira politica publica.
VARIAVEIS_PERFIL = [
    "taxa_2023", "media_portugues_2023", "participacao_2023", "esforco_exigido",
]

COR_RISCO = "#C4553B"
COR_SEGURO = "#185FA5"
COR_NEUTRA = "#2A9D8F"
COR_GRID = "#E3E9EC"
COR_TEXTO = "#1C2B33"


def carregar():
    caminho_modelo = PASTA_MODELOS / "modelo.joblib"
    if not caminho_modelo.exists():
        print(
            "ERRO: modelo nao encontrado.\n"
            "      Rode antes: python src/modeling/treinar_modelo.py",
            file=sys.stderr,
        )
        sys.exit(1)

    modelo = joblib.load(caminho_modelo)
    with open(PASTA_REPORTS / "resultado_modelo.json", encoding="utf-8") as f:
        relatorio = json.load(f)
    base = pd.read_parquet(
        PASTA_PROCESSED / "base_enriquecida.parquet", engine="pyarrow"
    )
    return modelo, base, relatorio


def calcular_risco(modelo, base: pd.DataFrame, variaveis: list) -> pd.DataFrame:
    """Aplica o modelo e monta a tabela de risco por municipio."""
    X = base[variaveis]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probabilidade = modelo.predict_proba(X)[:, 1]

    risco = pd.DataFrame(index=base.index)
    risco["nome_municipio"] = base.get("nome_municipio")
    risco["sigla_uf"] = base["sigla_uf"]
    risco["regiao"] = base["regiao"]
    risco["taxa_2023"] = base["taxa_2023"]
    risco["meta_2024"] = base["meta_2024"]
    risco["esforco_exigido"] = base["esforco_exigido"]
    risco["participacao_2023"] = base.get("participacao_2023")
    risco["probabilidade_atingir"] = probabilidade.round(4)
    risco["score_risco"] = (1 - probabilidade).round(4)
    risco["resultado_real"] = base[ALVO]

    # Faixas para leitura rapida. Os cortes seguem tercos da escala de
    # probabilidade, nao quantis: o objetivo e comunicar nivel absoluto
    # de risco, nao posicao relativa.
    risco["faixa_risco"] = pd.cut(
        risco["score_risco"],
        bins=[-0.01, 0.35, 0.50, 0.65, 1.01],
        labels=["baixo", "moderado", "alto", "critico"],
    )
    return risco.sort_values("score_risco", ascending=False)


def escolher_perfis(Z: np.ndarray, maximo: int = 6) -> pd.DataFrame:
    """Avalia a qualidade do agrupamento para diferentes quantidades.

    A silhueta mede o quanto cada ponto esta mais proximo do proprio
    grupo do que do vizinho mais proximo. Valores maiores indicam
    grupos mais separados.
    """
    linhas = []
    for k in range(2, maximo + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        linhas.append({"k": k, "silhueta": round(silhouette_score(Z, km.labels_), 3)})
    return pd.DataFrame(linhas)


def agrupar_perfis(base: pd.DataFrame, n_perfis: int):
    """Agrupa municipios por perfil, sem usar o alvo.

    Este e um metodo NAO SUPERVISIONADO: o alvo nao entra. Por isso os
    grupos representam semelhanca de situacao, nao de resultado - e o
    resultado pode entao ser comparado entre grupos, como evidencia.
    """
    presentes = [c for c in VARIAVEIS_PERFIL if c in base.columns]
    dados = base[presentes]

    imputados = SimpleImputer(strategy="median").fit_transform(dados)
    Z = StandardScaler().fit_transform(imputados)

    qualidade = escolher_perfis(Z)
    km = KMeans(n_clusters=n_perfis, n_init=10, random_state=42).fit(Z)

    perfis = pd.DataFrame(index=base.index)
    perfis["perfil"] = km.labels_
    return perfis, qualidade, presentes


def descrever_perfis(base: pd.DataFrame, perfis: pd.DataFrame,
                     variaveis: list) -> pd.DataFrame:
    """Resume cada perfil e nomeia conforme o que o caracteriza.

    A referencia para 'alto' e 'baixo' e a mediana de TODOS os
    municipios, nao a mediana entre os perfis. Comparar os perfis entre
    si produziria rotulos enganosos: um perfil com participacao de 91%
    seria chamado de baixo apenas por estar abaixo dos outros tres, e
    nao por ser baixo em termos absolutos.
    """
    juntos = base.join(perfis)
    resumo = juntos.groupby("perfil")[variaveis].mean().round(1)
    resumo["municipios"] = juntos.groupby("perfil").size()
    resumo["pct_atingiu"] = (
        juntos.groupby("perfil")[ALVO].mean() * 100
    ).round(1)

    referencia_taxa = base["taxa_2023"].median()
    tem_part = "participacao_2023" in resumo.columns
    referencia_part = base["participacao_2023"].median() if tem_part else None

    nomes = []
    for _, linha in resumo.iterrows():
        taxa_alta = linha["taxa_2023"] >= referencia_taxa
        part_alta = (
            linha["participacao_2023"] >= referencia_part if tem_part else True
        )

        if taxa_alta and part_alta:
            nomes.append("consolidado")
        elif taxa_alta and not part_alta:
            nomes.append("desempenho_ok_cobertura_fragil")
        elif not taxa_alta and part_alta:
            nomes.append("desafio_com_gestao_ativa")
        else:
            nomes.append("vulneravel")

    # Se dois perfis caem no mesmo quadrante, desempata pelo desempenho
    # para que cada rotulo continue identificando um grupo unico.
    contagem = {}
    unicos = []
    for nome in nomes:
        contagem[nome] = contagem.get(nome, 0) + 1
        unicos.append(nome if contagem[nome] == 1 else f"{nome}_{contagem[nome]}")
    resumo["nome"] = unicos

    print(f"     referencia: taxa mediana {referencia_taxa:.1f}%"
          + (f", participacao mediana {referencia_part:.1f}%" if tem_part else ""))

    return resumo.sort_values("pct_atingiu")


def resumir_por_uf(risco: pd.DataFrame) -> pd.DataFrame:
    """Agrega o risco por unidade da federacao."""
    r = risco.groupby("sigla_uf").agg(
        municipios=("score_risco", "size"),
        risco_medio=("score_risco", "mean"),
        em_risco_alto=("faixa_risco", lambda s: int(s.isin(["alto", "critico"]).sum())),
        atingiu_real=("resultado_real", "mean"),
    )
    r["risco_medio"] = r.risco_medio.round(3)
    r["atingiu_real"] = (r.atingiu_real * 100).round(1)
    r["pct_em_risco"] = (r.em_risco_alto / r.municipios * 100).round(1)
    return r.sort_values("risco_medio", ascending=False)


def preparar_eixo(ax):
    ax.xaxis.grid(True, color=COR_GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COR_GRID)
    ax.tick_params(colors=COR_TEXTO, labelsize=9, length=0)


def grafico_risco_por_uf(resumo_uf: pd.DataFrame, destino: Path):
    dados = resumo_uf.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(dados) + 1.8))
    ax.barh(dados.index, dados.pct_em_risco, color=COR_RISCO, height=0.62)
    for i, (valor, n) in enumerate(zip(dados.pct_em_risco, dados.municipios)):
        ax.text(valor + 1, i, f"{valor:.0f}%  (n={n})",
                va="center", fontsize=8.5, color=COR_TEXTO)
    preparar_eixo(ax)
    ax.set_xlim(0, max(dados.pct_em_risco) * 1.25)
    ax.set_xlabel("% de municipios em risco alto ou critico",
                  color=COR_TEXTO, fontsize=10)
    ax.set_title(
        "Concentracao de risco por unidade da federacao\n"
        "15 estados com maior proporcao de municipios em risco",
        color=COR_TEXTO, fontsize=12, loc="left", pad=14,
    )
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grafico_perfis(resumo: pd.DataFrame, destino: Path):
    """Compara os perfis em duas dimensoes que os separam."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    tamanhos = resumo.municipios / resumo.municipios.max() * 900 + 100
    cores = [COR_RISCO if p < 50 else COR_SEGURO for p in resumo.pct_atingiu]

    ax.scatter(resumo.taxa_2023, resumo.participacao_2023,
               s=tamanhos, c=cores, alpha=0.75, edgecolors="white", linewidths=1.8)

    # Margem folgada nos eixos: sem ela, rotulos de pontos proximos da
    # borda saem da figura.
    margem_x = (resumo.taxa_2023.max() - resumo.taxa_2023.min()) * 0.22 + 6
    margem_y = (resumo.participacao_2023.max() - resumo.participacao_2023.min()) * 0.30 + 2
    ax.set_xlim(resumo.taxa_2023.min() - margem_x, resumo.taxa_2023.max() + margem_x)
    ax.set_ylim(resumo.participacao_2023.min() - margem_y,
                resumo.participacao_2023.max() + margem_y)

    centro_y = resumo.participacao_2023.mean()
    for _, linha in resumo.iterrows():
        # Rotulo acima do ponto quando ele esta na metade de baixo, e
        # abaixo quando esta na metade de cima: evita colisao com os
        # eixos nos dois extremos.
        acima = linha.participacao_2023 < centro_y
        deslocamento = 34 if acima else -44
        ax.annotate(
            f"{linha['nome']}\n{int(linha.municipios)} mun · {linha.pct_atingiu:.0f}% atingiu",
            (linha.taxa_2023, linha.participacao_2023),
            textcoords="offset points", xytext=(0, deslocamento),
            ha="center", fontsize=8.5, color=COR_TEXTO,
        )

    ax.xaxis.grid(True, color=COR_GRID, linewidth=0.8)
    ax.yaxis.grid(True, color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("bottom", "left"):
        ax.spines[lado].set_color(COR_GRID)
    ax.tick_params(colors=COR_TEXTO, labelsize=9, length=0)

    ax.set_xlabel("taxa de alfabetizacao em 2023 (%)", color=COR_TEXTO, fontsize=10)
    ax.set_ylabel("participacao na avaliacao (%)", color=COR_TEXTO, fontsize=10)
    ax.set_title(
        "Perfis municipais\n"
        "tamanho = numero de municipios · vermelho = menos da metade atingiu a meta",
        color=COR_TEXTO, fontsize=12, loc="left", pad=14,
    )
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aplicacao estrategica do modelo")
    parser.add_argument("--perfis", type=int, default=4,
                        help="quantidade de perfis no agrupamento")
    parser.add_argument("--top", type=int, default=30,
                        help="quantos municipios listar no ranking de risco")
    args = parser.parse_args()

    print("\nAplicacao estrategica")
    print("=" * 76)

    modelo, base, relatorio = carregar()
    risco = calcular_risco(modelo, base, relatorio["variaveis"])

    # ---------------- pergunta 1: municipios em risco ----------------
    print(f"\n  1. MUNICIPIOS EM MAIOR RISCO")
    print("  " + "-" * 72)
    distribuicao = risco.faixa_risco.value_counts().reindex(
        ["baixo", "moderado", "alto", "critico"]
    )
    for faixa, n in distribuicao.items():
        print(f"     {faixa:10} {n:>5} municipios ({n/len(risco)*100:4.1f}%)")

    criticos = risco[risco.faixa_risco.isin(["alto", "critico"])]
    print(f"\n     {len(criticos)} municipios em risco alto ou critico")
    print(f"\n     Os {min(10, args.top)} de maior risco:")
    colunas = ["nome_municipio", "sigla_uf", "taxa_2023", "esforco_exigido",
               "score_risco"]
    colunas = [c for c in colunas if c in risco.columns]
    print(risco.head(10)[colunas].to_string(
        index=True, justify="left", float_format=lambda v: f"{v:.2f}"
    ))

    # ---------------- pergunta 2: perfis ----------------
    print(f"\n  2. PERFIS MUNICIPAIS (agrupamento nao supervisionado)")
    print("  " + "-" * 72)
    perfis, qualidade, vars_perfil = agrupar_perfis(base, args.perfis)
    print("     qualidade do agrupamento (silhueta):")
    print("     " + "  ".join(f"k={r.k}:{r.silhueta}" for _, r in qualidade.iterrows()))
    print(f"     escolhido k={args.perfis} - privilegia interpretabilidade")

    resumo_perfis = descrever_perfis(base, perfis, vars_perfil)
    print()
    print(resumo_perfis.to_string())

    melhor = resumo_perfis.iloc[-1]
    pior = resumo_perfis.iloc[0]
    print(f"\n     contraste: '{pior['nome']}' atingiu {pior.pct_atingiu}% "
          f"com taxa media {pior.taxa_2023}")
    print(f"                '{melhor['nome']}' atingiu {melhor.pct_atingiu}% "
          f"com taxa media {melhor.taxa_2023}")

    # ---------------- pergunta 3: concentracao territorial ----------------
    print(f"\n  3. CONCENTRACAO TERRITORIAL DO RISCO")
    print("  " + "-" * 72)
    resumo_uf = resumir_por_uf(risco)
    print(f"     {'UF':4} {'munic.':>7} {'risco medio':>12} {'% em risco':>11} "
          f"{'atingiu real':>13}")
    for uf, r in resumo_uf.head(10).iterrows():
        print(f"     {uf:4} {int(r.municipios):>7} {r.risco_medio:>12.3f} "
              f"{r.pct_em_risco:>10.1f}% {r.atingiu_real:>12.1f}%")

    # ---------------- graficos e exportacao ----------------
    PASTA_IMAGES.mkdir(parents=True, exist_ok=True)
    PASTA_REPORTS.mkdir(parents=True, exist_ok=True)

    grafico_risco_por_uf(resumo_uf, PASTA_IMAGES / "risco_por_uf.png")
    if "participacao_2023" in resumo_perfis.columns:
        grafico_perfis(resumo_perfis, PASTA_IMAGES / "perfis_municipais.png")

    completo = risco.join(perfis)
    completo.to_csv(PASTA_REPORTS / "ranking_risco_municipios.csv", encoding="utf-8")
    resumo_perfis.to_csv(PASTA_REPORTS / "perfis_municipais.csv", encoding="utf-8")
    resumo_uf.to_csv(PASTA_REPORTS / "risco_por_uf.csv", encoding="utf-8")

    print(f"\n  arquivos gerados")
    print("  " + "-" * 72)
    print("     reports/ranking_risco_municipios.csv  lista completa priorizada")
    print("     reports/perfis_municipais.csv         caracterizacao dos perfis")
    print("     reports/risco_por_uf.csv              agregado estadual")
    print("     images/risco_por_uf.png")
    print("     images/perfis_municipais.png\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())