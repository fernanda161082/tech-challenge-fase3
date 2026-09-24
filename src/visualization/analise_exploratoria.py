"""
Analise exploratoria da base analitica (EDA).

Primeira etapa analitica do projeto: entender os dados ANTES de modelar.
Cada secao responde a um item exigido no enunciado da Fase 3 e termina
em uma decisao de modelagem ou em uma hipotese a testar.

    1. Comportamento geral  -> tamanho, tipos, nulos, equilibrio do alvo
    2. Distribuicoes        -> como as variaveis se espalham em cada classe
    3. Correlacoes          -> redundancias e multicolinearidade
    4. Padroes              -> atingimento por UF e por porte
    5. Variaveis relevantes -> poder de separacao de cada variavel sozinha
    6. Hipoteses            -> teto de previsibilidade e papel da participacao

SOBRE O USO DA TAXA DE 2024
---------------------------
As secoes 6.1 e 6.2 usam a coluna _auditoria_taxa_2024 para descrever o
problema (quanto o indicador oscila de um ano para o outro). Isso e
analise descritiva, nao modelagem: essa coluna continua proibida como
variavel de entrada, e o script de treino a remove. Usa-la aqui para
entender o fenomeno e o que permite saber, por exemplo, que nenhum
modelo honesto chegaria a 95% de acuracia neste problema.

Saidas:
    images/eda_distribuicoes.png
    images/eda_correlacoes.png
    images/eda_atingimento_por_uf.png
    images/eda_riqueza_vs_desempenho.png
    images/eda_teto_previsibilidade.png
    images/eda_hipotese_participacao.png
    reports/eda_resumo.json

Execucao:
    python src/visualization/analise_exploratoria.py
"""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem abrir janela
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_PROCESSED = RAIZ_PROJETO / "data" / "processed"
PASTA_IMAGES = RAIZ_PROJETO / "images"
PASTA_REPORTS = RAIZ_PROJETO / "reports"

ALVO = "atingiu_meta"
AUDITORIA = "_auditoria_taxa_2024"

# Mesma paleta dos graficos de interpretacao, validada para daltonismo.
# Azul = atingiu a meta, terracota = nao atingiu.
COR_POSITIVA = "#185FA5"
COR_NEGATIVA = "#C4553B"
COR_NEUTRA = "#8C979D"
COR_GRID = "#E3E9EC"
COR_TEXTO = "#1C2B33"
COR_TEXTO_SUAVE = "#5B6B73"

# Limite a partir do qual duas variaveis sao tratadas como
# praticamente a mesma informacao.
LIMITE_CORRELACAO = 0.80

# Faixas de participacao usadas no teste de hipotese da secao 6.2.
FAIXAS_PARTICIPACAO = [0, 85, 95, 100.01]
ROTULOS_PARTICIPACAO = ["abaixo de 85%", "85% a 95%", "95% ou mais"]


# ---------------------------------------------------------------------
# Apoio
# ---------------------------------------------------------------------
def ler_base() -> pd.DataFrame:
    caminho = PASTA_PROCESSED / "base_enriquecida.parquet"
    if not caminho.exists():
        print(
            "ERRO: base enriquecida nao encontrada.\n"
            "      Rode antes:\n"
            "        python src/preprocessing/construir_base_analitica.py\n"
            "        python src/preprocessing/enriquecer_base.py",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_parquet(caminho, engine="pyarrow")


def estilizar(ax) -> None:
    """Eixos discretos: o dado aparece, a moldura some."""
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COR_GRID)
    ax.tick_params(colors=COR_TEXTO_SUAVE, length=0, labelsize=9)
    ax.grid(axis="y", color=COR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def salvar(fig, nome: str) -> None:
    PASTA_IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_IMAGES / nome, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"    -> images/{nome}")


def titulo(texto: str) -> None:
    print(f"\n  {texto}")
    print("  " + "-" * 72)


def numericas(base: pd.DataFrame) -> list[str]:
    """Colunas numericas candidatas, sem alvo nem colunas tecnicas."""
    return [
        c for c in base.select_dtypes(include=[np.number]).columns
        if c != ALVO and not c.startswith("_")
    ]


# ---------------------------------------------------------------------
# 1. Comportamento geral
# ---------------------------------------------------------------------
def visao_geral(base: pd.DataFrame) -> dict:
    titulo("1. Comportamento geral dos dados")

    positivos = base[ALVO].mean()
    print(f"    municipios              : {len(base)}")
    print(f"    colunas                 : {base.shape[1]}")
    print(f"    UFs representadas       : {base['sigla_uf'].nunique()}")
    print(f"    atingiram a meta        : {base[ALVO].sum()} ({positivos*100:.1f}%)")
    print(f"    nao atingiram           : {(1-base[ALVO]).sum()} ({(1-positivos)*100:.1f}%)")

    # Equilibrio entre 40% e 60% dispensa tecnicas de rebalanceamento
    # (SMOTE, pesos de classe). Acima disso, a acuracia enganaria.
    if 0.4 <= positivos <= 0.6:
        print("    -> classes equilibradas: acuracia e AUC sao interpretaveis")
        print("       sem rebalanceamento")

    nulos = base.isna().mean().mul(100).round(2)
    nulos = nulos[nulos > 0]
    if len(nulos):
        print("    colunas com nulos:")
        for col, pct in nulos.items():
            print(f"      {col:28} {pct:5.2f}%")
    else:
        print("    nulos                   : nenhum")
    print("    -> a imputacao na pipeline existe como protecao para dados")
    print("       novos, que podem chegar incompletos")

    ufs_esperadas = {
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
        "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
        "SE", "SP", "TO",
    }
    ausentes = sorted(ufs_esperadas - set(base["sigla_uf"].unique()))
    if ausentes:
        print(f"    UFs ausentes            : {', '.join(ausentes)}")
        if "DF" in ausentes:
            # Brasilia nao e municipio: o DF acumula funcoes de estado e
            # municipio e nao tem rede municipal. A ausencia e estrutural.
            print("      DF: nao possui municipios nem rede municipal (estrutural)")
        outras = [u for u in ausentes if u != "DF"]
        if outras:
            print(f"      {', '.join(outras)}: sem municipios com rede municipal na base")

    return {
        "municipios": int(len(base)),
        "colunas": int(base.shape[1]),
        "ufs": int(base["sigla_uf"].nunique()),
        "ufs_ausentes": ausentes,
        "proporcao_atingiu": round(float(positivos), 4),
        "nulos_percentual": nulos.to_dict(),
    }


# ---------------------------------------------------------------------
# 2. Distribuicoes
# ---------------------------------------------------------------------
def distribuicoes(base: pd.DataFrame) -> dict:
    titulo("2. Distribuicoes por classe")

    painel = [
        ("taxa_2023", "Taxa de alfabetização 2023 (%)"),
        ("esforco_exigido", "Esforço exigido (meta 2024 − taxa 2023, p.p.)"),
        ("participacao_2023", "Participação na avaliação 2023 (%)"),
        ("log_populacao", "População (escala logarítmica)"),
    ]
    painel = [(c, r) for c, r in painel if c in base.columns]

    fig, eixos = plt.subplots(2, 2, figsize=(11, 7))
    resumo = {}

    for ax, (coluna, rotulo) in zip(eixos.flat, painel):
        dados = base[[coluna, ALVO]].dropna()
        bins = np.histogram_bin_edges(dados[coluna], bins=30)

        # Contorno em vez de preenchimento: as duas classes se sobrepoem
        # e barras cheias esconderiam uma a outra.
        for classe, cor, nome in [(1, COR_POSITIVA, "Atingiu a meta"),
                                  (0, COR_NEGATIVA, "Não atingiu")]:
            ax.hist(dados.loc[dados[ALVO] == classe, coluna], bins=bins,
                    histtype="step", linewidth=2, color=cor, label=nome)

        if coluna == "esforco_exigido":
            ax.axvline(0, color=COR_TEXTO_SUAVE, linewidth=1, linestyle="--")
            ax.text(0, ax.get_ylim()[1] * 0.95, "  bastava não piorar ←",
                    ha="right", va="top", fontsize=8, color=COR_TEXTO_SUAVE)

        ax.set_title(rotulo, fontsize=10, color=COR_TEXTO, loc="left")
        ax.set_ylabel("municípios", fontsize=8, color=COR_TEXTO_SUAVE)
        estilizar(ax)

        medias = dados.groupby(ALVO)[coluna].mean()
        assimetria = float(dados[coluna].skew())
        resumo[coluna] = {
            "media_atingiu": round(float(medias.get(1, np.nan)), 2),
            "media_nao_atingiu": round(float(medias.get(0, np.nan)), 2),
            "mediana": round(float(dados[coluna].median()), 2),
            "assimetria": round(assimetria, 2),
        }
        print(f"    {coluna:22} media atingiu {medias.get(1):8.2f} | "
              f"nao atingiu {medias.get(0):8.2f} | assimetria {assimetria:+.2f}")

    for ax in eixos.flat[len(painel):]:
        ax.set_visible(False)

    handles, labels = eixos.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9,
               ncol=2, bbox_to_anchor=(0.98, 1.0))
    fig.suptitle("Distribuição das principais variáveis por resultado em 2024",
                 x=0.02, ha="left", fontsize=12, color=COR_TEXTO)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    salvar(fig, "eda_distribuicoes.png")

    if "populacao" in base.columns:
        print(f"    populacao sem log: assimetria {base['populacao'].skew():+.1f}"
              " -> justifica usar log_populacao")
        resumo["populacao_assimetria_sem_log"] = round(float(base["populacao"].skew()), 2)

    return resumo


# ---------------------------------------------------------------------
# 3. Correlacoes
# ---------------------------------------------------------------------
def correlacoes(base: pd.DataFrame) -> dict:
    titulo("3. Correlacoes e redundancias")

    colunas = numericas(base)

    # Variancia zero: a coluna tem o mesmo valor em todas as linhas e
    # nao carrega informacao nenhuma.
    constantes = [c for c in colunas if base[c].nunique(dropna=True) <= 1]
    for c in constantes:
        print(f"    [CONSTANTE] {c} = {base[c].dropna().iloc[0]} em todos os municipios")

    colunas = [c for c in colunas if c not in constantes]
    matriz = base[colunas].corr()

    pares = []
    for i, a in enumerate(colunas):
        for b in colunas[i + 1:]:
            r = matriz.loc[a, b]
            if abs(r) > LIMITE_CORRELACAO:
                pares.append({"var_1": a, "var_2": b, "r": round(float(r), 3)})
    pares.sort(key=lambda p: -abs(p["r"]))

    print(f"    pares com |r| > {LIMITE_CORRELACAO}: {len(pares)}")
    for p in pares:
        marca = "  <- mesma informacao" if abs(p["r"]) > 0.99 else ""
        print(f"      {p['var_1']:22} x {p['var_2']:22} {p['r']:+.3f}{marca}")

    redundantes = [p for p in pares if abs(p["r"]) > 0.99]

    # ---- mapa de calor ----
    # Divergente: azul para correlacao positiva, terracota para negativa,
    # cinza claro no zero. So o triangulo inferior, porque a matriz e
    # simetrica e a metade de cima repetiria a de baixo.
    mapa = LinearSegmentedColormap.from_list(
        "divergente", [COR_NEGATIVA, "#F2F4F5", COR_POSITIVA]
    )
    n = len(colunas)
    mascara = np.triu(np.ones((n, n), dtype=bool))
    valores = np.ma.array(matriz.values, mask=mascara)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    imagem = ax.imshow(valores, cmap=mapa, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i):
            r = matriz.values[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(r) > 0.6 else COR_TEXTO,
                    fontweight="bold" if abs(r) > LIMITE_CORRELACAO else "normal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(colunas, rotation=45, ha="right", fontsize=8, color=COR_TEXTO)
    ax.set_yticklabels(colunas, fontsize=8, color=COR_TEXTO)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.tick_params(length=0)
    barra = fig.colorbar(imagem, ax=ax, shrink=0.7)
    barra.outline.set_visible(False)
    barra.ax.tick_params(labelsize=8, colors=COR_TEXTO_SUAVE, length=0)
    ax.set_title(f"Correlação entre variáveis numéricas (negrito: |r| > {LIMITE_CORRELACAO})",
                 fontsize=11, color=COR_TEXTO, loc="left")
    fig.tight_layout()
    salvar(fig, "eda_correlacoes.png")

    print("    -> decisoes:")
    for c in constantes:
        print(f"       remover {c} (constante)")
    for p in redundantes:
        print(f"       remover {p['var_2']} (duplica {p['var_1']})")
    print("       os demais pares ficam, mas a importancia das variaveis")
    print("       precisa de teste de estabilidade (multicolinearidade)")

    return {
        "constantes": constantes,
        "pares_alta_correlacao": pares,
        "redundantes": redundantes,
    }


# ---------------------------------------------------------------------
# 4. Padroes territoriais e de porte
# ---------------------------------------------------------------------
def padroes(base: pd.DataFrame) -> dict:
    titulo("4. Padroes: atingimento por UF e por porte")

    geral = base[ALVO].mean()
    por_uf = (base.groupby("sigla_uf")[ALVO]
              .agg(municipios="size", atingiu="mean")
              .sort_values("atingiu"))

    amplitude = por_uf["atingiu"].max() - por_uf["atingiu"].min()
    print(f"    media nacional: {geral*100:.1f}%")
    print(f"    maior: {por_uf.index[-1]} {por_uf['atingiu'].iloc[-1]*100:.1f}%   "
          f"menor: {por_uf.index[0]} {por_uf['atingiu'].iloc[0]*100:.1f}%   "
          f"amplitude {amplitude*100:.0f} p.p.")

    # ---- grafico por UF ----
    fig, ax = plt.subplots(figsize=(9, 8))
    cores = [COR_POSITIVA if v >= geral else COR_NEGATIVA for v in por_uf["atingiu"]]
    posicoes = np.arange(len(por_uf))
    ax.barh(posicoes, por_uf["atingiu"] * 100, color=cores, height=0.72)
    ax.axvline(geral * 100, color=COR_TEXTO_SUAVE, linewidth=1, linestyle="--")
    ax.text(geral * 100, len(por_uf) - 0.2, f" média nacional {geral*100:.1f}%",
            fontsize=8, color=COR_TEXTO_SUAVE, va="bottom")
    ax.set_yticks(posicoes)
    ax.set_yticklabels([f"{uf}  (n={n})" for uf, n in por_uf["municipios"].items()],
                       fontsize=8, color=COR_TEXTO)

    # Rotulo de valor so nos extremos: tres de cada ponta.
    for i in list(range(3)) + list(range(len(por_uf) - 3, len(por_uf))):
        valor = por_uf["atingiu"].iloc[i] * 100
        ax.text(valor + 0.8, i, f"{valor:.0f}%", va="center", fontsize=8,
                color=COR_TEXTO)

    ax.set_xlim(0, 100)
    ax.set_xlabel("% de municípios que atingiram a meta de 2024",
                  fontsize=9, color=COR_TEXTO_SUAVE)
    estilizar(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=COR_GRID, linewidth=0.8)
    ax.set_title("Atingimento da meta por UF — azul acima da média, terracota abaixo",
                 fontsize=11, color=COR_TEXTO, loc="left")
    fig.tight_layout()
    salvar(fig, "eda_atingimento_por_uf.png")

    resultado = {
        "media_nacional": round(float(geral), 4),
        "amplitude_entre_ufs_pp": round(float(amplitude * 100), 1),
        "por_uf": {uf: {"municipios": int(r.municipios), "atingiu": round(float(r.atingiu), 4)}
                   for uf, r in por_uf.iterrows()},
    }

    if "porte_municipio" in base.columns:
        por_porte = base.groupby("porte_municipio")[ALVO].agg(municipios="size", atingiu="mean")
        print("    por porte populacional:")
        for porte, r in por_porte.iterrows():
            aviso = "  (amostra pequena)" if r.municipios < 30 else ""
            print(f"      {porte:12} {int(r.municipios):>5} municipios  "
                  f"{r.atingiu*100:5.1f}%{aviso}")
        validos = por_porte[por_porte["municipios"] >= 30]["atingiu"]
        print(f"    -> variacao entre portes: {(validos.max()-validos.min())*100:.0f} p.p. "
              f"contra {amplitude*100:.0f} p.p. entre UFs")
        print("       hipotese: territorio pesa mais que tamanho")
        resultado["por_porte"] = {p: {"municipios": int(r.municipios), "atingiu": round(float(r.atingiu), 4)}
                                  for p, r in por_porte.iterrows()}

    return resultado


# ---------------------------------------------------------------------
# 5. Variaveis relevantes
# ---------------------------------------------------------------------
def codificar_por_alvo(coluna: pd.DataFrame, alvo: pd.Series) -> np.ndarray:
    """Troca cada categoria pela taxa de atingimento do seu grupo.

    A forma como o TargetEncoder recebe a particao mudou no
    scikit-learn 1.9: antes era o parametro random_state, agora e um
    objeto de validacao cruzada. O codigo tenta a forma nova e cai na
    antiga se a versao instalada nao a aceitar, para rodar nas duas.
    """
    from sklearn.preprocessing import TargetEncoder
    from sklearn.model_selection import StratifiedKFold

    particoes = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    try:
        return TargetEncoder(target_type="binary", cv=particoes).fit_transform(
            coluna, alvo
        )
    except (ValueError, TypeError):
        return TargetEncoder(target_type="binary", random_state=0).fit_transform(
            coluna, alvo
        )


def riqueza_e_desempenho(base: pd.DataFrame) -> dict:
    """Testa se a riqueza do estado explica o desempenho educacional.

    A UF e a variavel mais importante do modelo. Uma explicacao
    possivel e que "estado" seja apenas "riqueza" disfarcada de
    geografia. Esta secao confronta as duas grandezas no nivel em que
    o efeito aparece: o estadual.

    Se a hipotese economica valesse, os pontos formariam uma diagonal
    ascendente - quanto mais rico o estado, maior o atingimento.
    """
    titulo("4.1 Riqueza do estado explica o desempenho?")

    coluna = "pib_per_capita" if "pib_per_capita" in base.columns else None
    if coluna is None:
        print("    [AUSENTE] PIB por habitante - secao pulada")
        print("              rode antes: python src/preprocessing/baixar_dados_ibge.py")
        return {}

    por_uf = base.groupby("sigla_uf").agg(
        municipios=(ALVO, "size"),
        pib_mediano=(coluna, "median"),
        atingiu=(ALVO, "mean"),
    ).dropna()

    # Correlacao entre a riqueza mediana do estado e a proporcao de
    # municipios que atingiram a meta. Uma unidade por UF: 24 pontos.
    r = float(np.corrcoef(por_uf["pib_mediano"], por_uf["atingiu"])[0, 1])

    # Mesma pergunta dentro de cada estado, para separar o efeito
    # entre estados do efeito entre municipios do mesmo estado.
    dentro = base.groupby("sigla_uf").apply(
        lambda g: np.corrcoef(g[coluna], g[ALVO])[0, 1]
        if g[coluna].nunique() > 1 and g[ALVO].nunique() > 1 else np.nan,
        include_groups=False,
    ).dropna()

    print(f"    correlacao entre UFs   (riqueza x atingimento): {r:+.3f}")
    print(f"    correlacao media dentro das UFs               : {dentro.mean():+.3f}")
    print(f"    PIB por habitante mediano: maior {por_uf.pib_mediano.idxmax()} "
          f"R$ {por_uf.pib_mediano.max():,.0f} | menor {por_uf.pib_mediano.idxmin()} "
          f"R$ {por_uf.pib_mediano.min():,.0f}")

    if abs(r) < 0.3:
        print("    -> a riqueza do estado NAO explica o desempenho educacional")
        print("       o efeito da UF no modelo e institucional, nao economico")
    else:
        print("    -> ha associacao entre riqueza e desempenho estadual")

    # ---- grafico ----
    fig, ax = plt.subplots(figsize=(9, 6))
    destaques = {"CE": COR_POSITIVA, "RS": COR_NEGATIVA, "BA": COR_NEGATIVA}

    for uf, linha in por_uf.iterrows():
        cor = destaques.get(uf, COR_NEUTRA)
        destaque = uf in destaques
        ax.scatter(linha.pib_mediano, linha.atingiu * 100,
                   s=90 if destaque else 55, color=cor,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(uf, (linha.pib_mediano, linha.atingiu * 100),
                    xytext=(0, 9 if linha.atingiu < 0.75 else -16),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=COR_TEXTO if destaque else COR_TEXTO_SUAVE,
                    fontweight="bold" if destaque else "normal")

    ax.axhline(base[ALVO].mean() * 100, color=COR_GRID, linewidth=1, zorder=1)
    # Separador de milhar no padrao brasileiro, para o eixo ficar legivel.
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    )
    ax.set_xlabel("PIB por habitante — mediana dos municípios do estado (R$)",
                  fontsize=9, color=COR_TEXTO_SUAVE)
    ax.set_ylabel("% de municípios que atingiram a meta", fontsize=9,
                  color=COR_TEXTO_SUAVE)
    veredito = ("Riqueza do estado não explica alfabetização"
                if abs(r) < 0.3 else "Riqueza e desempenho andam juntos entre os estados")
    ax.set_title(f"{veredito} — correlação entre UFs: {r:+.2f}",
                 fontsize=11, color=COR_TEXTO, loc="left")
    estilizar(ax)
    ax.grid(axis="x", color=COR_GRID, linewidth=0.8)
    fig.tight_layout()
    salvar(fig, "eda_riqueza_vs_desempenho.png")

    return {
        "correlacao_entre_ufs": round(r, 3),
        "correlacao_media_dentro_das_ufs": round(float(dentro.mean()), 3),
        "por_uf": {uf: {"pib_mediano": round(float(l.pib_mediano), 2),
                        "atingiu": round(float(l.atingiu), 4)}
                   for uf, l in por_uf.iterrows()},
    }


def relevancia_individual(base: pd.DataFrame) -> dict:
    """AUC de cada variavel usada sozinha para separar as classes.

    AUC 0,5 = a variavel nao separa nada (moeda). Quanto mais longe de
    0,5, mais ela separa. Abaixo de 0,5 a relacao e inversa (valores
    maiores associados a NAO atingir); por isso o poder de separacao e
    reportado como max(AUC, 1 - AUC), com a direcao ao lado.

    E uma medida univariada: nao enxerga combinacoes nem redundancias.
    Serve para orientar, nao para escolher variaveis sozinha.
    """
    titulo("5. Poder de separacao de cada variavel sozinha (AUC univariada)")

    linhas = []
    for coluna in numericas(base):
        dados = base[[coluna, ALVO]].dropna()
        if dados[coluna].nunique() <= 1:
            continue
        auc = roc_auc_score(dados[ALVO], dados[coluna])
        linhas.append({
            "variavel": coluna,
            "poder_separacao": round(float(max(auc, 1 - auc)), 3),
            "direcao": "maior -> mais chance" if auc >= 0.5 else "maior -> menos chance",
        })
    # Categoricas nao tem ordem, entao a AUC direta nao se aplica. Cada
    # categoria e trocada pela taxa de atingimento do seu grupo, e essa
    # taxa vira o "valor". O TargetEncoder faz isso por validacao
    # cruzada: a taxa de cada municipio e calculada SEM ele proprio.
    # Calcular com o proprio municipio dentro vazaria o alvo e inflaria
    # a AUC - o mesmo vazamento que a pipeline de treino evita.
    try:
        for coluna in ["sigla_uf", "regiao", "porte_municipio"]:
            if coluna not in base.columns:
                continue
            dados = base[[coluna, ALVO]].dropna()
            codificado = codificar_por_alvo(dados[[coluna]], dados[ALVO]).ravel()
            auc = roc_auc_score(dados[ALVO], codificado)
            linhas.append({
                "variavel": f"{coluna} (categorica)",
                "poder_separacao": round(float(max(auc, 1 - auc)), 3),
                "direcao": "depende da categoria",
            })
    except ImportError:
        print("    [AVISO] scikit-learn sem TargetEncoder - categoricas puladas")

    linhas.sort(key=lambda l: -l["poder_separacao"])

    for l in linhas:
        barra = "#" * int((l["poder_separacao"] - 0.5) * 100)
        print(f"    {l['variavel']:28} {l['poder_separacao']:.3f}  {barra:<25} {l['direcao']}")
    print("    (0,500 = nao separa; o modelo final combina todas e chega a ~0,78)")
    print("    -> o desempenho de 2023, sozinho, quase nao separa as classes:")
    print("       quem ia bem nao necessariamente atinge a meta, porque a meta")
    print("       acompanha o ponto de partida. O que separa e onde o municipio")
    print("       esta e como a rede se organiza (participacao).")

    return {"auc_univariada": linhas}


# ---------------------------------------------------------------------
# 6. Hipoteses
# ---------------------------------------------------------------------
def teto_previsibilidade(base: pd.DataFrame) -> dict:
    """6.1 - Quanto do resultado e previsivel?

    Compara o tamanho do sinal (o quanto o municipio precisa crescer)
    com o tamanho do ruido (o quanto o indicador oscila de um ano para
    o outro). Se o ruido for muito maior, existe um teto para qualquer
    modelo, por melhor que seja.
    """
    titulo("6.1 Hipotese: existe um teto de previsibilidade")

    if AUDITORIA not in base.columns:
        print("    [AUSENTE] coluna de auditoria - secao pulada")
        return {}

    variacao = base[AUDITORIA] - base["taxa_2023"]
    ruido = float(variacao.std())
    sinal = float(base["esforco_exigido"].median())
    negativo = base["esforco_exigido"] < 0
    falharam_negativo = float(1 - base.loc[negativo, ALVO].mean())

    print(f"    desvio-padrao da variacao 2023 -> 2024 : {ruido:5.2f} p.p.  (ruido)")
    print(f"    esforco exigido mediano               : {sinal:5.2f} p.p.  (sinal)")
    print(f"    razao ruido / sinal                   : {ruido/sinal:5.1f}x")
    print(f"    municipios com esforco negativo       : {negativo.mean()*100:5.1f}%")
    print(f"      ...e que mesmo assim nao atingiram  : {falharam_negativo*100:5.1f}%")
    print("    -> conclusao: AUC muito alta neste problema seria sinal de")
    print("       vazamento, nao de um modelo excelente")

    # ---- grafico ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(variacao.dropna(), bins=60, color=COR_NEUTRA, edgecolor="white",
            linewidth=0.5)
    ax.axvline(0, color=COR_TEXTO_SUAVE, linewidth=1)
    ax.axvspan(0, sinal, color=COR_POSITIVA, alpha=0.35, linewidth=0)
    media = float(variacao.mean())
    for lado in (-1, 1):
        ax.axvline(media + lado * ruido, color=COR_TEXTO, linewidth=1,
                   linestyle=":")
    topo = ax.get_ylim()[1]
    ax.text(sinal + 1, topo * 0.97,
            f"← esforço mediano exigido: {sinal:.2f} p.p.",
            fontsize=9, color=COR_POSITIVA, va="top")
    ax.text(media - ruido - 2, topo * 0.97,
            f"linhas pontilhadas: ±1 desvio-padrão\n"
            f"= ±{ruido:.1f} p.p.  (≈{ruido/sinal:.0f}× o esforço)",
            fontsize=9, color=COR_TEXTO, va="top", ha="right")
    ax.set_xlabel("variação da taxa de alfabetização entre 2023 e 2024 (p.p.)",
                  fontsize=9, color=COR_TEXTO_SUAVE)
    ax.set_ylabel("municípios", fontsize=9, color=COR_TEXTO_SUAVE)
    estilizar(ax)
    ax.set_title("O indicador oscila muito mais do que a meta exige — faixa azul = esforço mediano",
                 fontsize=11, color=COR_TEXTO, loc="left")
    fig.tight_layout()
    salvar(fig, "eda_teto_previsibilidade.png")

    return {
        "desvio_padrao_variacao_pp": round(ruido, 2),
        "esforco_mediano_pp": round(sinal, 2),
        "razao_ruido_sinal": round(ruido / sinal, 1),
        "proporcao_esforco_negativo": round(float(negativo.mean()), 4),
        "falharam_mesmo_com_esforco_negativo": round(falharam_negativo, 4),
    }


def hipotese_participacao(base: pd.DataFrame) -> dict:
    """6.2 - A participacao na avaliacao importa? Por que?

    Duas explicacoes concorrentes:

      A) Ruido: com menos alunos avaliados, o indicador e mais instavel.
         Previsao: a VOLATILIDADE da variacao cai quando a participacao sobe.

      B) Gestao: redes que conseguem levar quase todos os alunos a prova
         tambem sao redes mais organizadas.
         Previsao: a MELHORIA MEDIA sobe com a participacao, mesmo que a
         volatilidade fique igual.

    O teste separa as duas olhando volatilidade e melhoria por faixa.
    """
    titulo("6.2 Hipotese: participacao como ruido ou como gestao")

    if AUDITORIA not in base.columns or "participacao_2023" not in base.columns:
        print("    [AUSENTE] colunas necessarias - secao pulada")
        return {}

    variacao = base[AUDITORIA] - base["taxa_2023"]
    faixa = pd.cut(base["participacao_2023"], FAIXAS_PARTICIPACAO,
                   labels=ROTULOS_PARTICIPACAO, right=False)

    tabela = pd.DataFrame({
        "municipios": variacao.groupby(faixa, observed=False).size(),
        "volatilidade": variacao.groupby(faixa, observed=False).std(),
        "melhoria_media": variacao.groupby(faixa, observed=False).mean(),
        "atingiu": base[ALVO].groupby(faixa, observed=False).mean(),
    })
    correlacao = float(np.corrcoef(base["participacao_2023"], variacao.abs())[0, 1])

    print(f"    {'faixa':16} {'municipios':>10} {'volatilidade':>13} "
          f"{'melhoria':>10} {'atingiu':>9}")
    for rotulo, r in tabela.iterrows():
        print(f"    {rotulo:16} {int(r.municipios):>10} {r.volatilidade:>12.2f}  "
              f"{r.melhoria_media:>+9.2f} {r.atingiu*100:>8.1f}%")
    print(f"    correlacao participacao x |variacao| : {correlacao:+.3f}")

    diferenca_vol = tabela["volatilidade"].iloc[-1] - tabela["volatilidade"].iloc[0]
    diferenca_mel = tabela["melhoria_media"].iloc[-1] - tabela["melhoria_media"].iloc[0]
    if abs(correlacao) < 0.1 and diferenca_mel > 2:
        veredito = "B (gestao)"
        print("    -> hipotese A (ruido) REJEITADA: a volatilidade nao cai com a")
        print("       participacao")
        print("    -> hipotese B (gestao) SUSTENTADA: a melhoria media cresce "
              f"{diferenca_mel:+.1f} p.p.")
        print("       da faixa mais baixa para a mais alta")
    else:
        veredito = "inconclusivo"
        print("    -> resultado inconclusivo: rever faixas ou metodo")

    # ---- grafico: duas medidas de escala diferente, dois paineis ----
    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(tabela))

    esq.bar(x, tabela["volatilidade"], color=COR_NEUTRA, width=0.6)
    for i, v in enumerate(tabela["volatilidade"]):
        esq.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=COR_TEXTO)
    esq.set_title("A) Volatilidade: desvio-padrão da variação (p.p.)\n"
                  "praticamente igual → não é ruído",
                  fontsize=10, color=COR_TEXTO, loc="left")

    dir_.bar(x, tabela["melhoria_media"], color=COR_POSITIVA, width=0.6)
    for i, v in enumerate(tabela["melhoria_media"]):
        dir_.text(i, max(v, 0) + 0.15, f"{v:+.1f}", ha="center", fontsize=9,
                  color=COR_TEXTO)
    dir_.axhline(0, color=COR_TEXTO_SUAVE, linewidth=0.8)
    dir_.set_title("B) Melhoria média 2023 → 2024 (p.p.)\n"
                   "cresce com a participação → capacidade de gestão",
                   fontsize=10, color=COR_TEXTO, loc="left")

    for ax in (esq, dir_):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n(n={int(n)})" for r, n in tabela["municipios"].items()],
                           fontsize=8, color=COR_TEXTO)
        ax.set_xlabel("participação na avaliação de 2023", fontsize=8,
                      color=COR_TEXTO_SUAVE)
        estilizar(ax)
    fig.tight_layout()
    salvar(fig, "eda_hipotese_participacao.png")

    return {
        "faixas": {r: {"municipios": int(l.municipios),
                       "volatilidade_pp": round(float(l.volatilidade), 2),
                       "melhoria_media_pp": round(float(l.melhoria_media), 2),
                       "atingiu": round(float(l.atingiu), 4)}
                   for r, l in tabela.iterrows()},
        "correlacao_participacao_x_variacao_absoluta": round(correlacao, 3),
        "diferenca_volatilidade_pp": round(float(diferenca_vol), 2),
        "diferenca_melhoria_pp": round(float(diferenca_mel), 2),
        "hipotese_sustentada": veredito,
    }


# ---------------------------------------------------------------------
def main() -> int:
    print("\nAnalise exploratoria")
    print("=" * 76)

    base = ler_base()

    resumo = {
        "visao_geral": visao_geral(base),
        "distribuicoes": distribuicoes(base),
        "correlacoes": correlacoes(base),
        "padroes": padroes(base),
        "riqueza_e_desempenho": riqueza_e_desempenho(base),
        "relevancia": relevancia_individual(base),
        "teto_previsibilidade": teto_previsibilidade(base),
        "hipotese_participacao": hipotese_participacao(base),
    }

    PASTA_REPORTS.mkdir(parents=True, exist_ok=True)
    with open(PASTA_REPORTS / "eda_resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print("\n  resumo numerico em reports/eda_resumo.json\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
