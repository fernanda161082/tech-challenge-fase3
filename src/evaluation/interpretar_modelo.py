"""
Interpretabilidade do modelo preditivo.

Responde a pergunta de negocio central do desafio - quais fatores mais
influenciam o atingimento da meta de alfabetizacao - usando tres
tecnicas complementares e uma verificacao de estabilidade.

O PROBLEMA QUE PRECISA SER ENFRENTADO ANTES
-------------------------------------------
A analise exploratoria encontrou multicolinearidade severa: taxa_2023,
meta_2024 e nivel_oficial tem correlacao acima de 0,96 entre si. Quando
variaveis dizem quase a mesma coisa, o modelo pode atribuir a
importancia a qualquer uma delas, e o ranking vira loteria.

Isso afeta TODAS as tecnicas de interpretacao - coeficientes,
permutation importance e SHAP. Nenhuma resolve sozinha.

Por isso este script nao se limita a produzir um ranking: ele mede se
o ranking se sustenta. A analise de estabilidade refaz a importancia
com varias sementes aleatorias e reporta a variacao. Variavel cuja
posicao oscila muito nao suporta afirmacao causal.

TECNICAS
--------
1. Coeficientes    - so para modelos lineares; dizem direcao e forca
2. Permutation     - embaralha uma variavel e mede quanto a metrica
                     piora; funciona para qualquer modelo
3. SHAP            - atribui a cada previsao a contribuicao de cada
                     variavel; opcional, pois exige a biblioteca shap

Execucao:
    python src/evaluation/interpretar_modelo.py
    python src/evaluation/interpretar_modelo.py --sementes 10
"""

from pathlib import Path
import argparse
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # backend sem janela, para rodar em qualquer ambiente
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import joblib

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_PROCESSED = RAIZ_PROJETO / "data" / "processed"
PASTA_MODELOS = RAIZ_PROJETO / "models"
PASTA_IMAGES = RAIZ_PROJETO / "images"
PASTA_REPORTS = RAIZ_PROJETO / "reports"

ALVO = "atingiu_meta"

# Paleta divergente validada para daltonismo (delta E 18,4 em protanopia).
# Azul e laranja sao o par seguro para polaridade; verde e vermelho nao.
COR_POSITIVA = "#185FA5"
COR_NEGATIVA = "#C4553B"
COR_NEUTRA = "#2A9D8F"
COR_GRID = "#E3E9EC"
COR_TEXTO = "#1C2B33"


def carregar():
    """Le o modelo treinado e a base, refazendo a mesma separacao."""
    caminho_modelo = PASTA_MODELOS / "modelo.joblib"
    if not caminho_modelo.exists():
        print(
            "ERRO: modelo nao encontrado.\n"
            "      Rode antes: python src/modeling/treinar_modelo.py",
            file=sys.stderr,
        )
        sys.exit(1)

    modelo = joblib.load(caminho_modelo)

    caminho_rel = PASTA_REPORTS / "resultado_modelo.json"
    with open(caminho_rel, encoding="utf-8") as f:
        relatorio = json.load(f)

    base = pd.read_parquet(
        PASTA_PROCESSED / "base_enriquecida.parquet", engine="pyarrow"
    )
    X = base[relatorio["variaveis"]].copy()
    y = base[ALVO].copy()

    # Mesma semente e proporcao do treino: garante que o conjunto de
    # teste aqui seja exatamente o mesmo que o modelo nunca viu.
    _, X_teste, _, y_teste = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=relatorio["semente"]
    )
    return modelo, X, y, X_teste, y_teste, relatorio


def nomes_das_colunas(modelo) -> list:
    """Recupera os nomes gerados pelo pre-processamento.

    O one-hot transforma uma coluna em varias, entao os nomes de saida
    diferem dos de entrada.
    """
    try:
        return list(modelo.named_steps["preproc"].get_feature_names_out())
    except Exception:
        return []


def analisar_coeficientes(modelo) -> pd.DataFrame | None:
    """Extrai coeficientes, se o modelo final for linear.

    Em regressao logistica o coeficiente indica direcao e forca: valor
    positivo aumenta a chance de atingir a meta, negativo diminui. Como
    as variaveis foram padronizadas, as magnitudes sao comparaveis
    entre si.
    """
    final = modelo.named_steps["modelo"]
    if not hasattr(final, "coef_"):
        return None

    nomes = nomes_das_colunas(modelo)
    coefs = final.coef_[0]
    if len(nomes) != len(coefs):
        return None

    return (
        pd.DataFrame({"variavel": nomes, "coeficiente": coefs})
        .assign(magnitude=lambda d: d.coeficiente.abs())
        .sort_values("magnitude", ascending=False)
        .reset_index(drop=True)
    )


def analisar_permutacao(modelo, X_teste, y_teste, semente: int,
                        repeticoes: int = 15) -> pd.DataFrame:
    """Mede a queda de desempenho ao embaralhar cada variavel.

    A logica e direta: se embaralhar uma coluna nao piora o modelo,
    aquela coluna nao estava sendo usada. A medicao e feita no conjunto
    de teste, nao no treino, para refletir utilidade real.

    ATENCAO: com variaveis redundantes, esta tecnica subestima ambas.
    Embaralhar uma delas nao piora nada porque a outra carrega a mesma
    informacao.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(
            modelo, X_teste, y_teste,
            n_repeats=repeticoes, random_state=semente,
            scoring="roc_auc", n_jobs=-1,
        )
    return (
        pd.DataFrame({
            "variavel": X_teste.columns,
            "queda_auc": r.importances_mean,
            "desvio": r.importances_std,
        })
        .sort_values("queda_auc", ascending=False)
        .reset_index(drop=True)
    )


def testar_estabilidade(modelo, X_teste, y_teste, n_sementes: int) -> pd.DataFrame:
    """Refaz a importancia com varias sementes e mede a variacao.

    Se a posicao de uma variavel no ranking muda a cada execucao, a
    interpretacao individual dela nao e confiavel - e melhor saber
    disso do que publicar um grafico bonito e instavel.
    """
    posicoes = {c: [] for c in X_teste.columns}

    for semente in range(n_sementes):
        imp = analisar_permutacao(modelo, X_teste, y_teste, semente, repeticoes=5)
        for posicao, variavel in enumerate(imp.variavel, start=1):
            posicoes[variavel].append(posicao)

    return (
        pd.DataFrame({
            "variavel": list(posicoes),
            "posicao_media": [np.mean(p) for p in posicoes.values()],
            "posicao_min": [min(p) for p in posicoes.values()],
            "posicao_max": [max(p) for p in posicoes.values()],
        })
        .assign(amplitude=lambda d: d.posicao_max - d.posicao_min)
        .sort_values("posicao_media")
        .reset_index(drop=True)
    )


def analisar_shap(modelo, X_teste, amostras: int = 300):
    """Calcula valores SHAP, se a biblioteca estiver disponivel.

    SHAP atribui a cada previsao individual a contribuicao de cada
    variavel, o que permite explicar casos especificos - util para um
    gestor perguntar "por que este municipio foi classificado assim".

    A analise e opcional: se a biblioteca nao estiver instalada ou o
    calculo falhar, o script segue com as outras tecnicas.
    """
    try:
        import shap
    except ImportError:
        print("  [AUSENTE] biblioteca shap - pulando esta analise")
        print("            instale com: pip install shap")
        return None

    try:
        amostra = X_teste.sample(min(amostras, len(X_teste)), random_state=0)
        transformado = modelo.named_steps["preproc"].transform(amostra)
        nomes = nomes_das_colunas(modelo)
        final = modelo.named_steps["modelo"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Explainer generico: funciona com qualquer modelo, ainda
            # que mais devagar que os especializados.
            explicador = shap.Explainer(
                final.predict_proba, transformado,
                feature_names=nomes, max_evals=800,
            )
            valores = explicador(transformado[:100])

        importancia = np.abs(valores.values[:, :, 1]).mean(axis=0)
        return (
            pd.DataFrame({"variavel": nomes, "shap_medio": importancia})
            .sort_values("shap_medio", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as erro:
        print(f"  [AVISO] SHAP falhou: {type(erro).__name__}")
        return None


# ---------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------
def preparar_eixo(ax):
    """Aplica o estilo comum: grid recessivo, sem bordas desnecessarias."""
    ax.xaxis.grid(True, color=COR_GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COR_GRID)
    ax.tick_params(colors=COR_TEXTO, labelsize=9, length=0)


def grafico_coeficientes(coefs: pd.DataFrame, destino: Path, limite: int = 12):
    """Barras divergentes: direcao e forca de cada variavel."""
    dados = coefs.head(limite).iloc[::-1]
    cores = [COR_POSITIVA if c > 0 else COR_NEGATIVA for c in dados.coeficiente]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(dados) + 1.6))
    ax.barh(dados.variavel, dados.coeficiente, color=cores, height=0.62)
    ax.axvline(0, color=COR_TEXTO, linewidth=0.8)
    preparar_eixo(ax)
    ax.set_xlabel("coeficiente padronizado", color=COR_TEXTO, fontsize=10)
    ax.set_title(
        "Direcao e forca de cada variavel\n"
        "azul aumenta a chance de atingir a meta; laranja diminui",
        color=COR_TEXTO, fontsize=12, loc="left", pad=14,
    )
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grafico_permutacao(imp: pd.DataFrame, destino: Path):
    """Barras de magnitude, com barra de erro da repeticao."""
    dados = imp.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(dados) + 1.6))
    ax.barh(dados.variavel, dados.queda_auc, xerr=dados.desvio,
            color=COR_NEUTRA, height=0.62,
            error_kw={"ecolor": COR_TEXTO, "elinewidth": 0.9, "capsize": 3})
    ax.axvline(0, color=COR_TEXTO, linewidth=0.8)
    preparar_eixo(ax)
    ax.set_xlabel("queda de AUC ao embaralhar a variavel",
                  color=COR_TEXTO, fontsize=10)
    ax.set_title(
        "Quanto o modelo perde sem cada variavel\n"
        "barras de erro mostram a variacao entre repeticoes",
        color=COR_TEXTO, fontsize=12, loc="left", pad=14,
    )
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grafico_estabilidade(est: pd.DataFrame, destino: Path):
    """Faixa da posicao no ranking entre sementes diferentes."""
    dados = est.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(dados) + 1.8))
    for i, linha in enumerate(dados.itertuples()):
        instavel = linha.amplitude >= 3
        cor = COR_NEGATIVA if instavel else COR_NEUTRA
        ax.plot([linha.posicao_min, linha.posicao_max], [i, i],
                color=cor, linewidth=3, solid_capstyle="round", alpha=0.75)
        ax.plot(linha.posicao_media, i, "o", color=cor, markersize=7)

    ax.set_yticks(range(len(dados)))
    ax.set_yticklabels(dados.variavel)
    preparar_eixo(ax)
    ax.invert_xaxis()
    ax.set_xlabel("posicao no ranking de importancia (1 = mais importante)",
                  color=COR_TEXTO, fontsize=10)
    ax.set_title(
        "Estabilidade do ranking entre sementes aleatorias\n"
        "faixa longa em laranja = posicao nao confiavel",
        color=COR_TEXTO, fontsize=12, loc="left", pad=14,
    )
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interpreta o modelo treinado")
    parser.add_argument("--sementes", type=int, default=8,
                        help="quantas sementes usar no teste de estabilidade")
    args = parser.parse_args()

    print("\nInterpretabilidade do modelo")
    print("=" * 76)

    modelo, X, y, X_teste, y_teste, relatorio = carregar()
    print(f"  modelo: {relatorio['modelo_escolhido']}")
    print(f"  avaliando em {len(X_teste)} municipios do conjunto de teste")

    PASTA_IMAGES.mkdir(parents=True, exist_ok=True)
    saida = {}

    # ---------------- coeficientes ----------------
    coefs = analisar_coeficientes(modelo)
    if coefs is not None:
        print(f"\n  Coeficientes (modelo linear)")
        print("  " + "-" * 72)
        print(f"  {'variavel':38} {'coef':>8}  efeito")
        for _, r in coefs.head(10).iterrows():
            efeito = "aumenta a chance" if r.coeficiente > 0 else "reduz a chance"
            print(f"  {r.variavel:38} {r.coeficiente:+8.3f}  {efeito}")
        grafico_coeficientes(coefs, PASTA_IMAGES / "coeficientes.png")
        saida["coeficientes"] = coefs.to_dict("records")
    else:
        print("\n  modelo nao linear - coeficientes nao se aplicam")

    # ---------------- permutacao ----------------
    print(f"\n  Permutation importance")
    print("  " + "-" * 72)
    imp = analisar_permutacao(modelo, X_teste, y_teste, relatorio["semente"])
    print(f"  {'variavel':30} {'queda AUC':>11} {'desvio':>8}")
    for _, r in imp.iterrows():
        print(f"  {r.variavel:30} {r.queda_auc:11.4f} {r.desvio:8.4f}")
    grafico_permutacao(imp, PASTA_IMAGES / "importancia_permutacao.png")
    saida["permutacao"] = imp.to_dict("records")

    # ---------------- estabilidade ----------------
    print(f"\n  Estabilidade do ranking ({args.sementes} sementes)")
    print("  " + "-" * 72)
    est = testar_estabilidade(modelo, X_teste, y_teste, args.sementes)
    print(f"  {'variavel':30} {'pos. media':>11} {'faixa':>12}")
    for _, r in est.iterrows():
        aviso = "  <- instavel" if r.amplitude >= 3 else ""
        print(f"  {r.variavel:30} {r.posicao_media:11.1f} "
              f"{int(r.posicao_min):>5}-{int(r.posicao_max):<5}{aviso}")
    grafico_estabilidade(est, PASTA_IMAGES / "estabilidade_ranking.png")
    saida["estabilidade"] = est.to_dict("records")

    instaveis = est[est.amplitude >= 3]
    if len(instaveis):
        print(f"\n  [ATENCAO] {len(instaveis)} variavel(is) com posicao instavel.")
        print("            A multicolinearidade impede atribuir importancia")
        print("            individual a elas com confianca.")

    # ---------------- SHAP ----------------
    print(f"\n  SHAP")
    print("  " + "-" * 72)
    shap_imp = analisar_shap(modelo, X_teste)
    if shap_imp is not None:
        for _, r in shap_imp.head(10).iterrows():
            print(f"  {r.variavel:38} {r.shap_medio:8.4f}")
        saida["shap"] = shap_imp.to_dict("records")

    # ---------------- persistencia ----------------
    PASTA_REPORTS.mkdir(parents=True, exist_ok=True)
    with open(PASTA_REPORTS / "interpretabilidade.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False, default=float)

    print(f"\n  graficos em images/")
    print(f"  dados em reports/interpretabilidade.json\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())