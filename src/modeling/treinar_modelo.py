"""
Treinamento e validacao do modelo preditivo.

Constroi uma pipeline scikit-learn que integra pre-processamento e
modelo em um unico objeto, compara algoritmos por validacao cruzada e
avalia o vencedor em um conjunto de teste nunca visto.

POR QUE PIPELINE
----------------
Duas razoes, uma pratica e uma de correcao estatistica.

Pratica: o objeto treinado carrega consigo todas as transformacoes.
Para prever um municipio novo basta passar o dado cru - nao e preciso
lembrar de imputar, escalonar e codificar na ordem certa.

Correcao: parametros de transformacao (a mediana da imputacao, a media
e o desvio do escalonamento, as categorias do encoder) precisam ser
aprendidos SOMENTE no conjunto de treino. Calcula-los antes da
separacao faz informacao do teste vazar para o treino, e o resultado
fica otimista. Dentro de uma pipeline, a validacao cruzada reajusta o
pre-processamento a cada particao, o que torna esse erro impossivel.

DECISOES VINDAS DA ANALISE EXPLORATORIA
---------------------------------------
- meta_2030 e constante (todos os municipios = 80): removida.
- distancia_2030 tem correlacao -1,000 com taxa_2023: redundante,
  removida.
- Ha multicolinearidade severa entre as variaveis de desempenho
  (taxa, media de portugues, nivel oficial, meta). Modelos de arvore
  lidam bem com isso; modelos lineares ficam instaveis. A comparacao
  inclui os dois tipos para tornar o efeito visivel.
- Variaveis de alta cardinalidade (microrregiao, mesorregiao) ficam
  fora por padrao: one-hot geraria centenas de colunas esparsas.
  Use --alta-cardinalidade para inclui-las via TargetEncoder.

Execucao:
    python src/modeling/treinar_modelo.py
    python src/modeling/treinar_modelo.py --alta-cardinalidade
    python src/modeling/treinar_modelo.py --semente 7
"""

from pathlib import Path
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_PROCESSED = RAIZ_PROJETO / "data" / "processed"
PASTA_REPORTS = RAIZ_PROJETO / "reports"
PASTA_MODELOS = RAIZ_PROJETO / "models"

ALVO = "atingiu_meta"

# Descartadas pela analise exploratoria, nao por acaso.
DESCARTADAS = [
    "meta_2030",        # constante: todos os municipios tem meta 80
    "distancia_2030",   # correlacao -1,000 com taxa_2023
    "populacao",        # mantida apenas em log, para nao duplicar
]

ALTA_CARDINALIDADE = [
    "microrregiao", "mesorregiao", "regiao_imediata", "regiao_intermediaria",
]

IDENTIFICACAO = ["nome_municipio"]

PROPORCAO_TESTE = 0.25
PARTICOES_CV = 5


def ler_base() -> pd.DataFrame:
    caminho = PASTA_PROCESSED / "base_enriquecida.parquet"
    if not caminho.exists():
        caminho = PASTA_PROCESSED / "base_analitica.parquet"
    if not caminho.exists():
        print(
            "ERRO: nenhuma base encontrada em data/processed/\n"
            "      Rode antes:\n"
            "        python src/preprocessing/construir_base_analitica.py\n"
            "        python src/preprocessing/enriquecer_base.py",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  base: {caminho.name}")
    return pd.read_parquet(caminho, engine="pyarrow")


def separar_variaveis(base: pd.DataFrame, usar_alta_card: bool):
    """Separa X e y, descartando o que nao pode ou nao deve entrar."""
    # Colunas tecnicas guardam o resultado do ano-alvo. Deixa-las entrar
    # seria vazamento direto: e delas que o alvo foi calculado.
    tecnicas = [c for c in base.columns if c.startswith("_")]

    fora = set(tecnicas + DESCARTADAS + IDENTIFICACAO + [ALVO])
    if not usar_alta_card:
        fora |= set(ALTA_CARDINALIDADE)

    variaveis = [c for c in base.columns if c not in fora]

    X = base[variaveis].copy()
    y = base[ALVO].copy()

    print(f"  removidas por vazamento     : {sorted(tecnicas)}")
    print(f"  removidas pela analise      : {sorted(set(DESCARTADAS) & set(base.columns))}")
    if not usar_alta_card:
        presentes = sorted(set(ALTA_CARDINALIDADE) & set(base.columns))
        if presentes:
            print(f"  alta cardinalidade fora     : {presentes}")

    return X, y


def montar_preprocessador(X: pd.DataFrame, usar_alta_card: bool) -> ColumnTransformer:
    """Monta o pre-processamento por tipo de coluna.

    Numericas    : imputacao pela mediana + padronizacao
    Categoricas  : imputacao pela moda + one-hot
    Alta cardin. : imputacao pela moda + target encoding

    A mediana e preferida a media na imputacao porque resiste a valores
    extremos - populacao, por exemplo, tem metropoles que puxariam a
    media para cima.

    A padronizacao (media 0, desvio 1) so importa para modelos
    sensiveis a escala, como a regressao logistica. Arvores ignoram
    escala, mas aplicar a todos mantem a pipeline uniforme e nao
    prejudica.
    """
    numericas = X.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = [
        c for c in X.columns
        if c not in numericas and c not in ALTA_CARDINALIDADE
    ]
    altas = [c for c in X.columns if c in ALTA_CARDINALIDADE]

    blocos = [
        ("num", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalonar", StandardScaler()),
        ]), numericas),
        ("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            # handle_unknown='ignore': se aparecer uma categoria nova na
            # previsao (uma UF ausente do treino), o encoder devolve
            # zeros em vez de quebrar.
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categoricas),
    ]

    if altas and usar_alta_card:
        # TargetEncoder substitui a categoria pela media do alvo naquela
        # categoria. Fazer isso na mao vazaria informacao; a versao do
        # scikit-learn usa validacao cruzada interna para evitar isso.
        from sklearn.preprocessing import TargetEncoder
        blocos.append(
            ("alta", Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("codificar", TargetEncoder(random_state=0)),
            ]), altas)
        )

    print(f"\n  numericas ({len(numericas)}) : {numericas}")
    print(f"  categoricas ({len(categoricas)}) : {categoricas}")
    if altas and usar_alta_card:
        print(f"  alta cardinalidade ({len(altas)}) : {altas}")

    return ColumnTransformer(blocos, remainder="drop")


def catalogo_modelos(semente: int) -> dict:
    """Modelos comparados, do mais simples ao mais flexivel.

    O DummyClassifier existe para dar um piso: qualquer modelo que nao
    o supere claramente nao aprendeu nada util.
    """
    return {
        "baseline": DummyClassifier(strategy="prior", random_state=semente),
        "regressao_logistica": LogisticRegression(
            max_iter=2000, random_state=semente
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5,
            random_state=semente, n_jobs=-1,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06,
            max_leaf_nodes=31, l2_regularization=1.0,
            random_state=semente,
        ),
    }


def grades_de_busca() -> dict:
    """Valores de hiperparametro testados para cada modelo.

    O prefixo 'modelo__' aponta para a etapa chamada 'modelo' dentro da
    pipeline. E assim que o GridSearchCV sabe que deve mexer no
    classificador, e nao no pre-processamento.

    As grades sao pequenas de proposito: cada combinacao e treinada
    PARTICOES_CV vezes, entao o custo cresce rapido. Os valores foram
    escolhidos em escala logaritmica (0,01 / 0,1 / 1 / 10), que cobre
    varias ordens de grandeza com poucos pontos.
    """
    return {
        # C controla a regularizacao: valores pequenos = regularizacao
        # forte = coeficientes menores = menos risco de sobreajuste.
        "regressao_logistica": {"modelo__C": [0.01, 0.1, 1.0, 10.0, 100.0]},

        # min_samples_leaf e max_depth limitam o quanto cada arvore pode
        # se especializar. Sao os freios contra decorar o treino.
        "random_forest": {
            "modelo__min_samples_leaf": [1, 5, 15],
            "modelo__max_depth": [None, 12],
        },

        # learning_rate baixo aprende devagar e generaliza melhor;
        # l2_regularization penaliza folhas com valores extremos.
        "gradient_boosting": {
            "modelo__learning_rate": [0.03, 0.06, 0.12],
            "modelo__max_leaf_nodes": [15, 31],
            "modelo__l2_regularization": [0.0, 1.0],
        },
    }


def comparar(X, y, preproc, semente: int, buscar: bool = True):
    """Compara os modelos por validacao cruzada estratificada.

    Estratificada significa que cada particao preserva a proporcao das
    classes. Sem isso, uma particao poderia ficar com poucos exemplos
    de uma classe e distorcer a metrica.

    Com buscar=True, cada modelo passa por um GridSearchCV: em vez de
    um unico conjunto de hiperparametros escolhido na mao, o codigo
    testa varias combinacoes e fica com a melhor.

    POR QUE A BUSCA RODA SO NO TREINO
    ---------------------------------
    X e y aqui sao APENAS o conjunto de treino. Escolher hiperparametros
    olhando o teste seria uma forma silenciosa de vazamento: o teste
    deixaria de ser dado novo e viraria parte do processo de decisao.
    A busca usa validacao cruzada dentro do treino, e o teste continua
    intocado ate a avaliacao final.
    """
    cv = StratifiedKFold(n_splits=PARTICOES_CV, shuffle=True, random_state=semente)
    grades = grades_de_busca() if buscar else {}
    linhas = []
    ajustados = {}

    for nome, modelo in catalogo_modelos(semente).items():
        pipeline = Pipeline([("preproc", preproc), ("modelo", modelo)])
        grade = grades.get(nome)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if grade:
                busca = GridSearchCV(
                    pipeline, grade, scoring="roc_auc", cv=cv,
                    n_jobs=-1, refit=True, error_score="raise",
                )
                busca.fit(X, y)

                # O indice do melhor resultado permite recuperar o
                # desvio-padrao daquela combinacao especifica.
                i = busca.best_index_
                resultados = busca.cv_results_
                linha = {
                    "modelo": nome,
                    "auc": float(busca.best_score_),
                    "auc_desvio": float(resultados["std_test_score"][i]),
                    "combinacoes": int(len(resultados["params"])),
                    "pior_auc": float(np.min(resultados["mean_test_score"])),
                    "parametros": {
                        k.replace("modelo__", ""): v
                        for k, v in busca.best_params_.items()
                    },
                }
                ajustados[nome] = busca.best_estimator_
            else:
                # Sem grade (baseline) ou com --sem-busca: avaliacao
                # simples, com os valores padrao.
                r = cross_validate(
                    pipeline, X, y, cv=cv, scoring=["roc_auc"],
                    n_jobs=-1, error_score="raise",
                )
                linha = {
                    "modelo": nome,
                    "auc": float(r["test_roc_auc"].mean()),
                    "auc_desvio": float(r["test_roc_auc"].std()),
                    "combinacoes": 1,
                    "pior_auc": float(r["test_roc_auc"].mean()),
                    "parametros": {},
                }
                pipeline.fit(X, y)
                ajustados[nome] = pipeline

        linhas.append(linha)

    tabela = pd.DataFrame(linhas).sort_values("auc", ascending=False)
    return tabela, ajustados


def avaliar_no_teste(pipeline, X_teste, y_teste) -> dict:
    """Avalia o modelo final em dados que ele nunca viu."""
    previsto = pipeline.predict(X_teste)
    probabilidade = pipeline.predict_proba(X_teste)[:, 1]

    return {
        "auc": float(roc_auc_score(y_teste, probabilidade)),
        "acuracia": float(accuracy_score(y_teste, previsto)),
        "precisao": float(precision_score(y_teste, previsto)),
        "revocacao": float(recall_score(y_teste, previsto)),
        "f1": float(f1_score(y_teste, previsto)),
        "matriz_confusao": confusion_matrix(y_teste, previsto).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina o modelo preditivo")
    parser.add_argument("--semente", type=int, default=42,
                        help="semente aleatoria, para reprodutibilidade")
    parser.add_argument("--alta-cardinalidade", action="store_true",
                        help="inclui microrregiao e mesorregiao via TargetEncoder")
    parser.add_argument("--sem-busca", action="store_true",
                        help="pula a otimizacao de hiperparametros (mais rapido)")
    args = parser.parse_args()

    print("\nTreinamento do modelo preditivo")
    print("=" * 76)

    base = ler_base()
    X, y = separar_variaveis(base, args.alta_cardinalidade)

    print(f"\n  {len(X)} municipios   {X.shape[1]} variaveis")
    print(f"  alvo: {int(y.sum())} atingiram / {int((1-y).sum())} abaixo "
          f"({y.mean()*100:.1f}% positivos)")

    # A separacao e feita ANTES de qualquer ajuste. O conjunto de teste
    # fica intocado ate a avaliacao final: e a unica forma de estimar
    # desempenho em dados novos sem se enganar.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=PROPORCAO_TESTE, stratify=y, random_state=args.semente
    )
    print(f"  treino: {len(X_treino)}   teste: {len(X_teste)} (intocado)")

    preproc = montar_preprocessador(X, args.alta_cardinalidade)

    buscar = not args.sem_busca
    if buscar:
        total = sum(len(ParameterGrid(g)) for g in grades_de_busca().values())
        print(f"\n  Otimizacao de hiperparametros + validacao cruzada "
              f"({PARTICOES_CV} particoes)")
        print(f"  {total} combinacoes x {PARTICOES_CV} particoes = "
              f"{total * PARTICOES_CV} treinos. Pode levar alguns minutos.")
    else:
        print(f"\n  Comparacao por validacao cruzada ({PARTICOES_CV} particoes)")
        print("  (sem otimizacao: --sem-busca)")
    print("  " + "-" * 72)

    resultados, ajustados = comparar(X_treino, y_treino, preproc,
                                     args.semente, buscar=buscar)

    print(f"  {'modelo':22} {'AUC':>7} {'desvio':>8} {'combin.':>8} {'pior AUC':>9}")
    for _, r in resultados.iterrows():
        print(f"  {r.modelo:22} {r.auc:7.3f} {r.auc_desvio:8.3f} "
              f"{int(r.combinacoes):8} {r.pior_auc:9.3f}")

    if buscar:
        print("\n  Melhores hiperparametros encontrados:")
        for _, r in resultados.iterrows():
            if r.parametros:
                itens = ", ".join(f"{k}={v}" for k, v in r.parametros.items())
                ganho = r.auc - r.pior_auc
                print(f"    {r.modelo:22} {itens}")
                print(f"    {'':22} ganho sobre a pior combinacao: {ganho:+.3f}")

    vencedor = resultados.iloc[0]
    piso = resultados[resultados.modelo == "baseline"].auc.iloc[0]
    print(f"\n  melhor: {vencedor.modelo} (AUC {vencedor.auc:.3f})")
    print(f"  ganho sobre o baseline: {vencedor.auc - piso:+.3f}")

    if vencedor.auc - piso < 0.05:
        print("  [ATENCAO] ganho pequeno - o modelo aprendeu pouco alem do acaso")

    # ---- avalia o vencedor no teste ----
    # O GridSearchCV ja reajustou o melhor conjunto de hiperparametros
    # em TODO o conjunto de treino (refit=True), entao o modelo pronto
    # e simplesmente o que a busca devolveu.
    modelo_final = ajustados[vencedor.modelo]

    metricas = avaliar_no_teste(modelo_final, X_teste, y_teste)

    print(f"\n  Desempenho no conjunto de teste")
    print("  " + "-" * 72)
    print(f"    AUC        {metricas['auc']:.3f}")
    print(f"    acuracia   {metricas['acuracia']:.3f}")
    print(f"    precisao   {metricas['precisao']:.3f}  "
          f"(dos previstos como 'atingiu', quantos atingiram)")
    print(f"    revocacao  {metricas['revocacao']:.3f}  "
          f"(dos que atingiram, quantos o modelo encontrou)")
    print(f"    F1         {metricas['f1']:.3f}")

    mc = metricas["matriz_confusao"]
    print(f"\n    matriz de confusao")
    print(f"                    previsto: abaixo | atingiu")
    print(f"      real abaixo         {mc[0][0]:>6} | {mc[0][1]:>6}")
    print(f"      real atingiu        {mc[1][0]:>6} | {mc[1][1]:>6}")

    diferenca = abs(vencedor.auc - metricas["auc"])
    print(f"\n    AUC validacao {vencedor.auc:.3f} x teste {metricas['auc']:.3f}"
          f"  (diferenca {diferenca:.3f})")
    if diferenca > 0.05:
        print("    [ATENCAO] diferenca alta sugere sobreajuste")
    else:
        print("    diferenca pequena: o modelo generaliza")

    # ---- persistencia ----
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    PASTA_REPORTS.mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo_final, PASTA_MODELOS / "modelo.joblib")

    relatorio = {
        "semente": args.semente,
        "modelo_escolhido": vencedor.modelo,
        "busca_hiperparametros": bool(buscar),
        "hiperparametros_escolhidos": vencedor.parametros,
        "variaveis": list(X.columns),
        "municipios": int(len(X)),
        "proporcao_positivos": float(y.mean()),
        "validacao_cruzada": resultados.to_dict("records"),
        "teste": metricas,
    }
    with open(PASTA_REPORTS / "resultado_modelo.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)

    print(f"\n  modelo salvo em models/modelo.joblib")
    print(f"  metricas em reports/resultado_modelo.json\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
