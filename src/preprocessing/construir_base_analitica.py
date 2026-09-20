"""
Construcao da base analitica para modelagem supervisionada.

Le a camada Gold produzida no Tech Challenge da Fase 2 e monta uma
tabela com UMA LINHA POR MUNICIPIO, no formato exigido por um modelo
de classificacao.

DESENHO DO PROBLEMA
-------------------
Pergunta: um municipio atingira a meta de alfabetizacao do proximo ano?

    entrada (2023)  ->  modelo  ->  alvo (2024)

A separacao temporal e a defesa contra data leakage. Tudo que entra
como variavel existe ANTES do momento da previsao:

  - indicadores observados em 2023
  - a meta de 2024, que e pactuada com antecedencia

O resultado de 2024 so aparece na coluna prefixada com '_auditoria_',
que existe para conferencia manual e NUNCA deve ser usada como
variavel de entrada.

UNIDADE DE ANALISE
------------------
O enunciado do desafio fala em prever a alfabetizacao "de um aluno".
Nao existe microdado individual de estudante publicado pelo INEP, por
protecao de dados pessoais: a granularidade minima disponivel e
municipio x ano x serie x rede. A unidade adotada e, portanto, o
MUNICIPIO, e o alvo e o atingimento da meta municipal.

REDE DE ENSINO
--------------
A comparacao usa a rede municipal (codigo 3), que e a rede a que as
metas municipais se referem. Isso foi verificado na Fase 2: a taxa da
tabela de metas coincide com a rede 3 em 100% dos municipios.

Execucao:
    python src/preprocessing/construir_base_analitica.py
"""

from pathlib import Path
import sys

import pandas as pd

# ---------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------
RAIZ_PROJETO = Path(__file__).resolve().parents[2]

# A Gold da Fase 2 fica em um projeto irmao. Ajuste aqui se a sua
# estrutura de pastas for diferente.
PROJETO_FASE2 = RAIZ_PROJETO.parent / "TechChallengeFase2"
PASTA_GOLD = PROJETO_FASE2 / "data" / "gold"
PASTA_SILVER = PROJETO_FASE2 / "data" / "silver"

PASTA_SAIDA = RAIZ_PROJETO / "data" / "processed"

ANO_ENTRADA = 2023
ANO_ALVO = 2024
REDE_MUNICIPAL = 3


def ler_gold(nome: str) -> pd.DataFrame:
    """Le uma tabela da camada Gold da Fase 2.

    A Gold pode estar particionada (uma pasta com subpastas ano=...)
    ou em arquivo unico, conforme a regra de particionamento condicional
    adotada na Fase 2.
    """
    pasta = PASTA_GOLD / nome
    if not pasta.exists():
        print(
            f"ERRO: tabela '{nome}' nao encontrada em {pasta}\n"
            f"      Rode o pipeline da Fase 2 antes:\n"
            f"      cd {PROJETO_FASE2}\n"
            f"      python src/pipeline_batch.py",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_parquet(pasta, engine="pyarrow")


def ler_silver_opcional(nome: str) -> pd.DataFrame | None:
    """Le uma tabela da Silver, devolvendo None se ela nao existir."""
    caminho = PASTA_SILVER / nome / "dados.parquet"
    if not caminho.exists():
        return None
    return pd.read_parquet(caminho, engine="pyarrow")


def montar_entradas() -> pd.DataFrame:
    """Variaveis observadas no ano de entrada."""
    ind = ler_gold("indicador_municipio")
    ind = ind[(ind["ano"] == ANO_ENTRADA) & (ind["rede_codigo"] == REDE_MUNICIPAL)]

    entradas = ind.set_index("id_municipio")[
        ["sigla_uf", "regiao", "taxa_alfabetizacao", "media_portugues"]
    ].copy()
    entradas.columns = [
        "sigla_uf",
        "regiao",
        f"taxa_{ANO_ENTRADA}",
        f"media_portugues_{ANO_ENTRADA}",
    ]
    return entradas


def montar_contexto() -> pd.DataFrame:
    """Participacao na avaliacao do ano de entrada.

    O percentual de participacao mede quantos alunos efetivamente
    fizeram a prova. Um municipio com participacao baixa tem indicador
    menos confiavel, e essa informacao e util tanto como variavel
    quanto como criterio de ponderacao.
    """
    mvr = ler_gold("meta_vs_resultado")
    ctx = mvr[mvr["ano"] == ANO_ENTRADA].set_index("id_municipio")
    colunas = [c for c in ["percentual_participacao"] if c in ctx.columns]
    ctx = ctx[colunas].copy()
    ctx.columns = [f"participacao_{ANO_ENTRADA}"]
    return ctx


def montar_metas() -> pd.DataFrame:
    """Metas pactuadas: a do ano-alvo e a de 2030.

    Ambas sao conhecidas antes do periodo previsto, entao podem entrar
    como variaveis sem configurar vazamento.
    """
    mvr = ler_gold("meta_vs_resultado")
    alvo = mvr[mvr["ano"] == ANO_ALVO].set_index("id_municipio")

    colunas = ["meta_do_ano"]
    if "meta_alfabetizacao_2030" in alvo.columns:
        colunas.append("meta_alfabetizacao_2030")

    metas = alvo[colunas].copy()
    metas.columns = [f"meta_{ANO_ALVO}"] + (
        ["meta_2030"] if len(colunas) > 1 else []
    )
    return metas


def montar_auditoria() -> pd.DataFrame:
    """Resultado observado no ano-alvo.

    ATENCAO: esta coluna define o alvo. Ela e mantida na base apenas
    para conferencia manual, com prefixo '_auditoria_', e jamais deve
    ser usada como variavel de entrada do modelo.
    """
    ind = ler_gold("indicador_municipio")
    obs = ind[(ind["ano"] == ANO_ALVO) & (ind["rede_codigo"] == REDE_MUNICIPAL)]
    aud = obs.set_index("id_municipio")[["taxa_alfabetizacao"]].copy()
    aud.columns = [f"_auditoria_taxa_{ANO_ALVO}"]
    return aud


def acrescentar_nivel_oficial(base: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a classificacao oficial de nivel do INEP, se disponivel.

    A coluna 'nivel_alfabetizacao' (0 a 5) e uma classificacao publicada
    pelo proprio INEP. Ela vive na Silver e nao foi levada para a Gold
    na Fase 2, entao a leitura e opcional: sem ela o pipeline segue.
    """
    meta = ler_silver_opcional("meta_municipio")
    if meta is None or "nivel_alfabetizacao" not in meta.columns:
        print("  [AVISO] nivel oficial do INEP indisponivel - seguindo sem ele")
        return base

    nivel = meta[meta["ano"] == ANO_ENTRADA].set_index("id_municipio")
    base[f"nivel_oficial_{ANO_ENTRADA}"] = nivel["nivel_alfabetizacao"].reindex(
        base.index
    )
    return base


def construir() -> pd.DataFrame:
    """Monta a base analitica completa."""
    base = (
        montar_entradas()
        .join(montar_contexto(), how="inner")
        .join(montar_metas(), how="inner")
        .join(montar_auditoria(), how="inner")
    )

    # Sem meta definida nao ha como avaliar atingimento.
    antes = len(base)
    base = base[base[f"meta_{ANO_ALVO}"].notna()]
    descartados = antes - len(base)
    if descartados:
        print(f"  {descartados} municipios sem meta em {ANO_ALVO} - fora da base")

    base = acrescentar_nivel_oficial(base)

    # ---- variaveis derivadas ----
    # Quanto o municipio precisa avancar para cumprir a meta. E a
    # traducao direta da pergunta de negocio: a meta e sempre um
    # incremento sobre o ponto de partida.
    base["esforco_exigido"] = (
        base[f"meta_{ANO_ALVO}"] - base[f"taxa_{ANO_ENTRADA}"]
    ).round(2)

    if "meta_2030" in base.columns:
        base["distancia_2030"] = (
            base["meta_2030"] - base[f"taxa_{ANO_ENTRADA}"]
        ).round(2)

    # ---- variavel-alvo ----
    base["atingiu_meta"] = (
        base[f"_auditoria_taxa_{ANO_ALVO}"] >= base[f"meta_{ANO_ALVO}"]
    ).astype(int)

    return base.sort_index()


def validar(base: pd.DataFrame) -> None:
    """Verificacoes minimas antes de gravar."""
    print("\n  Validacao:")

    duplicadas = int(base.index.duplicated().sum())
    print(f"    indice unico por municipio  : {'OK' if not duplicadas else duplicadas}")

    tamanho_errado = int((base.index.str.len() != 7).sum())
    print(f"    id_municipio com 7 digitos  : {'OK' if not tamanho_errado else tamanho_errado}")

    taxa = base[f"taxa_{ANO_ENTRADA}"]
    fora = int(((taxa < 0) | (taxa > 100)).sum())
    print(f"    taxa entre 0 e 100          : {'OK' if not fora else fora}")

    equilibrio = base["atingiu_meta"].mean()
    print(f"    equilibrio das classes      : {equilibrio*100:.1f}% positivos")
    if not 0.2 < equilibrio < 0.8:
        print("    [ATENCAO] classes desequilibradas - revisar estrategia")

    nulos = base.isna().sum()
    nulos = nulos[nulos > 0]
    if len(nulos):
        print("    colunas com nulos:")
        for col, n in nulos.items():
            print(f"      {col:28} {n:>5} ({n/len(base)*100:.1f}%)")
    else:
        print("    nulos                       : nenhum")


def main() -> int:
    print("\nConstrucao da base analitica")
    print("-" * 72)
    print(f"  entrada: {ANO_ENTRADA}   alvo: {ANO_ALVO}   rede: municipal")

    base = construir()

    print(f"\n  {len(base)} municipios   {base.shape[1]} colunas")
    print(f"  {int(base.atingiu_meta.sum())} atingiram / "
          f"{int((1 - base.atingiu_meta).sum())} ficaram abaixo")
    print(f"  cobertura: {base.sigla_uf.nunique()} UFs")

    validar(base)

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    destino = PASTA_SAIDA / "base_analitica.parquet"
    base.to_parquet(destino, engine="pyarrow", compression="snappy")

    # Copia em CSV para inspecao manual (abrir no Excel, conferir a olho).
    base.to_csv(PASTA_SAIDA / "base_analitica.csv", encoding="utf-8")

    print(f"\n  gravado em {destino.relative_to(RAIZ_PROJETO)}")
    print("  (copia em CSV no mesmo diretorio, para inspecao)")

    variaveis = [
        c for c in base.columns
        if not c.startswith("_") and c != "atingiu_meta"
    ]
    print(f"\n  Variaveis candidatas ({len(variaveis)}):")
    for v in variaveis:
        print(f"    {v}")
    print(f"\n  Alvo: atingiu_meta")
    print(f"  Fora do modelo: _auditoria_taxa_{ANO_ALVO}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())