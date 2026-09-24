"""
Enriquecimento da base analitica com fontes externas.

Junta a base construida a partir da Gold da Fase 2 com os dados
territoriais, populacionais e economicos baixados do IBGE.

POR QUE ENRIQUECER
------------------
A base de origem descreve apenas o desempenho passado do municipio.
Sem variaveis de contexto - porte, territorio, riqueza - o modelo so
consegue responder "quem ia bem continua bem", o que e circular. As
variaveis externas sao o que permite responder a pergunta de negocio
"quais fatores impactam a alfabetizacao".

TOLERANCIA A FONTES AUSENTES
-----------------------------
Cada fonte e opcional. Se um arquivo nao existir, o script registra a
ausencia e segue com o que tem, em vez de interromper. Isso mantem o
pipeline util mesmo quando uma API externa esta fora do ar.

Execucao:
    python src/preprocessing/enriquecer_base.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_RAW = RAIZ_PROJETO / "data" / "raw"
PASTA_PROCESSED = RAIZ_PROJETO / "data" / "processed"

# Faixas de porte populacional usadas em politica publica brasileira.
# O corte em ordens de grandeza acompanha a mudanca de natureza da
# gestao municipal: quem administra 5 mil habitantes enfrenta um
# problema diferente de quem administra 500 mil.
FAIXAS_PORTE = [
    (0, 20_000, "pequeno_I"),
    (20_000, 50_000, "pequeno_II"),
    (50_000, 100_000, "medio"),
    (100_000, 900_000, "grande"),
    (900_000, float("inf"), "metropole"),
]

# Colunas com muitas categorias distintas. Ficam na base para a analise
# exploratoria, mas exigem cuidado na modelagem: codificar cada uma
# como coluna propria geraria centenas de variaveis esparsas.
ALTA_CARDINALIDADE = ["microrregiao", "mesorregiao", "regiao_imediata",
                      "regiao_intermediaria"]


def ler_base() -> pd.DataFrame:
    """Le a base analitica produzida na etapa anterior."""
    caminho = PASTA_PROCESSED / "base_analitica.parquet"
    if not caminho.exists():
        print(
            f"ERRO: base analitica nao encontrada em {caminho}\n"
            "      Rode antes: python src/preprocessing/construir_base_analitica.py",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_parquet(caminho, engine="pyarrow")


def ler_fonte(nome: str) -> pd.DataFrame | None:
    """Le uma fonte externa de data/raw, devolvendo None se ausente."""
    caminho = PASTA_RAW / f"{nome}.csv"
    if not caminho.exists():
        print(f"  [AUSENTE] {nome}.csv - seguindo sem esta fonte")
        return None
    # id_municipio como texto: preserva zeros a esquerda e garante que
    # a chave dos dois lados do join tenha o mesmo tipo.
    df = pd.read_csv(caminho, dtype={"id_municipio": str})
    df["id_municipio"] = df["id_municipio"].str.zfill(7)
    return df


def medir_cobertura(base: pd.DataFrame, fonte: pd.DataFrame,
                    nome: str) -> float:
    """Calcula e reporta quantos municipios da base encontram par.

    Cobertura baixa quase sempre indica chave malformada, e e melhor
    descobrir agora do que depois de treinar o modelo.
    """
    encontrados = base.index.isin(fonte["id_municipio"]).sum()
    cobertura = encontrados / len(base)
    marca = "OK" if cobertura > 0.95 else "ATENCAO"
    print(f"  [{marca}] {nome:20} cobertura {cobertura*100:5.1f}% "
          f"({encontrados} de {len(base)})")
    return cobertura


def juntar_localidades(base: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta nome do municipio e hierarquia territorial."""
    loc = ler_fonte("ibge_localidades")
    if loc is None:
        return base

    medir_cobertura(base, loc, "localidades")

    colunas = [c for c in [
        "id_municipio", "nome_municipio", "microrregiao", "mesorregiao",
        "regiao_imediata", "regiao_intermediaria", "sigla_uf_ibge",
    ] if c in loc.columns]

    loc = loc[colunas].drop_duplicates("id_municipio").set_index("id_municipio")
    base = base.join(loc, how="left")

    # Conferencia cruzada: a UF derivada do codigo IBGE na Fase 2 deve
    # coincidir com a UF que o proprio IBGE informa. Divergencia aqui
    # significaria erro na derivacao territorial.
    if "sigla_uf_ibge" in base.columns:
        comparavel = base[["sigla_uf", "sigla_uf_ibge"]].dropna()
        divergentes = int((comparavel.sigla_uf != comparavel.sigla_uf_ibge).sum())
        situacao = "OK" if divergentes == 0 else f"{divergentes} divergentes"
        print(f"  [{'OK' if not divergentes else 'ERRO'}] "
              f"{'UF derivada x IBGE':20} {situacao}")
        base = base.drop(columns=["sigla_uf_ibge"])

    return base


def classificar_porte(populacao: float) -> str:
    """Traduz populacao em faixa de porte municipal."""
    if pd.isna(populacao):
        return None
    for minimo, maximo, rotulo in FAIXAS_PORTE:
        if minimo <= populacao < maximo:
            return rotulo
    return "metropole"


def juntar_populacao(base: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta populacao e variaveis derivadas dela."""
    pop = ler_fonte("ibge_populacao")
    if pop is None:
        return base

    medir_cobertura(base, pop, "populacao")

    pop = pop[["id_municipio", "populacao"]].drop_duplicates(
        "id_municipio"
    ).set_index("id_municipio")
    base = base.join(pop, how="left")

    # Populacao e fortemente assimetrica: milhares de municipios com
    # poucos milhares de habitantes e algumas metropoles de milhoes.
    # O logaritmo aproxima a distribuicao de uma normal, o que ajuda
    # modelos lineares e melhora a legibilidade dos graficos.
    # log1p (log de 1+x) evita erro caso algum valor seja zero.
    base["log_populacao"] = np.log1p(base["populacao"])

    base["porte_municipio"] = base["populacao"].apply(classificar_porte)

    return base


def juntar_pib(base: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta o PIB por habitante.

    POR QUE ESTA VARIAVEL
    ---------------------
    O enunciado pede variaveis "educacionais, territoriais e
    socioeconomicas". Ate aqui havia so as duas primeiras. O PIB por
    habitante e a medida socioeconomica municipal mais recente
    disponivel (serie ate 2021), bem mais atual que o IDHM, preso ao
    Censo de 2010.

    Ha uma hipotese a testar junto: parte do peso da UF no modelo pode
    ser efeito economico disfarcado de geografia. Se for, a importancia
    da UF cai quando o PIB entra. Se nao cair, e evidencia de que o
    efeito estadual e institucional - politica educacional, formacao de
    professores, regime de colaboracao - e nao economico.
    """
    pib = ler_fonte("ibge_pib")
    if pib is None:
        return base

    medir_cobertura(base, pib, "pib")

    colunas = [c for c in ["id_municipio", "pib", "pib_variavel", "pib_unidade"]
               if c in pib.columns]
    pib = pib[colunas].drop_duplicates("id_municipio").set_index("id_municipio")
    base = base.join(pib, how="left")

    variavel = _primeiro_valor(base, "pib_variavel")
    unidade = _primeiro_valor(base, "pib_unidade")

    if "per capita" in variavel.lower():
        # O IBGE ja publica o valor por habitante: nada a calcular.
        base["pib_per_capita"] = base["pib"]
        origem = f"direto da fonte ({unidade})"
    elif "populacao" in base.columns:
        # PIB total: converte para reais (a serie costuma vir em mil
        # reais) e divide pela populacao. Um erro de fator 1000 aqui
        # passaria despercebido, por isso a unidade e conferida.
        fator = 1000 if "mil" in unidade.lower() else 1
        base["pib_per_capita"] = base["pib"] * fator / base["populacao"]
        origem = f"calculado: PIB ({unidade}) / populacao"
    else:
        print("  [AVISO] PIB total sem populacao para dividir - variavel ignorada")
        return base.drop(columns=[c for c in ["pib", "pib_variavel", "pib_unidade"]
                                  if c in base.columns])

    # Mesma razao do log da populacao: a distribuicao de renda entre
    # municipios e muito assimetrica, com poucos casos extremos
    # (municipios com refinaria, mineracao ou porto).
    base["log_pib_per_capita"] = np.log1p(base["pib_per_capita"])

    validos = base["pib_per_capita"].dropna()
    if len(validos):
        print(f"  PIB por habitante ({origem})")
        print(f"      mediana R$ {validos.median():>12,.0f}")
        print(f"      minimo   R$ {validos.min():>12,.0f}")
        print(f"      maximo   R$ {validos.max():>12,.0f}")
        print(f"      assimetria {validos.skew():+.1f} -> justifica o log")
        if validos.median() < 1000 or validos.median() > 500_000:
            print("      [ATENCAO] mediana fora da faixa plausivel - conferir unidade")

    return base.drop(columns=[c for c in ["pib", "pib_variavel", "pib_unidade"]
                              if c in base.columns])


def _primeiro_valor(base: pd.DataFrame, coluna: str) -> str:
    """Le o primeiro valor nao nulo de uma coluna de metadado."""
    if coluna not in base.columns:
        return ""
    valores = base[coluna].dropna()
    return str(valores.iloc[0]) if len(valores) else ""


def relatar(base: pd.DataFrame) -> None:
    """Resume o estado final da base enriquecida."""
    print("\n  Composicao final:")

    tecnicas = [c for c in base.columns if c.startswith("_")]
    alvo = ["atingiu_meta"]
    alta_card = [c for c in ALTA_CARDINALIDADE if c in base.columns]
    identificacao = [c for c in ["nome_municipio"] if c in base.columns]

    modelaveis = [
        c for c in base.columns
        if c not in tecnicas + alvo + alta_card + identificacao
    ]

    print(f"    variaveis para o modelo   : {len(modelaveis)}")
    for c in modelaveis:
        tipo = "num" if pd.api.types.is_numeric_dtype(base[c]) else "cat"
        nulos = base[c].isna().mean() * 100
        aviso = f"  {nulos:.1f}% nulo" if nulos else ""
        print(f"      [{tipo}] {c}{aviso}")

    if alta_card:
        print(f"    alta cardinalidade        : {len(alta_card)}")
        for c in alta_card:
            print(f"      {c} ({base[c].nunique()} categorias)")
        print("      -> mantidas para analise; exigem codificacao cuidadosa")

    if identificacao:
        print(f"    identificacao (nao modela): {', '.join(identificacao)}")
    print(f"    tecnicas (auditoria)      : {', '.join(tecnicas)}")
    print(f"    alvo                      : atingiu_meta")


def main() -> int:
    print("\nEnriquecimento da base analitica")
    print("-" * 72)

    base = ler_base()
    print(f"  base de origem: {len(base)} municipios, {base.shape[1]} colunas\n")

    base = juntar_localidades(base)
    base = juntar_populacao(base)
    base = juntar_pib(base)

    print(f"\n  base enriquecida: {len(base)} municipios, {base.shape[1]} colunas")

    relatar(base)

    destino = PASTA_PROCESSED / "base_enriquecida.parquet"
    base.to_parquet(destino, engine="pyarrow", compression="snappy")
    base.to_csv(PASTA_PROCESSED / "base_enriquecida.csv", encoding="utf-8")

    print(f"\n  gravado em {destino.relative_to(RAIZ_PROJETO)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
