"""
Aquisicao de dados territoriais e populacionais do IBGE.

Baixa duas fontes publicas e as salva em data/raw/, sem transformacao.
A juncao com a base analitica e feita por um script separado.

POR QUE SEPARAR AQUISICAO DE TRANSFORMACAO
-------------------------------------------
Se a API mudar ou sair do ar, o dado ja baixado continua disponivel e
o trabalho de transformacao nao se perde. E o mesmo principio da
camada Bronze: guardar a fonte como ela veio.

O script e idempotente por padrao: se o arquivo ja existe, ele nao
baixa de novo. Use --forcar para rebaixar.

FONTES
------
1. Localidades  - nome do municipio, microrregiao, mesorregiao, UF
   https://servicodados.ibge.gov.br/api/v1/localidades/municipios

2. Populacao    - populacao residente estimada por municipio
   API SIDRA, agregado 6579

3. PIB          - Produto Interno Bruto municipal
   API SIDRA, agregado 5938 (PIB dos Municipios)

Execucao:
    python src/preprocessing/baixar_dados_ibge.py
    python src/preprocessing/baixar_dados_ibge.py --forcar
    python src/preprocessing/baixar_dados_ibge.py --ano 2023
"""

from pathlib import Path
import argparse
import gzip
import json
import sys
import unicodedata
import urllib.error
import urllib.request

import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_RAW = RAIZ_PROJETO / "data" / "raw"

URL_LOCALIDADES = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
URL_POPULACAO = "https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/{ano}"

# PIB dos Municipios (agregado 5938). A variavel nao e fixada no codigo:
# o script consulta antes os metadados do agregado e escolhe a variavel
# pelo nome. Isso evita depender de um numero magico que pode mudar.
URL_METADADOS_PIB = "https://servicodados.ibge.gov.br/api/v3/agregados/5938/metadados"
URL_PIB = "https://apisidra.ibge.gov.br/values/t/5938/n6/all/v/{variavel}/p/{ano}"

# Usada apenas se os metadados nao estiverem disponiveis. 37 e o PIB a
# precos correntes; o valor per capita e calculado depois, dividindo
# pela populacao.
VARIAVEL_PIB_PADRAO = "37"

TEMPO_LIMITE = 120  # segundos


def buscar_json(url: str, descricao: str):
    """Faz a requisicao e devolve o JSON, ou encerra com mensagem clara."""
    print(f"  baixando {descricao}...")
    try:
        requisicao = urllib.request.Request(
            url, headers={"User-Agent": "tech-challenge-fase3/1.0"}
        )
        with urllib.request.urlopen(requisicao, timeout=TEMPO_LIMITE) as resposta:
            dados = resposta.read()

        # Servidores costumam comprimir a resposta para economizar banda.
        # O urllib nao descomprime sozinho, ao contrario de navegadores e
        # da biblioteca requests. A deteccao e feita pelos dois primeiros
        # bytes (1f 8b identifica gzip), que e mais confiavel do que ler
        # o cabecalho Content-Encoding: nem todo servidor o envia correto.
        if dados[:2] == b"\x1f\x8b":
            dados = gzip.decompress(dados)

        return json.loads(dados.decode("utf-8"))
    except urllib.error.HTTPError as erro:
        print(f"  [ERRO] o servidor respondeu {erro.code} para {descricao}", file=sys.stderr)
    except urllib.error.URLError as erro:
        print(f"  [ERRO] nao foi possivel alcancar o servidor: {erro.reason}", file=sys.stderr)
        print("         verifique sua conexao ou tente mais tarde", file=sys.stderr)
    except json.JSONDecodeError:
        print(f"  [ERRO] a resposta de {descricao} nao e um JSON valido", file=sys.stderr)
    except TimeoutError:
        print(f"  [ERRO] tempo esgotado ao baixar {descricao}", file=sys.stderr)
    except UnicodeDecodeError:
        print(f"  [ERRO] resposta de {descricao} em formato inesperado", file=sys.stderr)
    return None


def extrair(dicionario: dict, *caminho, padrao=None):
    """Percorre um dicionario aninhado sem quebrar em chave ausente.

    A API de localidades aninha UF dentro de mesorregiao dentro de
    microrregiao. Alguns municipios criados apos a reforma territorial
    de 2017 nao trazem microrregiao, entao a leitura precisa tolerar
    ausencias.
    """
    atual = dicionario
    for chave in caminho:
        if not isinstance(atual, dict) or chave not in atual:
            return padrao
        atual = atual[chave]
    return atual if atual is not None else padrao


def processar_localidades(bruto: list) -> pd.DataFrame:
    """Achata a estrutura aninhada da API em uma tabela."""
    linhas = []
    for item in bruto:
        micro = extrair(item, "microrregiao", padrao={})
        meso = extrair(micro, "mesorregiao", padrao={})
        linhas.append(
            {
                "id_municipio": str(item["id"]).zfill(7),
                "nome_municipio": item.get("nome"),
                "microrregiao": extrair(micro, "nome"),
                "mesorregiao": extrair(meso, "nome"),
                "sigla_uf_ibge": extrair(meso, "UF", "sigla"),
                "regiao_imediata": extrair(item, "regiao-imediata", "nome"),
                "regiao_intermediaria": extrair(
                    item, "regiao-imediata", "regiao-intermediaria", "nome"
                ),
            }
        )
    return pd.DataFrame(linhas)


def processar_populacao(bruto: list) -> pd.DataFrame:
    """Converte a resposta do SIDRA em tabela.

    O SIDRA devolve a primeira linha como cabecalho descritivo, nao
    como dado. Os codigos das colunas sao: D1C = codigo do municipio,
    V = valor da variavel.
    """
    if len(bruto) < 2:
        return pd.DataFrame()

    registros = bruto[1:]  # descarta a linha de cabecalho
    linhas = []
    for item in registros:
        codigo = item.get("D1C")
        valor = item.get("V")
        if not codigo:
            continue
        try:
            populacao = float(valor)
        except (TypeError, ValueError):
            populacao = None  # SIDRA usa '...' e '-' para indisponivel
        linhas.append(
            {"id_municipio": str(codigo).zfill(7), "populacao": populacao}
        )
    return pd.DataFrame(linhas)


def sem_acento(texto: str) -> str:
    """Remove acentos para comparar nomes de variaveis com seguranca."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    ).lower()


def escolher_variavel_pib(metadados) -> tuple[str, str]:
    """Descobre, nos metadados do agregado, qual variavel baixar.

    Prefere o PIB per capita, que ja vem pronto. Se ele nao existir na
    lista, cai para o PIB total, e o calculo por habitante e feito no
    script de enriquecimento.

    Devolve (id_da_variavel, nome_da_variavel).
    """
    if not isinstance(metadados, dict):
        return VARIAVEL_PIB_PADRAO, "desconhecida (metadados indisponiveis)"

    variaveis = metadados.get("variaveis") or []
    for item in variaveis:
        nome = sem_acento(str(item.get("nome", "")))
        if "per capita" in nome:
            print(f"  variavel escolhida: {item['id']} - {item.get('nome')}")
            return str(item["id"]), str(item.get("nome"))

    for item in variaveis:
        nome = sem_acento(str(item.get("nome", "")))
        if "produto interno bruto" in nome and "valor adicionado" not in nome:
            print(f"  variavel escolhida: {item['id']} - {item.get('nome')}")
            print("  (PIB total; o valor por habitante sera calculado depois)")
            return str(item["id"]), str(item.get("nome"))

    return VARIAVEL_PIB_PADRAO, "desconhecida (nome nao encontrado)"


def processar_pib(bruto: list, nome_variavel: str) -> pd.DataFrame:
    """Converte a resposta do SIDRA sobre PIB em tabela.

    Alem do valor, guarda o nome da variavel e a unidade de medida.
    Sem a unidade nao da para saber se o numero esta em reais ou em
    milhares de reais, e um erro de fator 1000 passaria despercebido.
    """
    if len(bruto) < 2:
        return pd.DataFrame()

    # A primeira linha do SIDRA e descritiva: traz os ROTULOS dos campos
    # ("Unidade de Medida"), nao os valores. A unidade real aparece em
    # cada linha de dado, no campo MN.
    unidade = str(bruto[1].get("MN", "")) if len(bruto) > 1 else ""

    linhas = []
    for item in bruto[1:]:
        codigo = item.get("D1C")
        if not codigo:
            continue
        try:
            valor = float(item.get("V"))
        except (TypeError, ValueError):
            valor = None
        linhas.append({
            "id_municipio": str(codigo).zfill(7),
            "pib": valor,
            "pib_variavel": nome_variavel,
            "pib_unidade": unidade,
        })
    return pd.DataFrame(linhas)


def salvar(df: pd.DataFrame, nome: str) -> Path:
    PASTA_RAW.mkdir(parents=True, exist_ok=True)
    destino = PASTA_RAW / f"{nome}.csv"
    df.to_csv(destino, index=False, encoding="utf-8")
    print(f"  {nome:24} {len(df):>6} linhas -> {destino.name}")
    return destino


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa dados do IBGE")
    parser.add_argument("--ano", type=int, default=2022,
                        help="ano da estimativa populacional")
    parser.add_argument("--ano-pib", type=int, default=2021,
                        help="ano do PIB municipal (serie disponivel ate 2021)")
    parser.add_argument("--forcar", action="store_true",
                        help="rebaixa mesmo se o arquivo ja existir")
    args = parser.parse_args()

    print("\nAquisicao de dados do IBGE")
    print("-" * 72)

    sucessos = 0

    # ---------------- localidades ----------------
    destino = PASTA_RAW / "ibge_localidades.csv"
    if destino.exists() and not args.forcar:
        print(f"  ibge_localidades.csv ja existe - pulando (use --forcar)")
        sucessos += 1
    else:
        bruto = buscar_json(URL_LOCALIDADES, "localidades")
        if bruto:
            salvar(processar_localidades(bruto), "ibge_localidades")
            sucessos += 1

    # ---------------- populacao ----------------
    destino = PASTA_RAW / "ibge_populacao.csv"
    if destino.exists() and not args.forcar:
        print(f"  ibge_populacao.csv ja existe - pulando (use --forcar)")
        sucessos += 1
    else:
        bruto = buscar_json(URL_POPULACAO.format(ano=args.ano), f"populacao {args.ano}")
        if bruto:
            tabela = processar_populacao(bruto)
            if len(tabela):
                salvar(tabela, "ibge_populacao")
                sucessos += 1
            else:
                print("  [AVISO] resposta de populacao vazia - verifique o ano")

    # ---------------- PIB municipal ----------------
    destino = PASTA_RAW / "ibge_pib.csv"
    if destino.exists() and not args.forcar:
        print(f"  ibge_pib.csv ja existe - pulando (use --forcar)")
        sucessos += 1
    else:
        metadados = buscar_json(URL_METADADOS_PIB, "metadados do PIB")
        variavel, nome_variavel = escolher_variavel_pib(metadados)
        bruto = buscar_json(
            URL_PIB.format(variavel=variavel, ano=args.ano_pib),
            f"PIB municipal {args.ano_pib}",
        )
        if bruto:
            tabela = processar_pib(bruto, nome_variavel)
            if len(tabela):
                salvar(tabela, "ibge_pib")
                sucessos += 1
            else:
                print("  [AVISO] resposta de PIB vazia - verifique o ano")
                print("          a serie do PIB municipal vai ate 2021")

    print("-" * 72)
    print(f"  {sucessos} de 3 fontes disponiveis em data/raw/\n")

    if sucessos < 3:
        print("  A base pode ser enriquecida com o que foi baixado.")
        print("  O script de juncao trata fontes ausentes sem quebrar.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
