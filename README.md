# Green MGGP

Identificação de modelos polinomiais NARX por programação genética multigene, com erro e custo aritmético tratados por seleção de Pareto (NSGA-II).

O custo do polinômio sem termos repetidos é `Cρ = a + ρ b`: adições e multiplicações, incluindo coeficientes. Os pesos fixados antes das buscas são `ρ = 1` e `ρK = 729/64`. Custo aritmético não é energia medida.

## Código e resultados

| Arquivo | Função |
|---|---|
| `corrected_experiments.py` | Busca corrigida com os dois pesos, orçamento igual e ablações |
| `green_mggp.py` | Custo aritmético e seleção Green MGGP |
| `mggp_model/` | Pacote MGGP vendorizado, preservado |
| `energy_freerun.py` | Exemplos e funções compartilhadas pelos scripts corrigidos |
| `scripts/analyse_corrected.py` | Análise dos resultados salvos, sem novas buscas |
| `corrected_energy.py` | Reconstrução, análise e medição dos modelos corrigidos |
| `power_meter.py` | Potência por powermetrics e intensidade de carbono por CodeCarbon |

- [Relatório dos resultados](Resultados/corrected_v1/REPORT.md).
- [Resultados e auditoria da sessão de energia](Resultados/corrected_v1/energia/equal_calls/).

São 1.440 buscas salvas: dois pesos, três exemplos e 30 sementes. A comparação principal inclui SO, MO-lex, Green e ablações por termos/nós, com exatamente 4.510 avaliações por corrida. O braço `native` é diagnóstico e mantém os orçamentos originais. O avaliador externo corrige os atrasos igualmente para todos; divergência permanece falha. Os exemplos são gerados pelo código, sem arquivos de dados externos.

## Instalação

Ambiente usado: Python **3.13.9**, macOS e Apple M2. As versões das dependências estão em `requirements.txt`. Resultados numéricos podem variar com plataforma, bibliotecas e implementação de álgebra linear.

```sh
git clone https://github.com/thalitanazare/GreenMGGP.git
cd GreenMGGP
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ./mggp_model
python -m unittest discover -s tests -v
```

O pacote vendorizado `mggp_model/` é de Rafael Ávila e colaboradores, proveniente de [RafaelGAT108/mggp_model](https://github.com/RafaelGAT108/mggp_model). Sua licença MIT está preservada em `mggp_model/LICENSE`. Nenhuma correção do preditor foi aplicada dentro desse pacote. O commit de origem está registrado em `reproduction_manifest.json`, junto aos hashes da importação original.

## Refazer a análise dos resultados salvos

Estes comandos não iniciam buscas evolutivas nem medem energia:

```sh
python scripts/analyse_corrected.py
python corrected_energy.py --analyse
```

A análise atualiza as métricas, os testes estatísticos e os relatórios em `Resultados/corrected_v1/`. A etapa de energia lê os blocos já medidos e recalcula seus resultados; não inicia uma nova sessão de medição.

`MGGP_corrigido.ipynb` e `Energia_corrigida.ipynb` também oferecem leitura e verificação dos resultados. O script `scripts/export_corrected.py` é um exportador editorial legado que depende de fontes do manuscrito ausentes deste repositório; ele não faz parte dos comandos de reprodução acima.

## Executar novamente as buscas

Para manter os resultados distribuídos, use uma pasta nova:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python corrected_experiments.py --out Resultados/reproduced_v1
```

O comando executa ambos os pesos e os dois braços, com 30 sementes por padrão. Pode levar horas, dependendo da máquina. Uma verificação pequena, que não substitui o estudo completo, é:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python corrected_experiments.py --seeds 1 --generations 3 --out Resultados/smoke
```

O runner retoma apenas corridas faltantes e verifica a compatibilidade do protocolo/código. Para resultados novos, não reutilize caches de um protocolo diferente. A análise aceita `--root`; os argumentos podem ser consultados com `python scripts/analyse_corrected.py --help`.

## Energia: preparação e medição separadas

```sh
python corrected_energy.py
```

Por padrão, apenas reconstrói os modelos e verifica custos e validação. A sessão já distribuída tem 311 blocos válidos e 446 modelos válidos entre 450 registros; quatro validações falhadas permanecem ausentes.

A medição requer **macOS/Apple Silicon**, acesso de administrador ao `powermetrics`, computador na tomada e sem outras tarefas. O CodeCarbon fornece apenas intensidade de carbono; o `powermetrics` mede potência CPU+GPU+ANE. Os tempos de execução permitem avaliar a razão entre multiplicação e adição; energia não define o peso da busca.

Para medir os modelos de uma nova reprodução, após suas buscas:

```sh
sudo -v
GMGGP_ENERGY_RUN=1 caffeinate -i python -u corrected_energy.py --root Resultados/reproduced_v1 --measure
```

A sessão dura cerca de uma hora. O script impede sobrescrever `energy_blocks.csv`. Não execute agentes, análises ou buscas durante a medição. Intensidade de carbono: `GMGGP_COUNTRY`, padrão `IRL`. Reproduzir a etapa em outro hardware exige adaptar e documentar o medidor; não atribua as medidas Apple M2 a outra plataforma.

## Interpretação

- O Green não apresentou vantagem consistente de hipervolume de validação sobre a ablação por termos.
- Em E3, a regra de escolha baseada apenas na identificação seleciona modelos com validação ruim, embora os conjuntos contenham modelos melhores nessa métrica.
- A redução de energia canônica frente ao SO é significativa em E1/E2, mas não em E3 após Holm. Executar todos os genes com NumPy muda o resultado, incluindo aumento de energia em E1.
- Os resultados de energia dizem respeito a uma máquina e uma sessão; carbono é estimado a partir de energia e intensidade da rede.

Os notebooks `MGGP_corrigido.ipynb` e `Energia_corrigida.ipynb` oferecem uma interface adicional. Os scripts, protocolos e caches são a referência de reprodução.
