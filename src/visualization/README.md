# Visualização

Scripts que geram os gráficos da análise exploratória em `images/`.

Os gráficos de interpretação do modelo (coeficientes, importância,
estabilidade) e da aplicação estratégica (risco por UF, perfis) ficam
junto dos scripts que calculam esses resultados, em `src/evaluation/`,
porque dependem do modelo treinado.

A separação segue o momento do pipeline: a visualização exploratória
acontece **antes** da modelagem e não depende dela.