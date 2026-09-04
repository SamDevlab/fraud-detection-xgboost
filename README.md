# Detecção de Anomalias em Transações com Python

Projeto de Machine Learning para identificar transações financeiras potencialmente fraudulentas em um conjunto de dados altamente desbalanceado.

O pipeline utiliza **XGBoost** para classificação, **SMOTE** para balanceamento apenas do conjunto de treino e métricas adequadas para fraude, como **Precision-Recall** e **AUC-PR**.

## Objetivo

Demonstrar um fluxo completo de classificação para detecção de fraude, cobrindo:

- carregamento de dados reais de referência;
- pré-processamento;
- divisão estratificada entre treino e teste;
- tratamento de desbalanceamento sem contaminar o conjunto de teste;
- treinamento com XGBoost;
- avaliação com matriz de confusão e relatório de classificação;
- curva Precision-Recall;
- análise de importância das variáveis.

## Dataset

O script utiliza o conjunto público de transações de cartão disponibilizado pelo TensorFlow:

```text
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

A variável alvo é `Class`, onde o conjunto contém uma forte diferença entre a quantidade de transações normais e fraudulentas.

## Pipeline

```text
creditcard.csv
      ↓
carregamento com Pandas
      ↓
padronização de Amount
      ↓
divisão estratificada treino/teste
      ↓
SMOTE somente no treino
      ↓
XGBoost
      ↓
predições e probabilidades
      ↓
classification report
matriz de confusão
Precision-Recall / AUC-PR
importância das variáveis
```

Um detalhe importante é que o **SMOTE é aplicado somente depois da separação entre treino e teste**. Isso evita gerar amostras sintéticas usando informação do conjunto de avaliação.

## Modelo

O classificador utilizado é o `XGBClassifier`, configurado com:

```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=1,
    eval_metric="logloss",
    n_jobs=-1
)
```

Como o conjunto de treino já é balanceado com SMOTE, o projeto mantém `scale_pos_weight=1`.

## Avaliação

O script gera:

- precision, recall, F1-score e support por classe;
- matriz de confusão;
- curva Precision-Recall;
- AUC da curva Precision-Recall;
- ranking das 10 variáveis mais importantes segundo o modelo.

A curva Precision-Recall é especialmente útil neste contexto porque accuracy isolada pode ser enganosa quando a classe de fraude é rara.

## Tecnologias

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn / SMOTE
- XGBoost
- Matplotlib
- Seaborn

## Instalação

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências versionadas no repositório:

```bash
pip install -r requirements.txt
```

## Execução

```bash
python main.py
```

O script baixa o dataset, treina o modelo e exibe os relatórios e gráficos de avaliação.

## Organização do ambiente

O repositório agora inclui:

```text
requirements.txt  # dependências Python do experimento
.gitignore        # evita novos ambientes virtuais e artefatos locais
main.py           # pipeline completo
```

> Observação: uma pasta `.venv` antiga ainda pode existir no histórico/estado atual do repositório. O novo `.gitignore` impede que novos arquivos do ambiente local sejam adicionados por engano.

## Limitações

Este projeto é um estudo de Machine Learning e não deve ser tratado como um motor antifraude pronto para produção. Um sistema real exigiria, entre outros pontos:

- validação temporal;
- calibração de threshold;
- monitoramento de drift;
- custo explícito de falsos positivos/falsos negativos;
- validação em dados externos;
- controles de segurança, explicabilidade e auditoria.

## Próximas melhorias

- transformar o pipeline em módulos reutilizáveis;
- incluir validação cruzada e comparação entre modelos;
- otimizar o threshold de decisão com base em Precision/Recall;
- salvar métricas e gráficos como artefatos reproduzíveis;
- adicionar testes automatizados para o pré-processamento.
