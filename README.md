# ❤️ Sistema Inteligente de Triagem Cardíaca

> **Desafio LIGIA 2026 - Time 7 (Startup Simulada)**

## 🎯 Sobre o Projeto
Somos uma solução de Inteligência Artificial desenvolvida para auxiliar equipes médicas em Unidades de Pronto Atendimento (UPAs).

Nosso objetivo é combater o erro de triagem em pacientes com doenças cardíacas. Utilizando dados simples (como idade, pressão e tipo de dor), nosso modelo atua como uma "segunda opinião" de segurança, identificando pacientes de alto risco que poderiam passar despercebidos na triagem manual.

* **Eixo Temático:** Saúde e Bem-Estar
* **Dataset Utilizado:** Heart Disease UCI (Cleveland)
* **Métrica Principal:** Recall (Sensibilidade) - *Foco em não deixar nenhum doente ir para casa sem atendimento.*

## 💡 O Problema de Negócio (O "Paradoxo dos Assintomáticos")
Durante nossa Análise Exploratória (EDA), descobrimos um padrão crítico que justifica o uso de IA:

> **72.5% dos pacientes que chegaram sem dor no peito (assintomáticos) estavam, na verdade, doentes.**

Uma triagem humana tradicional, baseada na pergunta *"O senhor sente dor?"*, falharia em detectar a maioria desses casos. O CardioSentinel cruza dados invisíveis (como depressão do segmento ST e frequência cardíaca máxima) para identificar esse risco silencioso.

## 🛠️ Estrutura do Projeto
O projeto simula um fluxo de Data Science profissional, com pré-processamento centralizado e modelagem distribuída:

```text
📁 projeto-cardio-sentinel/
│
├── 📁 data/
│   ├── raw/                 # Dados originais (heart_cleveland_upload.csv)
│   └── processed/           # Dados limpos e prontos (.pkl) gerados pelo notebook 02
│
├── 📁 models/               # Modelos treinados salvos (.pkl)
│
├── 📓 01_analise_exploratoria.ipynb       # EDA: Gráficos, estatísticas e insights de negócio
├── 📓 02_preprocessing_unificado.ipynb    # Engenharia de features e split (Gera os arquivos .pkl)
├── 📓 03_treinamento_TEMPLATE.ipynb       # Base para treino dos modelos (SVM, Random Forest, etc.)
│
├── 📜 requirements.txt      # Lista de dependências
└── 📜 README.md             # Este manual
```

## 🚀 Como Rodar o Projeto (Passo a Passo)

Se você nunca rodou um projeto Python antes, não se preocupe! Siga os passos abaixo no seu terminal (tela preta do VS Code).

### Passo 1: Criar e Ativar o Ambiente Virtual
Isso cria uma "caixa isolada" para não bagunçar seu computador.
```bash
# 1. Criar a venv
python3 -m venv venv

# 2. Ativar a venv (Linux/Mac)
source venv/bin/activate

# 2. Ativar a venv (Windows)
venv\Scripts\activate
```

### Passo 2:  Instalar as Dependências
Agora vamos baixar as ferramentas necessárias (Pandas, Seaborn, Scikit-Learn, etc) listadas no arquivo `requirements.txt.`
```bash
pip install -r requirements.txt
```

### Passo 3: Executar o Pipeline
```text
TO-DO
```

## 📊 Resultados Preliminares
Na nossa análise inicial, identificamos que o dataset Heart Disease é ideal porque:
1. Balanceado: Temos quase a mesma quantidade de pacientes doentes e saudáveis (50/50).
2. Sinais Claros: Variáveis como Dor no Peito Assintomática e Frequência Cardíaca Máxima são fortes indicativos da doença.
3. Auditável: Conseguimos explicar medicamente o porquê de cada previsão.

## Metodologia Técnica
1. **Engenharia de Features**
Criamos variáveis sintéticas para melhorar a precisão do modelo, como a Reserva de Frequência Cardíaca: $$ \text{Reserva} = \frac{\text{Thalach}}{220 - \text{Idade}} $$ Isso permite comparar o esforço cardíaco de um jovem com o de um idoso na mesma escala.

2. **Modelos em Teste**
O time está testando múltiplos algoritmos para encontrar o campeão em Recall:
- Regressão Logística (Baseline)
- Árvore de Decisão (Explicabilidade)
- Random Forest (Robustez)
- SVM (Fronteiras Complexas)
- KNN (Similaridade)
- Redes Neurais (Deep Learning)