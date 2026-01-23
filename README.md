# ❤️ Sistema Inteligente de Triagem Cardíaca

> **Desafio LIGIA 2026 - Time 7 (Startup Simulada)**

## 🎯 Sobre o Projeto
Somos uma solução de Inteligência Artificial desenvolvida para auxiliar equipes médicas em Unidades de Pronto Atendimento (UPAs).

Nosso objetivo é combater o erro de triagem em pacientes com doenças cardíacas. Utilizando dados simples (como idade, pressão e tipo de dor), nosso modelo atua como uma "segunda opinião" de segurança, identificando pacientes de alto risco que poderiam passar despercebidos na triagem manual.

* **Eixo Temático:** Saúde e Bem-Estar
* **Dataset Utilizado:** Heart Disease UCI (Cleveland)
* **Métrica Principal:** Recall (Sensibilidade) - *Foco em não deixar nenhum doente ir para casa sem atendimento.*

## 🛠️ Estrutura do Projeto
Para facilitar o entendimento, organizamos as pastas da seguinte forma:

* `data/`: Contém os arquivos de dados brutos (CSV). O arquivo principal é o `heart_cleveland_upload.csv`.
* `venv/`: É o nosso "Ambiente Virtual". Pense nele como uma caixa isolada onde instalamos as ferramentas do projeto sem bagunçar o seu computador.
* `requirements.txt`: Uma lista de compras. Diz ao Python exatamente quais bibliotecas (ingredientes) precisamos instalar.
* `01_analise_exploratoria.ipynb`: O caderno onde fazemos a investigação dos dados (gráficos e estatísticas).
* `README.md`: Este manual que você está lendo.

## 🚀 Como Rodar o Projeto (Passo a Passo)

Se você nunca rodou um projeto Python antes, não se preocupe! Siga os passos abaixo no seu terminal (tela preta do VS Code).

### Passo 1: Criar o Ambiente Virtual
Isso cria a pasta `venv` para isolar nosso projeto.
```bash
python3 -m venv venv
```

### Passo 2: Ativar o Ambiente
Isso "liga" o ambiente. Você verá (venv) aparecer no começo da linha do terminal.
```bash
# No Linux/Mac (Nosso caso):
source venv/bin/activate

# No Windows:
venv\Scripts\activate
```

### Passo 3: Instalar as Dependências
Agora vamos baixar as ferramentas necessárias (Pandas, Seaborn, Scikit-Learn, etc) listadas no arquivo `requirements.txt.`
```bash
pip install -r requirements.txt
```

## 📊 O Que Já Descobrimos (Resultados Preliminares)
Na nossa análise inicial, identificamos que o dataset Heart Disease é ideal porque:
1. Balanceado: Temos quase a mesma quantidade de pacientes doentes e saudáveis (50/50).
2. Sinais Claros: Variáveis como Dor no Peito Assintomática e Frequência Cardíaca Máxima são fortes indicativos da doença.
3. Auditável: Conseguimos explicar medicamente o porquê de cada previsão.
