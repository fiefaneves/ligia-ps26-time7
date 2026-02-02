# ❤️ CardioAI
> **Desafio LIGIA 2026 - Time 7 (Startup Simulada)**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Sklearn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Concluído-success)

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
📁 ligia-ps26-time7/
│
├── 📁 data/
│   ├── heart-disease-cleveland-uci/    # Dados originais 
│   └── processed/                      # Dados limpos e prontos (.pkl)
│
├── 📁 models/                              # Artefatos do modelo
│   └── modelo_RedesNeurais_Otimizado.pkl   # O Cérebro da IA
│
├── 📓 notebooks/                           # Análise e Treinamento
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_pre_processamento.ipynb (Pipeline Blindado)
│   ├── 03_xx_treinamento_modelos.ipynb      
│   └── 04_comite_decisao.ipynb           
│
├── 📜 app.py                # Aplicação Web (Streamlit)
├── 📜 requirements.txt      # Lista de dependências
└── 📜 README.md             
```

## 🚀 Como Rodar o Projeto (Passo a Passo)
Siga os passos abaixo no seu terminal.

### Pré-requisitos:
- Python 3.8 ou superior
- Git

### Passo 0: Clone o repositório e entre na pasta
```bash
git clone https://github.com/fiefaneves/ligia-ps26-time7
cd ligia-ps26-time7
```

### Passo 1: Criar e Ativar o Ambiente Virtual (Opcional)
```bash
python3 -m venv venv
# Windows:
.venv\Scripts\Activate
# Linux/Mac:
source venv/bin/activate
```

### Passo 2:  Instalar as Dependências
```bash
pip install -r requirements.txt
```

### Passo 3: Executar o Pipeline
A execução deve seguir uma ordem lógica para garantir que os arquivos .pkl existam antes de serem usados:
1. **Pré-processamento (Obrigatório):**
- Abra e execute todas as células do notebook `02_pre_processamento.ipynb`.
- O que ele faz: Limpa os dados, cria as novas features e salva os artefatos na pasta models/deploy e data/processed/.

2. **Treinamento dos Modelos:**
- Execute os notebooks de treino (ex: `03_4_model_SVM.ipynb`).
- Isso vai treinar os algoritmos e salvar os modelos individuais (.pkl) na pasta models/.

3. **Criação do Comitê:**
- Execute o `04_comite_decisao.ipynb`.
- Ele lerá todos os modelos treinados, escolherá os 3 melhores e criará o `modelo_VotingClassifier.pkl`.

### Passo 4: Rodar a aplicação (Interface visual)
Com o modelo final salvo, execute o comando abaixo no terminal para abrir o CardioAI no seu navegador:
```bash
streamlit run app.py
```

## 📊 Resultados Finais
O modelo final (Redes Neurais), operando com um limiar de decisão ajustado para 0.20 (priorizando a segurança do paciente), obteve os seguintes resultados em dados nunca vistos:

- Recall (Capacidade de detectar doentes): ~93%
- Acurácia Global: ~85%
- Segurança: O sistema prioriza o Falso Positivo (alertar um saudável) em vez do Falso Negativo (mandar um doente para casa).

## Metodologia Técnica
1. **Engenharia de Features**
Criamos variáveis sintéticas para melhorar a precisão do modelo, como a Reserva de Frequência Cardíaca: $$ \text{Reserva} = \frac{\text{Thalach}}{220 - \text{Idade}} $$ Isso permite comparar o esforço cardíaco de um jovem com o de um idoso na mesma escala.

2. **Modelos em Teste**
O time está testando múltiplos algoritmos para encontrar o campeão em Recall:
- Regressão Logística -> `03_1_model_RegLog.ipynb`
- Árvore de Decisão -> `03_2_model_ArvDec.ipynb`
- Random Forest -> `03_3_model_RanFor.ipynb`
- SVM -> `03_4_model_SVM.ipynb`
- KNN -> `03_5_model_KNN.ipynb`
- Redes Neurais -> `03_6_model_RN.ipynb`

### 3. A Batalha: Comitê vs. Especialista
Na fase final, tentamos superar os modelos individuais criando um **Comitê de Decisão (Ensemble Learning)**. Utilizamos um *Voting Classifier* com estratégia *Soft Voting* (média das probabilidades) combinando os 3 melhores modelos da fase anterior (Redes Neurais, KNN e RandomForest).

No entanto, a validação no dataset de teste (held-out) revelou um resultado contra-intuitivo:

| Arquitetura | Recall (Sensibilidade) | Diagnóstico |
| :--- | :---: | :--- |
| **Redes Neurais (Individual)** | **92.86%** | 🏆 **Melhor Generalização** |
| Comitê (Ensemble) | 89.29% | Perda de performance |

> **Decisão de Arquitetura:** O modelo de Redes Neurais (Multilayer Perceptron) provou ser um "especialista" tão forte que a mistura com modelos mais fracos (no Comitê) acabou diluindo a precisão. Optamos por seguir com a **Rede Neural**, garantindo menor complexidade de deploy e maior acerto.

### 4. Deploy e Inferência
O modelo final foi encapsulado em uma aplicação **Streamlit**. Para garantir a reprodutibilidade em produção:
1.  O sistema carrega o artefato `preprocessor.pkl` (a régua de normalização original).
2.  Recebe os dados brutos do médico.
3.  Transforma os dados e submete à Rede Neural.
4.  Aplica um **Limiar de Decisão Conservador (0.20)**: Se o modelo tiver mais de 20% de certeza de que é doença, ela emite o alerta. Isso prioriza a segurança do paciente (evita falsos negativos).