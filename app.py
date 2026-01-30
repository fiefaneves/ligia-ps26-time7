import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="CardioAI",
    page_icon="💓",
    layout="centered"
)

# Estilo CSS
st.markdown("""
<style>
    /* Fundo geral da aplicação */
    .main {
        background-color: #f0f2f6;
    }
    
    /* Estilo dos botões */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }

    .metric-card {
        background-color: #ffffff !important; /* Força fundo branco */
        color: #31333F !important; /* Força texto escuro (cinza chumbo) */
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
    }
    
    .metric-card h2 {
        color: #31333F !important;
        font-weight: 800;
    }
    .metric-card p {
         color: #31333F !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. CARGA DOS MODELOS
@st.cache_resource
def carregar_recursos():
    # Ajuste os caminhos conforme sua estrutura de pastas
    try:
        modelo = joblib.load('models/modelo_RedesNeurais_Otimizado.pkl')
        preprocessor = joblib.load('models/deploy/preprocessor.pkl')
        colunas = joblib.load('models/deploy/colunas_treino.pkl')
        return modelo, colunas, preprocessor
    except FileNotFoundError as e:
        st.error(f"Arquivos do modelo não encontrados. Verifique a pasta 'models/'. Erro: {e}")
        st.stop()
        return None, None, None

model, feature_columns, preprocessor = carregar_recursos()

# 3. INTERFACE DO USUÁRIO
st.title("🛡️ CardioAI")
st.markdown("### Sistema Inteligente de Triagem Cardíaca")
st.write("Preencha os dados clínicos abaixo para avaliação de risco cardíaco.")

with st.form("diagnostico_form"):
    st.subheader("1. Dados do Paciente")
    c1, c2 = st.columns(2)
    
    with c1:
        age = st.number_input("Idade", min_value=18, max_value=100, value=55)
        sex = st.selectbox("Sexo", options=[1, 0], format_func=lambda x: "Masculino" if x == 1 else "Feminino")
        trestbps = st.number_input("Pressão Arterial em Repouso (mm Hg)", 90, 200, 130)
        chol = st.number_input("Colesterol Sérico (mg/dl)", 100, 600, 240)
    
    with c2:
        # Recriando a Feature Engineering (Heart Rate Reserve)
        thalach = st.number_input("Frequência Cardíaca Máxima", 60, 220, 150)
        resting_hr = st.number_input("Frequência Cardíaca em Repouso", 40, 120, 70)
        heart_rate_reserve = thalach - resting_hr # Feature calculada!
        st.info(f"📊 Heart Rate Reserve calculada: {heart_rate_reserve}")
        
        fbs = st.selectbox("Glicemia em Jejum > 120 mg/dl?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")

    st.subheader("2. Avaliação Cardíaca")
    c3, c4 = st.columns(2)
    
    with c3:
        cp = st.selectbox("Tipo de Dor no Peito", [0, 1, 2, 3], 
                          format_func=lambda x: ["Típica", "Atípica", "Não-Anginosa", "Assintomática"][x])
        exang = st.selectbox("Angina Induzida por Exercício?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
        restecg = st.selectbox("Eletrocardiograma em Repouso", [0, 1, 2],
                                format_func=lambda x: ["Normal", "Anormalidade ST-T", "Hipertrofia ventricular esquerda"][x])
        
    with c4:
        oldpeak = st.number_input("Depressão ST (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Inclinação do Segmento ST", [0, 1, 2], 
                             format_func=lambda x: ["Ascendente", "Plano", "Descendente"][x])
        ca = st.selectbox("Número de Vasos Principais (Fluoroscopia)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Talassemia", [0, 1, 2], 
                            format_func=lambda x: ["Normal", "Defeito Fixo", "Defeito Reversível"][x])

    # Botão de Envio
    submit = st.form_submit_button("🔍 ANALISAR RISCO")

# 4. LÓGICA DE PREVISÃO
if submit and model is not None:
    # A. Montar DataFrame Bruto (Mesma estrutura do CSV original, sem o Target e sem FBS)
    input_dict = {
        'age': [age],
        'sex': [sex], 
        'cp': [cp], 
        'trestbps': [trestbps],
        'chol': [chol], 
        # 'fbs': [fbs], # REMOVIDO pois foi dropado no treino
        'restecg': [restecg], 
        'thalach': [thalach],
        'exang': [exang], 
        'oldpeak': [oldpeak], 
        'slope': [slope], 
        'ca': [ca],
        'thal': [thal], 
        'heart_rate_reserve': [heart_rate_reserve]
    }
    X_raw = pd.DataFrame(input_dict)

    # B. Aplicar o Pipeline de Pré-processamento
    # Isso faz o StandardScaler e o OneHotEncoder automaticamente
    try:
        X_processed = preprocessor.transform(X_raw)
    except ValueError as e:
        st.error(f"Erro no processamento dos dados. \nDetalhe: {e}")
        st.stop()

    X_processed_df = pd.DataFrame(X_processed, columns=feature_columns)

    # C. Previsão
    # Limiar ajustado para a Rede Neural (conforme discutido: 0.20 é agressivo para Recall)
    LIMIAR = 0.20 
    proba_doenca = model.predict_proba(X_processed)[0][1]
    predicao = 1 if proba_doenca >= LIMIAR else 0

    # D. Exibição
    st.divider()
    
    col_result, col_detail = st.columns([2, 1])

    with col_result:
        if predicao == 1:
            st.markdown("""
                <div class='metric-card' style='border-left: 10px solid #FF4B4B;'>
                    <h2 style='color: #FF4B4B !important;'>🚨 ALTA PROBABILIDADE DE DOENÇA</h2>
                    <p>O paciente apresenta características clínicas de risco.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='metric-card' style='border-left: 10px solid #28a745;'>
                    <h2 style='color: #28a745 !important;'>✅ BAIXA PROBABILIDADE</h2>
                    <p>Os sinais vitais indicam padrão saudável.</p>
                </div>
            """, unsafe_allow_html=True)

    with col_detail:
        st.markdown(f"""
            <div class='metric-card'>
                <h3>Score de Risco</h3>
                <h1>{proba_doenca:.1%}</h1>
            </div>
        """, unsafe_allow_html=True)

    # Debug (Só aparece se expandir)
    with st.expander("Dados Processados (Debug)"):
        st.write("Dados brutos:", X_raw)
        st.write("Dados transformados (Entrada na Rede Neural):", X_processed)