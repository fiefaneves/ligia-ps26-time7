import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ==========================================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================================
st.set_page_config(
    page_title="CardioSentinel AI",
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

# ==========================================================
# CARGA DOS MODELOS
# ==========================================================
@st.cache_resource
def carregar_recursos():
    # Ajuste os caminhos conforme sua estrutura de pastas
    try:
        modelo = joblib.load('models/modelo_FINAL_Comite.pkl')
        colunas = joblib.load('models/deploy/colunas_treino.pkl')
        # Se tiver scaler, carregue aqui também: scaler = joblib.load('models/scaler.pkl')
        return modelo, colunas
    except FileNotFoundError:
        st.error("Arquivos do modelo não encontrados. Verifique a pasta 'models/'.")
        return None, None

model, feature_columns = carregar_recursos()

# ==========================================================
# INTERFACE DO USUÁRIO
# ==========================================================
st.title("🛡️ CardioSentinel")
st.markdown("### Sistema Inteligente de Triagem Cardíaca")
st.write("Preencha os dados clínicos abaixo para análise de risco de isquemia.")

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
        max_hr = st.number_input("Frequência Cardíaca Máxima", 60, 220, 150)
        resting_hr = st.number_input("Frequência Cardíaca em Repouso", 40, 120, 70)
        heart_rate_reserve = max_hr - resting_hr # Feature calculada!
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

# ==========================================================
# LÓGICA DE PREVISÃO
# ==========================================================
if submit and model is not None:
    # 1. Criar DataFrame com os dados brutos
    # A ordem aqui não importa tanto se reordenarmos depois, mas os nomes DEVEM ser iguais aos do treino
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [max_hr],
        'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca],
        'thal': [thal], 'heart_rate_reserve': [heart_rate_reserve]
        # Adicione aqui outras colunas se o seu modelo usar (ex: thal_1, cp_2 se for one-hot manual)
    })

    # 2. Tratamento de Pré-processamento
    # Se você usou OneHotEncoder (pd.get_dummies) no treino, precisamos recriar as colunas aqui.
    # Como não temos o objeto preprocessor completo aqui, vamos garantir que o DF tenha as mesmas colunas do treino.
    
    # Simula o OneHotEncoding (Exemplo simplificado - ajuste conforme seu pré-processamento real)
    # Se você usou ColumnTransformer no pipeline, basta fazer: X_final = preprocessor.transform(input_data)
    # Se usou get_dummies, precisa alinhar:
    input_data = pd.get_dummies(input_data)
    
    # Garante que todas as colunas do treino existam (preenche com 0 as que faltarem)
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
            
    # Garante a mesma ordem e remove colunas extras
    X_final = input_data[feature_columns]

    # 3. Previsão com o Comitê
    # Probabilidade
    probabilidade = model.predict_proba(X_final)[0][1]
    
    # Aplica o Limiar Otimizado (Aquele que descobrimos no notebook 04)
    LIMIAR = 0.30 # Ajuste conforme o resultado do seu notebook
    predicao = 1 if probabilidade >= LIMIAR else 0

    # 4. Exibição dos Resultados
    st.divider()
    if predicao == 1:
        st.error(f"🚨 ALERTA: Risco de Doença Cardíaca Detectado")
        st.markdown(f"""
            <div class='metric-card'>
                <h2>Probabilidade Estimada: {probabilidade:.1%}</h2>
                <p>O modelo identificou padrões compatíveis com isquemia.</p>
                <p><strong>Ação Recomendada:</strong> Encaminhar para cardiologista imediatamente.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ Risco Baixo")
        st.markdown(f"""
            <div class='metric-card'>
                <h2>Probabilidade Estimada: {probabilidade:.1%}</h2>
                <p>Não foram encontrados indícios fortes de doença cardíaca.</p>
                <p>Acompanhamento de rotina recomendado.</p>
            </div>
        """, unsafe_allow_html=True)

    # Detalhes Técnicos (Opcional, bom para mostrar pro professor)
    with st.expander("Ver detalhes técnicos da inferência"):
        st.write("Input Processado:", X_final)
        st.write(f"Limiar de Decisão Utilizado: {LIMIAR}")