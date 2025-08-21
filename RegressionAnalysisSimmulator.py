import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import statsmodels.api as sm

# Set page configuration
st.set_page_config(page_title="Linear Regression Analysis", layout="wide")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import statsmodels.api as sm

# Configuração da página
st.set_page_config(page_title="Análise de Regressão Linear", layout="wide")

# Título e introdução
st.title("Ferramenta de Análise de Regressão Linear")

# Explicação sobre Regressão Linear e Suposições
st.header("Visão Geral da Regressão Linear")
st.write("""
A regressão linear simples é um método estatístico que nos permite estudar a relação entre duas variáveis 
contínuas: uma variável dependente (Y) e uma variável independente (X). A relação é modelada usando uma equação linear:
Y = β₀ + β₁X + ε

Onde:
- β₀ é o intercepto (valor de Y quando X = 0)
- β₁ é a inclinação (mudança em Y para uma mudança unitária em X)
- ε é o termo de erro
""")

st.subheader("Principais Suposições:")
st.write("""
1. **Linearidade**: Os parâmetros estimados por MQO devem ser lineares.
2. **Aleatoriedade**: Os dados utilizados devem ter sido aleatoriamente amostrados da população.
3. **Exogeneidade ou Independência**: Os regressores (variáveis independentes) não são correlacionados com o termo de erro.
4. **Homocedasticidade**: Os resíduos devem ter variância constante.
5. **Independência dos erros**: Não há correlação entre os erros da regressão.
""")

# Carregar e preparar dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv('series_temporais.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    return df

try:
    df = carregar_dados()
    
    # Seleção de variáveis
    variaveis = ['IR', 'GDP', 'CRED', 'ROE', 'LEV', 'NPL', 'INFLA', 'TARG', 'EMBI']
    
    col1, col2 = st.columns(2)
    
    with col1:
        variavel_dependente = st.selectbox(
            'Selecionar Variável Dependente (Y)',
            variaveis
        )
    
    with col2:
        variavel_independente = st.selectbox(
            'Selecionar Variável Independente (X)',
            [var for var in variaveis if var != variavel_dependente]
        )
    
    # Preparar dados para regressão
    X = df[variavel_independente].values.reshape(-1, 1)
    y = df[variavel_dependente].values
    
    # Adicionar constante ao X para statsmodels
    X_com_const = sm.add_constant(X)
    
    # Ajustar modelo de regressão
    modelo = sm.OLS(y, X_com_const).fit()
    
    # Plotar linha de regressão
    st.header("Análise de Regressão")
    
    fig = px.scatter(df, x=variavel_independente, y=variavel_dependente, 
                    title=f'<span style="font-size:48px"><b>Regressão Linear</b></span>: {variavel_dependente} vs {variavel_independente}')
    
    
    # Adicionar linha de regressão
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_com_const = sm.add_constant(x_range)
    y_pred_linha = modelo.predict(X_range_com_const)
    
    # Adicionar linha de regressão com equação
    texto_equacao = f'y = {modelo.params[0]:.2f} + {modelo.params[1]:.2f}x'
    fig.add_trace(
        go.Scatter(x=x_range.flatten(), y=y_pred_linha,
                  mode='lines', name='Linha de Regressão',
                  line=dict(color='red'))
    )
    
    # Atualizar layout para posicionar legenda no topo
    fig.update_layout(
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            orientation="h"
        ),
        title=dict(
            text=f'Regressão Linear: {variavel_dependente} vs {variavel_independente}<br><sub>{texto_equacao}</sub>',
            x=0.5,
            xanchor='center'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas do Modelo
    st.subheader("Estatísticas da Regressão")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Resumo do Modelo:**")
        st.write(f"R-quadrado: {modelo.rsquared:.4f}")
        st.write(f"R-quadrado Ajustado: {modelo.rsquared_adj:.4f}")
        st.write(f"Erro Quadrático Médio: {np.mean(modelo.resid**2):.4f}")
        st.write(f"Raiz do Erro Quadrático Médio: {np.sqrt(np.mean(modelo.resid**2)):.4f}")
    
    with col2:
        st.write("**Estatísticas dos Coeficientes:**")
        st.write(f"Intercepto (β₀): {modelo.params[0]:.4f}")
        st.write(f"Inclinação (β₁): {modelo.params[1]:.4f}")
        st.write(f"Valor-p (inclinação): {modelo.pvalues[1]:.4f}")
    
    # Predição Pontual
    st.subheader("Predição Pontual")
    
    # Permitir ao usuário selecionar um ponto específico
    indice_ponto = st.slider("Selecionar um ponto de dados", 0, len(df)-1, len(df)//2)
    
    x_real = X[indice_ponto][0]
    y_real = y[indice_ponto]
    y_predito = modelo.predict(X_com_const)[indice_ponto]
    
    st.write(f"**Ponto Selecionado:**")
    st.write(f"X ({variavel_independente}): {x_real:.4f}")
    st.write(f"Y Real ({variavel_dependente}): {y_real:.4f}")
    st.write(f"Y Predito: {y_predito:.4f}")
    st.write(f"Erro de Predição: {y_real - y_predito:.4f}")
    
    # Testes de Suposições
    st.subheader("Testes das Suposições do Modelo")
    
    # Teste de normalidade
    _, p_normalidade = stats.normaltest(modelo.resid)
    
    # Teste de heterocedasticidade
    _, p_het, _, _ = het_breuschpagan(modelo.resid, X_com_const)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Teste de Normalidade (D'Agostino-Pearson):**")
        st.write(f"valor-p: {p_normalidade:.4f}")
        st.write("Interpretação: Resíduos são normais se p > 0,05")
    
    with col2:
        st.write("**Teste de Heterocedasticidade (Breusch-Pagan):**")
        st.write(f"valor-p: {p_het:.4f}")
        st.write("Interpretação: Variância é constante se p > 0,05")

except Exception as e:
    st.error(f"Erro: {str(e)}")
    st.write("Por favor, certifique-se de que o arquivo 'series_temporais.csv' está disponível e contém as colunas necessárias.")