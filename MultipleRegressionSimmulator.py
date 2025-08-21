import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import statsmodels.api as sm

# Centralizar o título usando HTML
st.markdown(
    "<h5 style='text-align: center;'>Faça sua escolha!</h5>",
    unsafe_allow_html=True
)

options = ["Regressão Linear Simples", "Regressão Linear Múltipla"]

# Criar três colunas, colocando o espaço mais amplo (ou espaço igual) nas bordas
col1, col2, col3 = st.columns([1, 2, 1])

# Colocar o botão de rádio dentro da coluna do meio
with col2:
    geral = st.radio(
        label="", 
        options=options, 
        index=None, 
        disabled=False, 
        horizontal=True
    )
    
if geral == "Regressão Linear Múltipla":

    # Título e introdução
    st.title("Ferramenta de Análise de Regressão Linear Múltipla")

    # Explicação da Regressão Linear Múltipla e Pressupostos
    st.header("Visão Geral da Regressão Linear Múltipla")
    st.write("""
    A regressão linear múltipla estende a regressão linear simples incorporando múltiplas variáveis independentes 
    para prever uma variável dependente. O modelo é representado como:
    Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε

    Onde:
    - β₀ é o intercepto
    - βᵢ são os coeficientes para cada variável independente
    - Xᵢ são as variáveis independentes
    - ε é o termo de erro
    """)

    st.subheader("Pressupostos Principais:")
    st.write("""
    1. **Linearidade**: A relação entre variáveis independentes e a variável dependente deve ser linear
    2. **Independência**: Os resíduos devem ser independentes
    3. **Normalidade**: Os resíduos devem ser normalmente distribuídos
    4. **Homocedasticidade**: Os resíduos devem ter variância constante
    5. **Ausência de Multicolinearidade**: As variáveis independentes não devem ser altamente correlacionadas entre si
    """)

    # Carregar e preparar dados
    @st.cache_data
    def load_data():
        df = pd.read_csv('series_temporais.csv')
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df

    try:
        df = load_data()
        
        # Seleção de variáveis
        variables = ['CRED', 'IR', 'GDP', 'ROE', 'LEV', 'NPL', 'INFLA', 'TARG', 'EMBI']
        
        # Selecionar variável dependente
        dependent_var = st.selectbox(
            'Selecione a Variável Dependente (Y)',
            variables
        )
        
        # Selecionar múltiplas variáveis independentes
        remaining_vars = [var for var in variables if var != dependent_var]
        independent_vars = st.multiselect(
            'Selecione as Variáveis Independentes (X)',
            remaining_vars,
            default=remaining_vars[:2]  # Padrão selecionar as duas primeiras variáveis
        )
        
        if len(independent_vars) < 1:
            st.warning("Por favor, selecione pelo menos uma variável independente.")
        else:
            # Preparar dados para regressão
            X = df[independent_vars]
            y = df[dependent_var]
            
            # Adicionar constante ao X para statsmodels
            X_with_const = sm.add_constant(X)
            
            # Ajustar modelo de regressão
            model = sm.OLS(y, X_with_const).fit()
            
            # Exibir resultados da regressão
            st.header("Análise de Regressão")
            
            # Estatísticas do Modelo
            st.subheader("Desempenho do Modelo")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Ajuste Geral do Modelo:**")
                st.write(f"R-quadrado: {model.rsquared:.4f}")
                st.write(f"R-quadrado Ajustado: {model.rsquared_adj:.4f}")
                st.write(f"Estatística F: {model.fvalue:.4f}")
                st.write(f"P-valor da Estatística F: {model.f_pvalue:.4f}")
            
            with col2:
                st.write("**Métricas de Erro:**")
                st.write(f"Erro Quadrático Médio: {np.mean(model.resid**2):.4f}")
                st.write(f"Raiz do Erro Quadrático Médio: {np.sqrt(np.mean(model.resid**2)):.4f}")
                st.write(f"AIC: {model.aic:.4f}")
                st.write(f"BIC: {model.bic:.4f}")
            
            # Análise de Coeficientes
            st.subheader("Análise de Coeficientes")
            coef_df = pd.DataFrame({
                'Coeficiente': model.params.round(4),
                'Erro Padrão': model.bse.round(4),
                'Valor-t': model.tvalues.round(4),
                'P-valor': model.pvalues.round(4)
            })
            st.write(coef_df.style.format({
                'Coeficiente': '{:.4f}',
                'Erro Padrão': '{:.4f}',
                'Valor-t': '{:.4f}',
                'P-valor': '{:.4f}'
            }))
            
            # Análise de Multicolinearidade
            st.subheader("Análise de Multicolinearidade")
            
            st.write("""
            VIF significa **Fator de Inflação da Variância**. É uma medida numérica usada na análise de regressão para quantificar 
            o quanto a variância de um coeficiente de regressão é inflada devido à colinearidade (correlação) entre 
            as variáveis explicativas.

            Aqui está a ideia principal:
            * Para cada preditor xᵢ no modelo, você regride xᵢ sobre os outros preditores para obter R²ᵢ, o 
            coeficiente de determinação dessa regressão auxiliar.
            * O Fator de Inflação da Variância para xᵢ é calculado como VIFᵢ = 1/(1 - R²ᵢ).

            Se R²ᵢ é alto (significando que xᵢ pode ser bem explicado pelos outros preditores), então 1 - R²ᵢ é pequeno, 
            então VIFᵢ será grande, indicando um potencial problema de multicolinearidade.

            Uma regra prática comum é que:
            * VIF > 5 (às vezes 10, dependendo do campo e contexto) pode indicar níveis problemáticos de 
            multicolinearidade que poderiam enviesar os resultados da regressão ou tornar as estimativas dos 
            coeficientes instáveis.
            """)
            
            # Calcular e exibir valores VIF
            vif_data = pd.DataFrame()
            vif_data["Variável"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif_data["VIF"] = vif_data["VIF"].round(4)  # Arredondar para 4 casas decimais
            st.write(vif_data)
            
            # Gráfico Valores Reais vs Preditos
            st.subheader("Valores Reais vs Valores Preditos")
            fig_pred = px.scatter(
                x=y, y=model.fittedvalues,
                labels={'x': 'Valores Reais', 'y': 'Valores Preditos'},
                title='Valores Reais vs Valores Preditos'
            )
            
            # Adicionar linha de 45 graus
            fig_pred.add_trace(
                go.Scatter(
                    x=[y.min(), y.max()],
                    y=[y.min(), y.max()],
                    mode='lines',
                    name='Predição Perfeita',
                    line=dict(dash='dash', color='red')
                )
            )
            
            # Atualizar layout para melhor visualização
            fig_pred.update_layout(
                legend=dict(
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    orientation="h"
                )
            )
            
            st.plotly_chart(fig_pred, use_container_width=True, key='pred_plot')
            
            # Análise de Resíduos
            st.subheader("Análise de Resíduos")

            # Calcular e exibir média dos resíduos
            mean_residuals = np.mean(model.resid)
            st.write(f"Média dos Resíduos: {mean_residuals:.4f}")
            
            st.write("""
            Um bom gráfico "Resíduos vs Valores Ajustados" para uma regressão linear válida geralmente parece com uma 
            "nuvem" não estruturada de pontos em torno da linha horizontal zero, sem padrão ou tendência óbvios. Em particular:

            1. **Média dos Resíduos é Zero**
            Os resíduos devem se agrupar simetricamente em torno de zero, indicando que, em média, a regressão 
            não superestima ou subestima sistematicamente.

            2. **Nenhum Padrão Sistemático**
            Você não quer ver curvas, funis (ou seja, a dispersão ficando maior conforme os valores ajustados aumentam), 
            ou qualquer outra estrutura óbvia. Qualquer padrão pode sugerir problemas como não-linearidade, 
            heterocedasticidade ou variáveis faltantes.

            3. **Variância Aproximadamente Constante**
            A dispersão (variância) dos resíduos deve ser aproximadamente a mesma em toda a faixa de valores ajustados 
            (homocedasticidade). Se a variância parecer crescer (ou diminuir) em uma parte do gráfico, isso 
            indica possível heterocedasticidade.

            Quando essas condições são atendidas—nenhum padrão claro, uma dispersão aproximadamente constante, 
            e um centro próximo de zero—os pressupostos do modelo linear têm mais probabilidade de serem válidos, 
            tornando a análise de regressão mais confiável.
            """)
            
            # Gráfico Resíduos vs Valores Ajustados
            fig_resid = px.scatter(
                x=model.fittedvalues,
                y=model.resid,
                labels={'x': 'Valores Ajustados', 'y': 'Resíduos'},
                title='Resíduos vs Valores Ajustados'
            )
            
            # Adicionar linha horizontal em y=0
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_resid, use_container_width=True, key='resid_plot')
            
            # Gráfico Q-Q com bandas de confiança
            st.subheader("Análise do Gráfico Q-Q Normal")
            
            # Calcular quantis teóricos
            n = len(model.resid)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
            observed_quantiles = np.sort(model.resid)
            
            # Calcular bandas de confiança
            # Usando simulação para gerar bandas de confiança
            n_simulations = 1000
            simulated_bands = np.zeros((n_simulations, n))
            for i in range(n_simulations):
                simulated_data = stats.norm.rvs(size=n)
                simulated_bands[i] = np.sort(simulated_data)
            
            confidence_level = 0.90
            lower_band = np.percentile(simulated_bands, (1 - confidence_level) * 100 / 2, axis=0)
            upper_band = np.percentile(simulated_bands, (1 + confidence_level) * 100 / 2, axis=0)
            
            # Criar gráfico Q-Q com bandas de confiança
            fig_qq = go.Figure()
            
            # Adicionar bandas de confiança
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=upper_band,
                    mode='lines',
                    line=dict(color='rgba(173, 216, 230, 0.5)'),
                    name=f'Banda de Confiança {int(confidence_level * 100)}%',
                    showlegend=False
                )
            )
            
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=lower_band,
                    mode='lines',
                    line=dict(color='rgba(173, 216, 230, 0.5)'),
                    fill='tonexty',
                    name=f'Banda de Confiança {int(confidence_level * 100)}%'
                )
            )
            
            # Adicionar linha Q-Q
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=observed_quantiles,
                    mode='markers',
                    name='Quantis da Amostra',
                    marker=dict(color='blue')
                )
            )
            
            # Adicionar linha de referência de 45 graus
            fig_qq.add_trace(
                go.Scatter(
                    x=[min(theoretical_quantiles), max(theoretical_quantiles)],
                    y=[min(theoretical_quantiles), max(theoretical_quantiles)],
                    mode='lines',
                    name='Linha de Referência',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig_qq.update_layout(
                title='Gráfico Q-Q Normal com Bandas de Confiança',
                xaxis_title='Quantis Teóricos',
                yaxis_title='Quantis da Amostra',
                showlegend=True
            )
            
            st.plotly_chart(fig_qq, use_container_width=True, key='qq_plot')
            
            # Múltiplos Testes de Normalidade
            st.subheader("Testes de Normalidade")
            
            # Teste de D'Agostino-Pearson
            dagostino_stat, dagostino_p = stats.normaltest(model.resid)
            
            # Teste de Shapiro-Wilk
            shapiro_stat, shapiro_p = stats.shapiro(model.resid)
            
            # Teste de Kolmogorov-Smirnov
            ks_stat, ks_p = stats.kstest(stats.zscore(model.resid), 'norm')
            
            # Criar DataFrame com resultados dos testes
            normality_tests = pd.DataFrame({
                'Teste': ['D\'Agostino-Pearson', 'Shapiro-Wilk', 'Kolmogorov-Smirnov'],
                'Estatística': [dagostino_stat, shapiro_stat, ks_stat],
                'p-valor': [dagostino_p, shapiro_p, ks_p]
            })
            
            # Arredondar os valores
            normality_tests['Estatística'] = normality_tests['Estatística'].round(4)
            normality_tests['p-valor'] = normality_tests['p-valor'].round(4)
            
            st.write("""
            **Interpretação dos Testes de Normalidade:**
            - H₀ (hipótese nula): Os dados seguem uma distribuição normal
            - H₁ (hipótese alternativa): Os dados não seguem uma distribuição normal
            - Se p-valor > 0,05, falhamos em rejeitar a hipótese nula (dados parecem normais)
            - Se p-valor ≤ 0,05, rejeitamos a hipótese nula (dados parecem não-normais)
            """)
            
            st.write(normality_tests)
            
            # Predição Pontual
            st.subheader("Predição Pontual")
            st.write("Selecione valores para as variáveis independentes para fazer uma predição:")
            
            # Criar campos de entrada para cada variável independente
            input_values = {}
            for var in independent_vars:
                min_val = float(df[var].min())
                max_val = float(df[var].max())
                default_val = float(df[var].mean())
                input_values[var] = st.slider(
                    f"Selecione o valor para {var}",
                    min_val, max_val, default_val,
                    format="%.2f"
                )
            
            # Fazer predição
            # Criar uma lista de valores na mesma ordem das variáveis independentes
            input_list = [input_values[var] for var in independent_vars]
            
            # Criar o array de predição com formato explícito
            prediction_array = np.array(input_list).reshape(1, -1)
            
            # Criar DataFrame com nomes de colunas adequados e formato
            prediction_input = pd.DataFrame(prediction_array, columns=independent_vars)
            
            # Adicionar constante e fazer predição
            prediction_input_with_const = sm.add_constant(prediction_input, has_constant='add')
            prediction = model.predict(prediction_input_with_const).iloc[0]
            
            st.write(f"**{dependent_var} Predito:** {prediction:.4f}")
            
            # Testes dos Pressupostos
            st.subheader("Testes dos Pressupostos do Modelo")
            
            # Teste de normalidade
            _, normality_p = stats.normaltest(model.resid)
            
            # Teste de heterocedasticidade
            _, het_p, _, _ = het_breuschpagan(model.resid, X_with_const)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Teste de Normalidade (D'Agostino-Pearson):**")
                st.write(f"p-valor: {normality_p:.4f}")
                st.write("Interpretação: Resíduos são normais se p > 0,05")
            
            with col2:
                st.write("**Teste de Heterocedasticidade (Breusch-Pagan):**")
                st.write(f"p-valor: {het_p:.4f}")
                st.write("Interpretação: Variância é constante se p > 0,05")

    except Exception as e:
        st.error(f"Erro: {str(e)}")
        st.write("Por favor, certifique-se de que o arquivo 'series_temporais.csv' está disponível e contém as colunas necessárias.")
        
        
elif geral == "Regressão Linear Simples":        

    # Título e introdução
    st.title("Ferramenta de Análise de Regressão Linear")

    # Explicação da Regressão Linear e Pressupostos
    st.header("Visão Geral da Regressão Linear")
    st.write("""
    A regressão linear simples é um método estatístico que nos permite estudar a relação entre duas variáveis contínuas: 
    uma variável dependente (Y) e uma variável independente (X). A relação é modelada usando uma equação linear:
    Y = β₀ + β₁X + ε

    Onde:
    - β₀ é o intercepto (valor de Y quando X = 0)
    - β₁ é a inclinação (mudança em Y para uma mudança unitária em X)
    - ε é o termo de erro
    """)

    st.subheader("Pressupostos Principais:")
    st.write("""
    1. **Linearidade**: A relação entre X e Y deve ser linear
    2. **Independência**: Os resíduos devem ser independentes
    3. **Normalidade**: Os resíduos devem ser normalmente distribuídos
    4. **Homocedasticidade**: Os resíduos devem ter variância constante
    """)

    # Carregar e preparar dados
    @st.cache_data
    def load_data():
        df = pd.read_csv('series_temporais.csv')
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df

    try:
        df = load_data()
        
        # Seleção de variáveis
        variables = ['IR', 'GDP', 'CRED', 'ROE', 'LEV', 'NPL', 'INFLA', 'TARG', 'EMBI']
        
        col1, col2 = st.columns(2)
        
        with col1:
            dependent_var = st.selectbox(
                'Selecione a Variável Dependente (Y)',
                variables
            )
        
        with col2:
            independent_var = st.selectbox(
                'Selecione a Variável Independente (X)',
                [var for var in variables if var != dependent_var]
            )
        
        # Preparar dados para regressão
        X = df[independent_var].values.reshape(-1, 1)
        y = df[dependent_var].values
        
        # Adicionar constante ao X para statsmodels
        X_with_const = sm.add_constant(X)
        
        # Ajustar modelo de regressão
        model = sm.OLS(y, X_with_const).fit()
        
        # Plotar linha de regressão
        st.header("Análise de Regressão")
        
        fig = px.scatter(df, x=independent_var, y=dependent_var, 
                        title=f'Regressão Linear: {dependent_var} vs {independent_var}')
        
        # Adicionar linha de regressão
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_range_with_const = sm.add_constant(x_range)
        y_pred_line = model.predict(X_range_with_const)
        
        # Adicionar linha de regressão com equação
        equation_text = f'y = {model.params[0]:.2f} + {model.params[1]:.2f}x'
        fig.add_trace(
            go.Scatter(x=x_range.flatten(), y=y_pred_line,
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
                text=f'Regressão Linear: {dependent_var} vs {independent_var}<br><sub>{equation_text}</sub>',
                x=0.5,
                xanchor='center'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas do Modelo
        st.subheader("Estatísticas de Regressão")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Resumo do Modelo:**")
            st.write(f"R-quadrado: {model.rsquared:.4f}")
            st.write(f"R-quadrado Ajustado: {model.rsquared_adj:.4f}")
            st.write(f"Erro Quadrático Médio: {np.mean(model.resid**2):.4f}")
            st.write(f"Raiz do Erro Quadrático Médio: {np.sqrt(np.mean(model.resid**2)):.4f}")
        
        with col2:
            st.write("**Estatísticas dos Coeficientes:**")
            st.write(f"Intercepto (β₀): {model.params[0]:.4f}")
            st.write(f"Inclinação (β₁): {model.params[1]:.4f}")
            st.write(f"P-valor (inclinação): {model.pvalues[1]:.4f}")
        
        # Predição Pontual
        st.subheader("Predição Pontual")
        
        # Permitir ao usuário selecionar um ponto específico
        point_index = st.slider("Selecione um ponto de dados", 0, len(df)-1, len(df)//2)
        
        actual_x = X[point_index][0]
        actual_y = y[point_index]
        predicted_y = model.predict(X_with_const)[point_index]
        
        st.write(f"**Ponto Selecionado:**")
        st.write(f"X ({independent_var}): {actual_x:.4f}")
        st.write(f"Y Real ({dependent_var}): {actual_y:.4f}")
        st.write(f"Y Predito: {predicted_y:.4f}")
        st.write(f"Erro de Predição: {actual_y - predicted_y:.4f}")
        
        # Testes dos Pressupostos
        st.subheader("Testes dos Pressupostos do Modelo")
        
        # Teste de normalidade
        _, normality_p = stats.normaltest(model.resid)
        
        # Teste de heterocedasticidade
        _, het_p, _, _ = het_breuschpagan(model.resid, X_with_const)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Teste de Normalidade (D'Agostino-Pearson):**")
            st.write(f"p-valor: {normality_p:.4f}")
            st.write("Interpretação: Resíduos são normais se p > 0,05")
        
        with col2:
            st.write("**Teste de Heterocedasticidade (Breusch-Pagan):**")
            st.write(f"p-valor: {het_p:.4f}")
            st.write("Interpretação: Variância é constante se p > 0,05")

    except Exception as e:
        st.error(f"Erro: {str(e)}")
        st.write("Por favor, certifique-se de que o arquivo 'series_temporais.csv' está disponível e contém as colunas necessárias.")        

# Rodapé
st.divider()
st.caption("© 2025 Ferramenta de Ensino de Regressão Linear | Desenvolvido para fins educacionais")
st.caption("Prof. José Américo – Coppead")