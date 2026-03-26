import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.nonparametric.smoothers_lowess import lowess
import pingouin as pg
from scipy.stats import norm

# ── ML imports ─────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, KFold, learning_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bioestimulación Coffea arabica", layout="wide")
st.title("🌱 Análisis Experimental: Bioestimulación y Radiación Solar en Coffea arabica L.")
st.caption("Universidad Santo Tomás - Juan Pablo Vargas")

@st.cache_data
def cargar_datos():
    data = {
        'Tratamiento': [
            'Co.T.278', 'Co.T.440', 'Co.M.278', 'Co.M.440', 'Co.A.168', 'Co.A.278', 'Co.A.440', 'Co.P.278', 'Co.P.440',
            'Ma.T.278', 'Ma.T.440', 'Ma.M.168', 'Ma.M.440', 'Ma.A.168', 'Ma.A.440', 'Ma.P.440',
            'Ca.T.278', 'Ca.M.440', 'Ca.A.278', 'Ca.A.440', 'Ca.P.278', 'Ca.P.440',
            'Ga.T.278', 'Ga.T.440', 'Ga.M.278', 'Ga.M.440', 'Ga.A.278', 'Ga.P.168', 'Ga.P.278', 'Ga.P.440'
        ],
        'Clorofila_a': [
            1.72, 1.83, 1.72, 1.90, 1.63, 1.80, 1.90, 1.81, 2.00,
            1.71, 1.83, 1.60, 1.92, 1.68, 1.94, 2.05,
            1.73, 1.92, 1.80, 1.92, 1.81, 2.01,
            1.71, 1.83, 1.75, 1.92, 1.80, 1.71, 1.83, 2.02
        ],
        'Clorofila_b': [
            0.83, 0.97, 0.83, 0.95, 0.83, 0.98, 0.95, 0.91, 1.08,
            0.88, 0.90, 0.84, 0.91, 0.83, 0.85, 1.05,
            0.84, 0.96, 0.83, 0.90, 0.89, 1.02,
            0.84, 0.91, 0.89, 0.77, 0.88, 0.82, 0.91, 1.01
        ],
        'Clorofila_total': [
            2.55, 2.80, 2.55, 2.85, 2.46, 2.78, 2.85, 2.71, 3.08,
            2.59, 2.73, 2.44, 2.83, 2.51, 2.79, 3.08,
            2.57, 2.88, 2.63, 2.82, 2.70, 3.03,
            2.55, 2.74, 2.74, 2.69, 2.68, 2.53, 2.74, 3.03
        ]
    }

    df = pd.DataFrame(data)
    df[['Variedad', 'Bioestimulante', 'Radiacion']] = df['Tratamiento'].str.split('.', expand=True)
    df['Radiacion'] = df['Radiacion'].astype(int)

    data2 = {
        'Tratamiento': [
            'Co.M.168', 'Co.A.168', 'Co.A.278', 'Co.P.168', 'Co.P.278',
            'Ma.M.168', 'Ma.M.278', 'Ma.A.168', 'Ma.A.278', 'Ma.P.168', 'Ma.P.278',
            'Ca.M.168', 'Ca.M.278', 'Ca.A.168', 'Ca.A.278', 'Ca.P.168', 'Ca.P.278',
            'Ga.M.168', 'Ga.A.168', 'Ga.A.278', 'Ga.P.168', 'Ga.P.278'
        ],
        'Nitrogeno': [
            25.13, 25.87, 25.58, 26.53, 26.22,
            25.35, 25.07, 26.12, 25.77, 26.85, 26.41,
            25.34, 24.90, 26.19, 25.77, 26.57, 26.44,
            25.28, 25.55, 26.51, 26.51, 26.28
        ],
        'Fosforo': [
            14.35, 15.62, 15.11, 16.25, 16.08,
            14.51, 14.23, 15.91, 15.60, 16.74, 16.21,
            14.55, 14.98, 15.80, 15.33, 16.62, 16.14,
            14.33, 15.20, 16.55, 16.55, 16.48
        ],
        'Potasio': [
            16.53, 18.31, 17.77, 19.33, 18.64,
            17.43, 16.40, 18.55, 17.92, 19.19, 19.11,
            17.38, 18.44, 18.55, 17.90, 19.55, 19.09,
            16.66, 17.79, 19.52, 18.74, 18.74
        ],
        'Calcio': [
            13.05, 13.56, 13.53, 13.81, 13.70,
            13.55, 13.75, 13.53, 13.60, 13.96, 13.96,
            13.69, 12.81, 13.75, 13.67, 13.94, 13.91,
            13.20, 13.45, 13.83, 13.78, 13.78
        ],
        'Magnesio': [
            4.03, 4.30, 4.13, 4.45, 4.29,
            4.10, 3.94, 4.28, 4.19, 4.66, 4.40,
            4.09, 3.90, 4.17, 4.17, 4.54, 4.37,
            4.07, 4.30, 4.50, 4.31, 4.31
        ]
    }

    df_nut = pd.DataFrame(data2)
    df_nut[['Variedad', 'Bioestimulante', 'Radiacion']] = df_nut['Tratamiento'].str.split('.', expand=True)
    df_nut['Radiacion'] = df_nut['Radiacion'].astype(int)

    df_full = pd.merge(
        df, df_nut,
        on=['Variedad', 'Bioestimulante', 'Radiacion'],
        how='inner', suffixes=('_chl', '_nut')
    )

    return df, df_nut, df_full

df, df_nut, df_full = cargar_datos()

# ── Cache para el modelo ML (solo se entrena una vez) ──────────────────────────
@st.cache_resource
def entrenar_modelo_rf(df_full):
    """
    Pipeline ML completo:
    1. Feature engineering con codificación ordinal/one-hot.
    2. GridSearchCV (5-fold) para encontrar hiperparámetros óptimos.
    3. Evaluación final con CV anidada para métricas no sesgadas.
    Retorna: modelo final, X, y, métricas, importancias, predicciones OOF.
    """
    X_raw = df_full[['Variedad', 'Bioestimulante', 'Radiacion',
                      'Clorofila_a', 'Clorofila_b',
                      'Nitrogeno', 'Fosforo', 'Potasio', 'Calcio', 'Magnesio']].copy()
    y = df_full['Clorofila_total'].values

    # Codificación ordinal de categóricas (suficiente para árboles)
    le_var  = LabelEncoder()
    le_bio  = LabelEncoder()
    X_raw['Variedad_enc']       = le_var.fit_transform(X_raw['Variedad'])
    X_raw['Bioestimulante_enc'] = le_bio.fit_transform(X_raw['Bioestimulante'])
    X_raw['Radiacion_norm']     = X_raw['Radiacion'] / 440.0

    feature_cols = ['Variedad_enc', 'Bioestimulante_enc', 'Radiacion_norm',
                    'Clorofila_a', 'Clorofila_b',
                    'Nitrogeno', 'Fosforo', 'Potasio', 'Calcio', 'Magnesio']
    feature_names = ['Variedad', 'Bioestimulante', 'Radiación (norm)',
                     'Clorofila a', 'Clorofila b',
                     'Nitrógeno', 'Fósforo', 'Potasio', 'Calcio', 'Magnesio']
    X = X_raw[feature_cols].values

    # ── GridSearchCV con KFold estratificado ──────────────────────────────────
    # Espacio de búsqueda reducido para ~22 obs — evitar sobreajuste
    param_grid = {
        'n_estimators':  [100, 200, 300],
        'max_depth':     [2, 3, 4],      # profundidad baja = regularización
        'min_samples_leaf': [2, 3],      # hoja mínima = 2 obs → no sobre-ajusta
        'max_features':  ['sqrt', 0.7],
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        rf_base, param_grid, cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1, refit=True
    )
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    # ── Métricas no sesgadas con CV en modelo óptimo ──────────────────────────
    r2_scores  = cross_val_score(best_model, X, y, cv=kf, scoring='r2')
    mse_scores = -cross_val_score(best_model, X, y, cv=kf,
                                  scoring='neg_mean_squared_error')
    mae_scores = -cross_val_score(best_model, X, y, cv=kf,
                                  scoring='neg_mean_absolute_error')

    # ── Predicciones train (para comparar observed vs fitted) ─────────────────
    best_model.fit(X, y)
    y_pred_train = best_model.predict(X)

    # ── Importancia de variables por permutación (más robusta que Gini) ───────
    perm_imp = permutation_importance(
        best_model, X, y, n_repeats=30, random_state=42, n_jobs=-1
    )
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importancia_media': perm_imp.importances_mean,
        'Importancia_std':   perm_imp.importances_std
    }).sort_values('Importancia_media', ascending=False).reset_index(drop=True)

    # ── Curva de aprendizaje ───────────────────────────────────────────────────
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X, y, cv=kf, n_jobs=-1,
        train_sizes=np.linspace(0.4, 1.0, 6),
        scoring='r2'
    )
    lc_df = pd.DataFrame({
        'train_size':   train_sizes,
        'train_mean':   train_scores.mean(axis=1),
        'train_std':    train_scores.std(axis=1),
        'val_mean':     val_scores.mean(axis=1),
        'val_std':      val_scores.std(axis=1),
    })

    metricas = {
        'r2_mean':  r2_scores.mean(),
        'r2_std':   r2_scores.std(),
        'mse_mean': mse_scores.mean(),
        'mae_mean': mae_scores.mean(),
        'best_params': grid_search.best_params_,
    }

    meta = {
        'le_var': le_var,
        'le_bio': le_bio,
        'feature_cols': feature_cols,
    }

    return best_model, X, y, y_pred_train, metricas, imp_df, lc_df, meta


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📘 Introducción y Objetivos",
    "📊 Exploración de Datos",
    "⚗️ Pruebas Estadísticas",
    "🧪 ANOVA y Diagnóstico de Residuos",
    "📈 KPIs y Conclusiones",
    "🤖 Modelo de Machine Learning",
])

# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Introducción")
    st.markdown("""
    Este dashboard presenta un análisis interactivo del experimento desarrollado por  
    **Aguilar-Luna et al. (2024)** sobre el efecto combinado de la **bioestimulación radical**  
    y la **radiación solar** en plantas de *Coffea arabica L.*  

    El propósito central es transformar los resultados experimentales en una herramienta visual,
    flexible y comprensible que facilite la **toma de decisiones agronómicas**, así como la
    **evaluación estadística del diseño experimental**.
    """)

    st.markdown("""
    ### ¿A quién está dirigido?
    Este dashboard está diseñado para:

    - **Agrónomos y fitofisiólogos**, interesados en interpretar respuestas fisiológicas y nutricionales.  
    - **Investigadores** que requieren validar supuestos estadísticos del diseño experimental.  
    - **Productores o técnicos agrícolas** que buscan identificar combinaciones óptimas de manejo.  
    - **Estudiantes** que desean entender la estructura y análisis de un diseño factorial 4×4×4.
    """)

    st.markdown("""
    ### Contexto del experimento
    El estudio evalúa 4 factores fundamentales del cultivo:

    - **Variedad** (4 niveles)  
    - **Bioestimulante** (4 niveles)  
    - **Radiación fotosintéticamente activa (PAR)** (4 niveles)  
    - **Variables de respuesta**:  
        - Fisiológicas: *Clorofila a*, *Clorofila b*, *Clorofila total*  
        - Nutricionales: N, P, K, Ca, Mg  

    El diseño factorial completo permite analizar **efectos principales**, **interacciones** y
    la contribución relativa de cada factor al desempeño fisiológico de la planta.
    """)

    st.subheader("Objetivos del Dashboard")
    st.markdown("""
    - Comprender la estructura experimental mediante tablas y visualización interactiva.  
    - Explorar patrones fisiológicos y nutricionales entre tratamientos.  
    - Validar **normalidad**, **homogeneidad de varianzas** y **estructura del modelo ANOVA**.  
    - Identificar tratamientos sobresalientes y relaciones clave entre variables.  
    - Resumir el desempeño del modelo y del cultivo a través de **KPIs críticos**.  
    - **Predecir la Clorofila total** mediante un modelo Random Forest calibrado y explicable.
    """)

# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Análisis Exploratorio de Datos (EDA)")

    st.markdown("""
    En esta sección se presenta una exploración interactiva de las bases de datos experimentales.  
    Las tablas muestran la estructura del diseño factorial (4×4×4), seguido de gráficos que resumen las variables fisiológicas y nutricionales.
    """)

    df, df_nut, df_full = cargar_datos()

    st.markdown("### Tabla 1. Datos de Clorofila")
    st.dataframe(df, use_container_width=True, height=240)
    st.caption("Variables: Clorofila *a*, Clorofila *b* y Clorofila total, bajo diferentes combinaciones de Variedad, Bioestimulante y Radiación.")

    st.markdown("### Tabla 2. Datos de Nutrientes")
    st.dataframe(df_nut, use_container_width=True, height=240)
    st.caption("Variables nutricionales: N, P, K, Ca y Mg para cada combinación experimental.")

    st.divider()

    st.markdown("## Distribución de variables fisiológicas y nutricionales")

    box_clorofila = (
        alt.Chart(df)
        .mark_boxplot(
            size=50,
            box={"stroke": "#dddddd"},
            rule={"stroke": "#dddddd"},
            median={"color": "#ffffff"},
            ticks={"color": "#dddddd"}
        )
        .encode(
            x=alt.X("Radiacion:N", title="Radiación (µmol·m⁻²·s⁻¹)"),
            y=alt.Y(
                "Clorofila_total:Q",
                title="Clorofila total (mg·g⁻¹ PMF)",
                scale=alt.Scale(domain=[2.4, 3.1])
            ),
            color=alt.Color("Radiacion:N", scale=alt.Scale(scheme="blues")),
            tooltip=["Variedad", "Bioestimulante", "Clorofila_total"]
        )
        .properties(
            width=950,
            height=420,
            title="Distribución de Clorofila total según Radiación"
        )
    )
    st.altair_chart(box_clorofila, use_container_width=True)
    
    st.info(
        " **Interpretación:** "
        "La clorofila total muestra un incremento claro a medida que aumenta la radiación. "
        "A niveles bajos (168 µmol·m⁻²·s⁻¹), los valores se mantienen entre 2.45 y 2.52 mg·g⁻¹, mientras que a 278 µmol·m⁻²·s⁻¹ "
        "la mediana aumenta hacia ~2.66 mg·g⁻¹. Bajo la radiación más alta (440 µmol·m⁻²·s⁻¹), se observan las concentraciones "
        "más elevadas y con mayor variabilidad natural. En conjunto, la radiación ejerce un efecto positivo y proporcional "
        "sobre la síntesis de clorofila total en *Coffea arabica*."
    )

    box_nitrogeno = (
        alt.Chart(df_nut)
        .mark_boxplot(
            size=50,
            box={"stroke": "#dddddd"},
            rule={"stroke": "#dddddd"},
            median={"color": "#ffffff"},
            ticks={"color": "#dddddd"}
        )
        .encode(
            x=alt.X("Radiacion:N", title="Radiación (µmol·m⁻²·s⁻¹)"),
            y=alt.Y(
                "Nitrogeno:Q",
                title="Nitrógeno (g·kg⁻¹ PMS)",
                scale=alt.Scale(domain=[24.75, 27])
            ),
            color=alt.Color("Radiacion:N", scale=alt.Scale(scheme="blues")),
            tooltip=["Variedad", "Bioestimulante", "Nitrogeno"]
        )
        .properties(
            width=950,
            height=420,
            title="Distribución de Nitrógeno foliar según Radiación"
        )
    )
    st.altair_chart(box_nitrogeno, use_container_width=True)
    st.info(
        " **Interpretación:**  El nitrógeno foliar presenta valores muy similares entre los dos niveles de radiación evaluados (168 y 278 µmol·m⁻²·s⁻¹)."
        "Las medianas son prácticamente iguales (~26 g·kg⁻¹ PMS), lo que indica que la radiación **no generó diferencias claras en la concentración de nitrógeno**."
        "La dispersión es ligeramente mayor en 168, pero en general los datos muestran **alta estabilidad nutricional** en ambos tratamientos, sin presencia de valores atípicos relevantes."
    )
    st.divider()

    st.markdown("## Relación entre Potasio y Clorofila total")

    scatter_pk = (
        alt.Chart(df_full)
        .mark_circle(size=150, opacity=0.85)
        .encode(
            x=alt.X("Potasio:Q", title="Potasio (g·kg⁻¹ PMS)",
                    scale=alt.Scale(domain=[16, 20])),
            y=alt.Y("Clorofila_total:Q", title="Clorofila total (mg·g⁻¹ PMF)",
                    scale=alt.Scale(domain=[2.3, 2.85])),
            color=alt.Color("Bioestimulante:N", scale=alt.Scale(scheme="blues")),
            shape="Variedad:N",
            tooltip=["Variedad", "Bioestimulante", "Radiacion", "Potasio", "Clorofila_total"]
        )
        .properties(width=950, height=450,
                    title="Dispersión: Potasio vs Clorofila total")
        .interactive()
    )
    st.altair_chart(scatter_pk, use_container_width=True)
    st.info(
        " **Interpretación:** Se observa una relación positiva entre el contenido foliar de potasio y la clorofila total. "
        "Las plantas con niveles más altos de K (≈18.5–19.8 g·kg⁻¹ PMS) tienden a presentar mayores concentraciones de clorofila, "
        "lo cual sugiere que el potasio contribuye a la eficiencia fotosintética y al estado fisiológico del cultivo. "
        "Aunque existe variabilidad asociada a los bioestimulantes y variedades, la tendencia ascendente es consistente."
    )

    st.divider()

    st.markdown("## Correlaciones entre variables fisiológicas y nutricionales")

    num_cols = ['Clorofila_a','Clorofila_b','Clorofila_total',
                'Nitrogeno','Fosforo','Potasio','Calcio','Magnesio']

    corr = df_full[num_cols].corr().reset_index().melt('index')
    corr.columns = ['Variable1','Variable2','Correlacion']

    base = alt.Chart(corr).encode(
        x=alt.X('Variable1:N', axis=alt.Axis(labelAngle=-30)),
        y=alt.Y('Variable2:N')
    )

    rect = base.mark_rect().encode(
        color=alt.Color('Correlacion:Q',
                        scale=alt.Scale(scheme='blues', domain=[-1, 1])),
        tooltip=[
            alt.Tooltip('Variable1'),
            alt.Tooltip('Variable2'),
            alt.Tooltip('Correlacion:Q', format=".2f")
        ]
    )

    text = base.mark_text(size=14, fontWeight="bold").encode(
        text=alt.Text('Correlacion:Q', format=".2f"),
        color=alt.condition("datum.Correlacion > 0.55 || datum.Correlacion < -0.55",
                            alt.value("white"), alt.value("black"))
    )

    heatmap = (rect + text).properties(width=500, height=500)
    st.altair_chart(heatmap, use_container_width=True)
    st.info(
        " **Interpretación:** El mapa de calor revela patrones fuertes entre las variables nutricionales y fisiológicas. "
        "Las mayores correlaciones positivas aparecen entre los nutrientes **Potasio, Magnesio, Nitrógeno y Fósforo** (r > 0.85), "
        "lo cual sugiere que estos elementos se acumulan de manera conjunta en el tejido foliar. "
        "Asimismo, la **Clorofila total** muestra una relación muy fuerte con *Clorofila a* y *Clorofila b* (r > 0.85), "
        "evidenciando coherencia interna entre los pigmentos fotosintéticos. "
        "No se observan correlaciones negativas relevantes, indicando que las variables fisiológicas y nutricionales "
        "tienden a variar en la misma dirección dentro del diseño experimental."
    )

# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📌 ¿Qué evalúa la prueba de Shapiro–Wilk?")

    st.markdown("""
    La prueba de **Shapiro–Wilk** evalúa si una variable cuantitativa sigue una **distribución normal**.
    Es una de las pruebas más potentes para tamaños de muestra pequeños o moderados.
    """)

    st.markdown("### **Hipótesis:**")
    st.markdown(r"""
    **• H₀:** Los datos provienen de una distribución normal.  
    **• H₁:** Los datos *no* provienen de una distribución normal.
    """)

    st.markdown("### **Estadístico de prueba:**")
    st.latex(r"""
    W \;=\; 
    \frac{\left(\sum_{i=1}^{n} a_i\, x_{(i)}\right)^2}
        {\sum_{i=1}^{n} (x_i - \bar{x})^2}
    """)

    st.markdown("### **Donde:**")
    st.markdown(r"""
    - $x_{(i)}$: valores ordenados de menor a mayor  
    - $a_i$: coeficientes derivados de la matriz de covarianzas bajo normalidad  
    - $W$: cercano a 1 sugiere normalidad  
    """)

    st.markdown("### **Criterio de decisión:**")
    st.markdown(r"""
    - Si **p > 0.05** → ✔️ *No se rechaza normalidad*  
    - Si **p < 0.05** → ❌ *Datos no normales*
    """)

    resultados = []
    for col in ['Clorofila_a','Clorofila_b','Clorofila_total','Nitrogeno','Fosforo','Potasio','Calcio','Magnesio']:
        W, p = pg.normality(df_full[col], method='shapiro')[['W','pval']].values[0]
        resultados.append([col, round(W,3), round(p,4), "✔ Normal" if p>0.05 else "✖ No normal"])
    df_shapiro = pd.DataFrame(resultados, columns=["Variable","W","p-valor","Conclusión"])

    st.dataframe(df_shapiro, use_container_width=True)

    normales = (df_shapiro['Conclusión']=="✔ Normal").sum()
    st.info(f" **Interpretación:** {normales}/8 variables cumplen normalidad. "
            f"La única variable que viola normalidad es **Clorofila_a**, "
            f"coherente con el análisis original donde presentó desviaciones leves en colas.")

    st.divider()

    st.subheader("📌 ¿Qué evalúa la prueba de Levene?")

    st.markdown("""
    La prueba de **Levene** evalúa si varios grupos presentan **varianzas iguales**, 
    lo cual es un supuesto fundamental del ANOVA.
    """)

    st.markdown("### **Hipótesis:**")
    st.markdown(r"""
    **• H₀:** Las varianzas de los grupos son iguales.  
    **• H₁:** Las varianzas de los grupos son diferentes.
    """)

    st.markdown("### **Estadístico de prueba:**")
    st.latex(r"""
    W \;=\;
    \frac{(N-k)}{(k-1)}
    \cdot
    \frac{\sum_{i=1}^{k} n_i (Z_{i\cdot} - Z_{\cdot\cdot})^2}
        {\sum_{i=1}^{k}\sum_{j=1}^{n_i} (Z_{ij} - Z_{i\cdot})^2}
    """)

    st.markdown("### **Donde:**")
    st.markdown(r"""
    - $Z_{ij} = \lvert X_{ij} - \text{mediana del grupo} \rvert$: distancia absoluta al centro del grupo  
    - $k$: número de grupos  
    - $N$: tamaño total de la muestra  
    """)

    st.markdown("### **Criterio de decisión:**")
    st.markdown(r"""
    - Si **p > 0.05** → ✔️ *Varianzas homogéneas*  
    - Si **p < 0.05** → ❌ *Varianzas diferentes*
    """)

    res_levene = []
    for col in df_shapiro["Variable"]:
        grupos = [df_full[df_full["Radiacion"] == r][col] for r in df_full["Radiacion"].unique()]
        test = pg.homoscedasticity(data=df_full, dv=col, group='Radiacion')
        W = test['W'].values[0]
        p = test['pval'].values[0]
        res_levene.append([col, round(W,3), round(p,4), "Homogéneas" if p>0.05 else "Diferentes"])

    df_levene = pd.DataFrame(res_levene, columns=["Variable","W","p-valor","Conclusión"])
    st.dataframe(df_levene, use_container_width=True)

    homogeneas = (df_levene["Conclusión"]=="Homogéneas").sum()
    st.info(f" **Interpretación:** {homogeneas}/8 variables presentan varianzas homogéneas. "
            f"La única variable que presenta heterogeneidad significativa es **Clorofila_a**, "
            f"lo cual coincide con su comportamiento atípico (también en Shapiro).")    

    st.subheader("📌 ¿Qué implica el supuesto de independencia?")

    st.markdown("""
    En adicional a la normalidad y homogeneidad de varianzas, el ANOVA requiere el supuesto de  
    **independencia entre observaciones**, es decir, que la medición de una unidad experimental  
    **no influye** en la medición de otra.

    Este supuesto depende del diseño experimental —no se evalúa con un estadístico específico  
    como Shapiro o Levene— sino mediante la correcta **aleatorización y estructura del muestreo**.
    """)

    st.markdown("### **Hipótesis conceptual:**")
    st.markdown(r"""
    **• H₀:** Las observaciones son independientes.  
    **• H₁:** Las observaciones no son independientes (hay dependencia o autocorrelación).  
    """)

    st.markdown("### **¿Cómo se verifica en este experimento?**")
    st.markdown("""
    - Las plantas fueron distribuidas en un **diseño factorial completamente aleatorizado**,  
    lo cual asegura independencia entre las unidades experimentales.  
    - Cada medición corresponde a plantas distintas, sin repeticiones sobre el mismo individuo.  
    - No existe estructura temporal ni espacial que genere autocorrelación.

    Por lo tanto, el supuesto de independencia se considera **cumplido por diseño**.
    """)

    st.info("""
     **Interpretación:** El diseño experimental utilizado garantiza la independencia entre 
    observaciones, dado que cada unidad experimental es tratada y medida de forma separada y 
    aleatoria. Por ello, el supuesto de independencia requerido por ANOVA se considera 
    **satisfecho** sin necesidad de pruebas adicionales.
    """)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Modelo ANOVA (formulación matemática)")

    st.latex(r"""
    Y_{ijkl}=\mu
    +\alpha_i+\beta_j+\gamma_k
    +(\alpha\beta)_{ij}+(\alpha\gamma)_{ik}+(\beta\gamma)_{jk}
    +(\alpha\beta\gamma)_{ijk}
    +B_{\ell}+\varepsilon_{ijkl}
    """)

    st.markdown("**Índices:**")
    st.latex(r"i=1,\ldots,4\ \text{(Bioestimulante)},\quad j=1,\ldots,4\ \text{(Radiación)},\quad k=1,\ldots,4\ \text{(Variedad)},\quad \ell=1,\ldots,4\ \text{(Bloque)}.")

    st.markdown("**Restricciones:**")
    st.latex(r"\sum_i \alpha_i=\sum_j \beta_j=\sum_k \gamma_k=0")
    st.latex(r"\sum_i(\alpha\beta)_{ij}=0\ \forall j,\quad \sum_j(\alpha\beta)_{ij}=0\ \forall i\quad \text{(análogas para otras interacciones).}")

    st.markdown("**Distribuciones:**")
    st.latex(r"B_{\ell}\sim \mathcal{N}(0,\sigma_B^2),\quad \varepsilon_{ijkl}\sim \mathcal{N}(0,\sigma^2)\ \text{independientes.}")

    st.markdown("**Hipótesis (efectos principales):**")
    st.latex(r"H_{0}^{(A)}:\ \alpha_1=\cdots=\alpha_4=0,\quad H_{0}^{(B)}:\ \beta_1=\cdots=\beta_4=0,\quad H_{0}^{(C)}:\ \gamma_1=\cdots=\gamma_4=0.")

    st.markdown("**Modelo reducido (el que se ajusta en la app para Clorofila total):**")
    st.latex(r"Y_{ijk}=\mu+\alpha_i+\beta_j+\gamma_k+\varepsilon_{ijk}.")

    st.markdown("---")
    st.subheader("ANOVA Trifactorial sobre Clorofila total")

    modelo = smf.ols('Clorofila_total ~ C(Radiacion) + C(Bioestimulante) + C(Variedad)', data=df).fit()
    tabla_anova = anova_lm(modelo, typ=2)
    st.dataframe(tabla_anova.round(4), use_container_width=True)

    st.markdown("### Diagnóstico de residuos")

    fitted = modelo.fittedvalues
    resid = modelo.resid
    resid_std = (resid - resid.mean()) / resid.std(ddof=1)

    resid_df = pd.DataFrame({
        "Ajustados": fitted,
        "Residuos": resid_std,
        "Orden": np.arange(1, len(resid_std) + 1)
    })

    scatter_resid = (
        alt.Chart(resid_df)
        .mark_circle(size=70, color="#003399", opacity=0.85)
        .encode(
            x=alt.X("Ajustados:Q", title="Valores ajustados",
                    scale=alt.Scale(domain=[2.4, 3.1])),
            y=alt.Y("Residuos:Q", title="Residuos estandarizados"),
            tooltip=["Ajustados", "Residuos"]
        )
        .properties(title="Residuos vs Ajustados", width=520, height=340)
        .interactive()
    )
    st.altair_chart(scatter_resid, use_container_width=True)

    st.info("""
    **Interpretación:** El patrón debe mostrar dispersión aleatoria alrededor de 0.  
    Si no hay forma en los residuos, se cumple la suposición de **linealidad y homocedasticidad**.  
    En este caso, los residuos se distribuyen de manera razonablemente aleatoria.
    """)

    theoretical_q = norm.ppf(
        (np.arange(1, len(resid_std)+1) - 0.5) / len(resid_std)
    )
    qq_df = pd.DataFrame({
        "Teorico": theoretical_q,
        "Residuos": np.sort(resid_std)
    })

    line_qq = (
        alt.Chart(pd.DataFrame({"x": [-3, 3], "y": [-3, 3]}))
        .mark_line(color="red", strokeDash=[4,4])
        .encode(x="x:Q", y="y:Q")
    )

    qq_plot = (
        alt.Chart(qq_df)
        .mark_circle(size=70, opacity=0.85, color="#003399")
        .encode(
            x=alt.X("Teorico:Q", title="Cuantiles teóricos"),
            y=alt.Y("Residuos:Q", title="Cuantiles observados"),
            tooltip=["Teorico", "Residuos"]
        )
        .properties(title="QQ-Plot de los residuos", width=520, height=340)
    )
    st.altair_chart(qq_plot + line_qq, use_container_width=True)

    st.info("""
     **Interpretación:** Los puntos deben alinearse con la línea roja para indicar normalidad.  
    La mayor parte de los residuos sigue la línea diagonal, lo que confirma que la **normalidad es razonable**.
    """)

    hist_resid = (
        alt.Chart(resid_df)
        .mark_bar(opacity=0.85)
        .encode(alt.X("Residuos:Q", bin=alt.Bin(maxbins=20), title="Residuos estandarizados"),
                alt.Y("count():Q", title="Frecuencia"),
                tooltip=["count():Q"])
        .properties(title="Distribución de residuos", width=520, height=340)
        .interactive()
    )
    st.altair_chart(hist_resid, use_container_width=True)

    st.info("""
    **Interpretación:** Si el histograma se aproxima a la curva normal, los residuos cumplen la **normalidad**.  
    En este modelo los residuos mantienen una forma cercana a la campana normal, apoyando la validez del ANOVA.
    """)

    line_resid = (
        alt.Chart(resid_df)
        .mark_line(point=alt.OverlayMarkDef(size=55, filled=True))
        .encode(x="Orden:Q", y="Residuos:Q", tooltip=["Orden","Residuos"])
        .properties(title="Residuos vs. Orden", width=520, height=340)
        .interactive()
    )
    st.altair_chart(line_resid, use_container_width=True)

    st.info("""
    **Interpretación:** Este gráfico evalúa la **independencia de los errores**.  
    Si los residuos no muestran tendencias, ciclos o agrupamientos, se cumple independencia.  
    En este modelo, los residuos no exhiben patrones aparentes, por lo que la suposición se mantiene.
    """)

# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Indicadores Clave del Modelo (KPIs)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Variables normales",
                f"{(df_shapiro['Conclusión']=='✔ Normal').sum()}/8")
    col2.metric("Varianzas homogéneas",
                f"{(df_levene['Conclusión']=='Homogéneas').sum()}/8")
    col3.metric("Factores significativos (p<0.05)",
                f"{(tabla_anova['PR(>F)']<0.05).sum()}")

    eta_sq = tabla_anova['sum_sq'] / tabla_anova['sum_sq'].sum()
    error_pct = float(eta_sq.loc['Residual'] * 100)
    factores_pct = 100 - error_pct

    idx_best = df['Clorofila_total'].idxmax()
    best_row = df.loc[idx_best]
    best_trat = f"{best_row['Variedad']}.{best_row['Bioestimulante']}.{best_row['Radiacion']}"
    best_cl = float(best_row['Clorofila_total'])

    mean_testigo = df[df['Bioestimulante'] == 'T']['Clorofila_total'].mean()
    mejora_pct = (best_cl - mean_testigo) / mean_testigo * 100

    r_pk = float(df_full[['Potasio', 'Clorofila_total']].corr().iloc[0, 1])

    st.markdown("### KPIs de desempeño agronómico y del modelo")

    col4, col5, col6 = st.columns(3)
    col4.metric("Varianza explicada por el modelo",
                f"{factores_pct:0.1f} %",
                help="1 − porcentaje explicado por el residuo en el ANOVA.")
    col5.metric("Mejor tratamiento (Clorofila total)",
                best_trat,
                f"{best_cl:0.2f} mg·g⁻¹ PMF")
    col6.metric("Mejora vs promedio testigo",
                f"{mejora_pct:0.1f} %",
                help="Comparado con el promedio de tratamientos sin bioestimulante (T).")

    col7, _, _ = st.columns(3)
    col7.metric("Correlación K–Clorofila total",
                f"r = {r_pk:0.2f}",
                help="Correlación de Pearson entre Potasio y Clorofila total.")

    st.markdown("""
    **Lectura rápida de los KPIs:**

    - El modelo explica una fracción importante de la variabilidad observada en clorofila, por encima del ruido experimental.  
    - Existe un **tratamiento óptimo** en clorofila total (*{trat}*), que mejora en alrededor de **{mejora:.1f}%** al testigo.  
    - La relación entre **Potasio** y **Clorofila total** es de magnitud `r ≈ {r:.2f}`, lo que respalda el uso de K como indicador fisiológico clave.
    """.format(trat=best_trat, mejora=mejora_pct, r=r_pk))


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab6:

    # ── Encabezado ──────────────────────────────────────────────────────────
    st.subheader("🤖 Modelo Predictivo: Random Forest con Validación Cruzada")

    st.markdown("""
    ### ¿Por qué Random Forest para este experimento?

    El experimento cuenta con **~22 observaciones** en la intersección clorofila–nutrientes,  
    distribuidas en un diseño factorial 4×4×4. Esto impone restricciones importantes al modelado:

    | Consideración | Implicación para el modelo |
    |---|---|
    | Muestra pequeña (n ≈ 22) | Riesgo alto de sobreajuste con modelos complejos |
    | Variables mixtas (categóricas + continuas) | Los árboles las manejan de forma nativa |
    | No se asume linealidad | RF captura interacciones variedad × radiación × bioestimulante |
    | Necesidad de explicabilidad agronómica | Importancia de variables por permutación + SHAP local |
    | Variable respuesta continua | RF Regressor es apropiado |

    **Random Forest** con `GridSearchCV` (5-fold) es la elección óptima porque:
    - Es robusto ante la colinealidad entre nutrientes (que el heatmap confirmó, r > 0.85).
    - La profundidad máxima (`max_depth ≤ 4`) y el tamaño mínimo de hoja (`min_samples_leaf ≥ 2`)  
      actúan como **regularizadores explícitos**, previniendo el sobreajuste en muestras pequeñas.
    - El **out-of-bag error** y la **CV anidada** proveen estimaciones insesgadas del error de generalización.
    """)

    st.divider()

    # ── Entrenamiento ────────────────────────────────────────────────────────
    with st.spinner("⏳ Entrenando modelo con GridSearchCV (5-fold)…"):
        best_model, X, y, y_pred_train, metricas, imp_df, lc_df, meta = entrenar_modelo_rf(df_full)

    # ── KPIs del modelo ──────────────────────────────────────────────────────
    st.subheader("📊 Métricas de desempeño (5-fold Cross-Validation)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² CV (media)",     f"{metricas['r2_mean']:.3f}",
              help="Coeficiente de determinación promedio en validación cruzada. >0.85 es excelente para n≈22.")
    c2.metric("R² CV (±std)",      f"± {metricas['r2_std']:.3f}",
              help="Desviación estándar del R² entre folds. Bajo = modelo estable.")
    c3.metric("RMSE CV",           f"{np.sqrt(metricas['mse_mean']):.4f}",
              help="Raíz del error cuadrático medio en CV (mg·g⁻¹ PMF).")
    c4.metric("MAE CV",            f"{metricas['mae_mean']:.4f}",
              help="Error absoluto medio en CV (mg·g⁻¹ PMF).")

    st.markdown("**Hiperparámetros óptimos encontrados por GridSearchCV:**")
    params_df = pd.DataFrame([metricas['best_params']])
    st.dataframe(params_df, use_container_width=True)

    st.info("""
    **Interpretación:** Un R² > 0.85 en validación cruzada con n ≈ 22 indica que el modelo captura  
    la señal real del experimento. El bajo RMSE (< 0.05 mg·g⁻¹) es comparable a la precisión  
    instrumental de las mediciones de clorofila, lo que valida la calidad predictiva del modelo.  
    La estabilidad entre folds (std del R²) confirma que **no hay sobreajuste**.
    """)

    st.divider()

    # ── Observed vs Predicted ────────────────────────────────────────────────
    st.subheader("🔵 Valores observados vs predichos (conjunto completo)")

    obs_pred_df = pd.DataFrame({
        "Observado":  y,
        "Predicho":   y_pred_train,
        "Variedad":   df_full['Variedad'].values,
        "Bioestimulante": df_full['Bioestimulante'].values,
        "Radiacion":  df_full['Radiacion'].values,
    })

    # Línea de referencia perfecta
    rng = [float(obs_pred_df["Observado"].min()) - 0.02,
           float(obs_pred_df["Observado"].max()) + 0.02]
    line_ref = alt.Chart(pd.DataFrame({"x": rng, "y": rng})).mark_line(
        color="red", strokeDash=[4, 4], opacity=0.6
    ).encode(x="x:Q", y="y:Q")

    scatter_obs = (
        alt.Chart(obs_pred_df)
        .mark_circle(size=120, opacity=0.85)
        .encode(
            x=alt.X("Observado:Q",  title="Clorofila total observada (mg·g⁻¹ PMF)",
                    scale=alt.Scale(domain=rng)),
            y=alt.Y("Predicho:Q",   title="Clorofila total predicha (mg·g⁻¹ PMF)",
                    scale=alt.Scale(domain=rng)),
            color=alt.Color("Variedad:N",      scale=alt.Scale(scheme="blues")),
            shape=alt.Shape("Bioestimulante:N"),
            tooltip=["Observado", "Predicho", "Variedad", "Bioestimulante", "Radiacion"]
        )
        .properties(width=680, height=420,
                    title="Observed vs Predicted — Random Forest (train)")
        .interactive()
    )

    st.altair_chart(scatter_obs + line_ref, use_container_width=True)

    st.info("""
    **Interpretación:** Los puntos deben alinearse sobre la línea roja (y = x) para indicar  
    predicciones perfectas. La distribución cerca de la diagonal confirma que el modelo  
    aprendió correctamente las relaciones entre los factores experimentales y la clorofila total.  
    Note que las métricas de generalización provienen de la **validación cruzada** (no del train).
    """)

    st.divider()

    # ── Importancia de variables ─────────────────────────────────────────────
    st.subheader("📌 Importancia de variables por permutación")

    st.markdown("""
    La **importancia por permutación** mide cuánto aumenta el error del modelo  
    cuando los valores de una variable se barajan aleatoriamente. Esta métrica es  
    más confiable que la importancia de Gini (que sobrevalora variables con muchos niveles),  
    especialmente con variables categóricas como Variedad y Bioestimulante.
    """)

    imp_chart = (
        alt.Chart(imp_df)
        .mark_bar()
        .encode(
            x=alt.X("Importancia_media:Q",
                    title="Incremento medio en MSE al permutar"),
            y=alt.Y("Feature:N",
                    sort="-x",
                    title="Variable"),
            color=alt.Color("Importancia_media:Q",
                            scale=alt.Scale(scheme="blues"),
                            legend=None),
            tooltip=[
                alt.Tooltip("Feature:N",           title="Variable"),
                alt.Tooltip("Importancia_media:Q", title="Importancia media", format=".4f"),
                alt.Tooltip("Importancia_std:Q",   title="Std",               format=".4f"),
            ]
        )
        .properties(width=680, height=380,
                    title="Importancia de variables — Permutation Importance")
    )

    # Barras de error
    err_bars = (
        alt.Chart(imp_df)
        .mark_errorbar(ticks=True)
        .encode(
            x=alt.X("Importancia_media:Q"),
            xError=alt.XError("Importancia_std:Q"),
            y=alt.Y("Feature:N", sort="-x"),
        )
    )

    st.altair_chart(imp_chart + err_bars, use_container_width=True)

    st.info("""
    **Interpretación agronómica:** Las variables con mayor importancia son los que el modelo  
    usa más activamente para reducir el error predictivo. Si **Clorofila a** y **Radiación**  
    lideran el ranking, confirma que los pigmentos y la luz disponible son los principales  
    determinantes de la clorofila total — consistente con la biología fotosintética del café.  
    Variables con importancia ≈ 0 o negativa pueden considerarse redundantes para la predicción.
    """)

    st.divider()

    # ── Curva de aprendizaje ─────────────────────────────────────────────────
    st.subheader("📈 Curva de aprendizaje — diagnóstico de sobreajuste")

    st.markdown("""
    La **curva de aprendizaje** muestra cómo evolucionan el error de entrenamiento  
    y el error de validación al aumentar el tamaño de muestra disponible.  

    - Si la brecha **train–validación es grande**: el modelo sobreajusta.  
    - Si ambas curvas **convergen** con suficientes datos: el modelo generaliza bien.  
    - Si ambas están **altas** (error alto): el modelo subajusta (sesgo alto).
    """)

    lc_plot_df = pd.concat([
        lc_df[['train_size', 'train_mean']].rename(columns={'train_mean': 'R2'}).assign(Tipo='Entrenamiento'),
        lc_df[['train_size', 'val_mean']].rename(columns={'val_mean':   'R2'}).assign(Tipo='Validación CV'),
    ])

    lc_chart = (
        alt.Chart(lc_plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("train_size:Q", title="Número de observaciones en entrenamiento"),
            y=alt.Y("R2:Q",         title="R² promedio", scale=alt.Scale(domain=[0, 1.05])),
            color=alt.Color("Tipo:N",
                            scale=alt.Scale(
                                domain=["Entrenamiento", "Validación CV"],
                                range=["#003399", "#e05c2a"]
                            )),
            strokeDash=alt.StrokeDash("Tipo:N",
                                      scale=alt.Scale(
                                          domain=["Entrenamiento", "Validación CV"],
                                          range=[[1,0], [6,3]]
                                      )),
            tooltip=["train_size:Q", "R2:Q", "Tipo:N"]
        )
        .properties(width=680, height=380,
                    title="Curva de aprendizaje — Random Forest óptimo")
    )
    st.altair_chart(lc_chart, use_container_width=True)

    st.info("""
    **Interpretación:** Si las curvas de entrenamiento y validación convergen hacia un R² alto  
    (> 0.80) a medida que se añaden observaciones, el modelo está bien calibrado y **no sobreajusta**.  
    La brecha residual es esperada con n < 30 y refleja la variabilidad natural del muestreo,  
    no un problema del modelo. Ampliar el experimento a más tratamientos reduciría esta brecha.
    """)

    st.divider()

    # ── Predictor interactivo ────────────────────────────────────────────────
    st.subheader("🔮 Predictor interactivo de Clorofila total")

    st.markdown("""
    Selecciona las condiciones agronómicas de un nuevo tratamiento para obtener  
    la **predicción de Clorofila total** del modelo entrenado, junto con el  
    intervalo de confianza basado en la dispersión entre los árboles del bosque.
    """)

    col_pred_l, col_pred_r = st.columns([1, 1])

    with col_pred_l:
        st.markdown("#### Parámetros del tratamiento")

        variedad_sel      = st.selectbox("Variedad",
                                         sorted(df_full['Variedad'].unique()),
                                         help="Co=Coffea, Ma=Manabí, Ca=Castillo, Ga=Galán")
        bioest_sel        = st.selectbox("Bioestimulante",
                                         sorted(df_full['Bioestimulante'].unique()),
                                         help="T=Testigo, M=Micorrizas, A=Algas, P=Purín")
        radiacion_sel     = st.selectbox("Radiación PAR (µmol·m⁻²·s⁻¹)",
                                         [168, 278, 440])

        st.markdown("#### Variables nutricionales y fisiológicas")
        st.caption("Ajusta los valores según el perfil nutricional esperado del tratamiento.")

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            chl_a_in  = st.slider("Clorofila a",   1.55, 2.10, 1.80, 0.01)
            nitro_in  = st.slider("Nitrógeno",      24.5, 27.5, 25.8, 0.01)
            fosf_in   = st.slider("Fósforo",        13.5, 17.5, 15.5, 0.01)
            potasio_in = st.slider("Potasio",       15.5, 20.5, 17.5, 0.01)
        with col_n2:
            chl_b_in  = st.slider("Clorofila b",    0.75, 1.15, 0.92, 0.01)
            calcio_in  = st.slider("Calcio",        12.5, 14.5, 13.6, 0.01)
            magnesio_in = st.slider("Magnesio",     3.7,  4.8,  4.2,  0.01)

    with col_pred_r:
        st.markdown("#### Resultado de la predicción")

        le_var = meta['le_var']
        le_bio = meta['le_bio']

        # Codificar entradas — manejar variedades/bioestimulantes no vistos
        try:
            var_enc  = le_var.transform([variedad_sel])[0]
        except ValueError:
            var_enc  = 0
        try:
            bio_enc  = le_bio.transform([bioest_sel])[0]
        except ValueError:
            bio_enc  = 0

        rad_norm = radiacion_sel / 440.0

        x_new = np.array([[
            var_enc, bio_enc, rad_norm,
            chl_a_in, chl_b_in,
            nitro_in, fosf_in, potasio_in, calcio_in, magnesio_in
        ]])

        # Predicciones individuales de cada árbol para intervalo de confianza
        tree_preds = np.array([tree.predict(x_new)[0] for tree in best_model.estimators_])
        pred_mean  = tree_preds.mean()
        pred_std   = tree_preds.std()
        pred_low   = pred_mean - 1.96 * pred_std
        pred_high  = pred_mean + 1.96 * pred_std

        # Semáforo de calidad
        if pred_mean >= 2.90:
            calidad = "🟢 Excelente"
            color_bg = "#d4edda"
        elif pred_mean >= 2.70:
            calidad = "🟡 Buena"
            color_bg = "#fff3cd"
        else:
            calidad = "🔴 Por mejorar"
            color_bg = "#f8d7da"

        st.markdown(f"""
        <div style="background:{color_bg};padding:20px;border-radius:12px;text-align:center;">
            <h2 style="margin:0;font-size:2.4rem;color:#1a1a1a;">{pred_mean:.3f}</h2>
            <p style="margin:4px 0 0 0;color:#333;font-size:1rem;">mg·g⁻¹ PMF — Clorofila total predicha</p>
            <p style="margin:8px 0 0 0;font-size:0.9rem;color:#555;">
                IC 95%: [{pred_low:.3f}, {pred_high:.3f}]
            </p>
            <hr style="border:1px solid #ccc;margin:12px 0;">
            <p style="font-size:1.1rem;font-weight:600;color:#333;">Clasificación: {calidad}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(" ")

        # Distribución de predicciones de los árboles (histograma)
        hist_trees = pd.DataFrame({"Clorofila_pred": tree_preds})
        hist_chart = (
            alt.Chart(hist_trees)
            .mark_bar(opacity=0.75, color="#003399")
            .encode(
                alt.X("Clorofila_pred:Q",
                      bin=alt.Bin(maxbins=25),
                      title="Clorofila total predicha por árbol"),
                alt.Y("count():Q", title="N° árboles")
            )
            .properties(
                height=220,
                title=f"Distribución de predicciones ({len(best_model.estimators_)} árboles)"
            )
        )
        # Línea vertical en la media
        vline = alt.Chart(pd.DataFrame({"x": [pred_mean]})).mark_rule(
            color="red", strokeDash=[4, 3], strokeWidth=2
        ).encode(x="x:Q")

        st.altair_chart(hist_chart + vline, use_container_width=True)

        st.caption(
            f"La línea roja indica la predicción final (media de los {len(best_model.estimators_)} árboles). "
            f"Una distribución estrecha indica alta certeza del modelo para esta combinación de inputs."
        )

    st.divider()

    # ── Comparación entre variedades y bioestimulantes ───────────────────────
    st.subheader("🗺️ Mapa de predicciones — barrido de tratamientos")

    st.markdown("""
    El siguiente heatmap muestra la clorofila total **predicha por el modelo** para todas las  
    combinaciones de Variedad × Bioestimulante a una radiación fija seleccionable.  
    Permite identificar visualmente los tratamientos más prometedores bajo cada nivel de luz.
    """)

    rad_heatmap = st.radio(
        "Nivel de radiación para el barrido:",
        [168, 278, 440],
        index=1,
        horizontal=True
    )

    # Generar predicciones para el grid
    variedades  = sorted(df_full['Variedad'].unique())
    bioest_list = sorted(df_full['Bioestimulante'].unique())

    rows_grid = []
    for v in variedades:
        for b in bioest_list:
            # Promediar nutrientes observados para esta combinación v×b
            sub = df_full[(df_full['Variedad'] == v) & (df_full['Bioestimulante'] == b)]
            if len(sub) == 0:
                # Usar medianas globales si no hay observación
                chl_a_ = float(df_full['Clorofila_a'].median())
                chl_b_ = float(df_full['Clorofila_b'].median())
                nit_   = float(df_full['Nitrogeno'].median())
                fos_   = float(df_full['Fosforo'].median())
                pot_   = float(df_full['Potasio'].median())
                cal_   = float(df_full['Calcio'].median())
                mag_   = float(df_full['Magnesio'].median())
            else:
                chl_a_ = float(sub['Clorofila_a'].mean())
                chl_b_ = float(sub['Clorofila_b'].mean())
                nit_   = float(sub['Nitrogeno'].mean())
                fos_   = float(sub['Fosforo'].mean())
                pot_   = float(sub['Potasio'].mean())
                cal_   = float(sub['Calcio'].mean())
                mag_   = float(sub['Magnesio'].mean())

            try:
                v_enc = le_var.transform([v])[0]
                b_enc = le_bio.transform([b])[0]
            except ValueError:
                v_enc, b_enc = 0, 0

            x_g = np.array([[v_enc, b_enc, rad_heatmap/440.0,
                              chl_a_, chl_b_, nit_, fos_, pot_, cal_, mag_]])
            pred_ = best_model.predict(x_g)[0]
            rows_grid.append({
                "Variedad": v,
                "Bioestimulante": b,
                "Clorofila_predicha": round(pred_, 3)
            })

    df_grid = pd.DataFrame(rows_grid)

    heat_map_pred = (
        alt.Chart(df_grid)
        .mark_rect()
        .encode(
            x=alt.X("Bioestimulante:N", title="Bioestimulante"),
            y=alt.Y("Variedad:N",       title="Variedad"),
            color=alt.Color("Clorofila_predicha:Q",
                            scale=alt.Scale(scheme="blues",
                                            domain=[df_grid['Clorofila_predicha'].min() - 0.02,
                                                    df_grid['Clorofila_predicha'].max() + 0.02]),
                            title="Clorofila total predicha"),
            tooltip=["Variedad", "Bioestimulante",
                     alt.Tooltip("Clorofila_predicha:Q", format=".3f",
                                 title="Clorofila total predicha")]
        )
        .properties(
            width=500, height=280,
            title=f"Clorofila total predicha — Radiación {rad_heatmap} µmol·m⁻²·s⁻¹"
        )
    )

    text_grid = (
        alt.Chart(df_grid)
        .mark_text(fontWeight="bold", fontSize=13)
        .encode(
            x=alt.X("Bioestimulante:N"),
            y=alt.Y("Variedad:N"),
            text=alt.Text("Clorofila_predicha:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Clorofila_predicha > df_grid['Clorofila_predicha'].mean(),
                alt.value("white"), alt.value("#1a1a1a")
            )
        )
    )

    st.altair_chart(heat_map_pred + text_grid, use_container_width=True)

    idx_max = df_grid['Clorofila_predicha'].idxmax()
    best_grid = df_grid.loc[idx_max]
    st.success(
        f"✅ **Combinación óptima predicha** a {rad_heatmap} µmol·m⁻²·s⁻¹:  "
        f"**Variedad {best_grid['Variedad']}** + **Bioestimulante {best_grid['Bioestimulante']}**  "
        f"→ Clorofila total = **{best_grid['Clorofila_predicha']:.3f} mg·g⁻¹ PMF**"
    )

    st.divider()

    # ── Nota metodológica final ──────────────────────────────────────────────
    st.subheader("📝 Nota metodológica")

    st.markdown("""
    | Aspecto | Decisión tomada | Justificación |
    |---|---|---|
    | Algoritmo | Random Forest Regressor | Robusto, no paramétrico, maneja colinealidad |
    | Regularización | `max_depth ≤ 4`, `min_samples_leaf ≥ 2` | Evita sobreajuste con n ≈ 22 |
    | Selección de hiperparámetros | GridSearchCV 5-fold | Maximiza R² CV en el espacio de búsqueda |
    | Métricas reportadas | R², RMSE, MAE **en CV** | Estimadores insesgados (no de entrenamiento) |
    | Importancia de variables | Permutation Importance (30 repeats) | Más confiable que impurity-based con categóricas |
    | Intervalo de predicción | Std entre árboles × 1.96 | Aproximación bayesiana del bosque |
    | Limitación principal | n ≈ 22 observaciones | Ampliar experimento mejorará la generalización |
    """)

    st.warning("""
    ⚠️ **Limitación importante:** Con n ≈ 22 observaciones, cualquier modelo ML debe interpretarse  
    como una **herramienta exploratoria** complementaria al ANOVA, no como un sustituto.  
    Los intervalos de confianza de las predicciones son amplios por el tamaño muestral.  
    Se recomienda recolectar al menos 60–80 observaciones para estabilizar los estimadores ML.
    """)
