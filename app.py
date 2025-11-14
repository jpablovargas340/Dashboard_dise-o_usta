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

st.set_page_config(page_title="Bioestimulación Coffea arabica", layout="wide")
st.title("🌱 Análisis Experimental: Bioestimulación y Radiación Solar en Coffea arabica L.")
st.caption("Universidad Santo Tomás — Vanessa Acosta - Juan Pablo Vargas - Lizeth Rodriguez")

@st.cache_data
def cargar_datos():
    # === 1. Base de Clorofila ===
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

    # === 2. Base de Nutrientes ===
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

    # === 3. Unión ===
    df_full = pd.merge(
        df, df_nut,
        on=['Variedad', 'Bioestimulante', 'Radiacion'],
        how='inner', suffixes=('_chl', '_nut')
    )

    return df, df_nut, df_full

df, df_nut, df_full = cargar_datos()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Introducción y Objetivos",
    "📊 Exploración de Datos",
    "⚗️ Pruebas Estadísticas",
    "🧪 ANOVA y Diagnóstico de Residuos",
    "📈 KPIs y Conclusiones"
])

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
    """)

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

    # --- Clorofila total ---
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
    " **Interpretación:** " \
    "La clorofila total muestra un incremento claro a medida que aumenta la radiación. "
    "A niveles bajos (168 µmol·m⁻²·s⁻¹), los valores se mantienen entre 2.45 y 2.52 mg·g⁻¹, mientras que a 278 µmol·m⁻²·s⁻¹ "
    "la mediana aumenta hacia ~2.66 mg·g⁻¹. Bajo la radiación más alta (440 µmol·m⁻²·s⁻¹), se observan las concentraciones "
    "más elevadas y con mayor variabilidad natural. En conjunto, la radiación ejerce un efecto positivo y proporcional "
    "sobre la síntesis de clorofila total en *Coffea arabica*."
)

    # Nitrógeno
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

    # Scatter Potasio–Clorofila total
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

    # Heatmap de correlaciones
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

    # Cálculo de Shapiro
    resultados = []
    for col in ['Clorofila_a','Clorofila_b','Clorofila_total','Nitrogeno','Fosforo','Potasio','Calcio','Magnesio']:
        W, p = pg.normality(df_full[col], method='shapiro')[['W','pval']].values[0]
        resultados.append([col, round(W,3), round(p,4), "✔ Normal" if p>0.05 else "✖ No normal"])
    df_shapiro = pd.DataFrame(resultados, columns=["Variable","W","p-valor","Conclusión"])

    st.dataframe(df_shapiro, use_container_width=True)

    # INTERPRETACIÓN AUTOMÁTICA
    normales = (df_shapiro['Conclusión']=="✔ Normal").sum()
    st.info(f" **Interpretación:** {normales}/8 variables cumplen normalidad. "
            f"La única variable que viola normalidad es **Clorofila_a**, "
            f"coherente con el análisis original donde presentó desviaciones leves en colas.")

    st.divider()

    # LEVENE
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


    # Cálculo de Levene
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

    # SUPUESTO DE INDEPENDENCIA
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

with tab4:
    st.subheader("Modelo ANOVA (formulación matemática)")

    # Modelo completo
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

    # Residuos vs Ajustados
    scatter_resid = (
        alt.Chart(resid_df)
        .mark_circle(size=70, color="#003399", opacity=0.85)
        .encode(
            x=alt.X(
                "Ajustados:Q",
                title="Valores ajustados",
                scale=alt.Scale(domain=[2.4, 3.1])
            ),
            y=alt.Y(
                "Residuos:Q",
                title="Residuos estandarizados"
            ),
            tooltip=["Ajustados", "Residuos"]
        )
        .properties(
            title="Residuos vs Ajustados",
            width=520,
            height=340
        )
        .interactive()
    )

    st.altair_chart(scatter_resid, use_container_width=True)

    st.info("""
    **Interpretación:** El patrón debe mostrar dispersión aleatoria alrededor de 0.  
    Si no hay forma en los residuos, se cumple la suposición de **linealidad y homocedasticidad**.  
    En este caso, los residuos se distribuyen de manera razonablemente aleatoria.
    """)

    #QQ-Plot
    theoretical_q = norm.ppf(
        (np.arange(1, len(resid_std)+1) - 0.5) / len(resid_std)
    )

    qq_df = pd.DataFrame({
        "Teorico": theoretical_q,
        "Residuos": np.sort(resid_std)
    })

    line_qq = (
        alt.Chart(pd.DataFrame({
            "x": [-3, 3],
            "y": [-3, 3]
        }))
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


    # Histograma
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

    # Residuos vs Orden
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


