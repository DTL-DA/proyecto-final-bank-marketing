"""Sección 5: modelo original L2E (motivación, formulación, pseudocódigo, propiedades, código, pruebas, complejidad,
optimización y ablación)."""

S51_MD = r'''## 5. Modelo original: logística de estado latente en dos escalas (L2E)

### 5.1 Motivación y decisiones de diseño

El diagnóstico de las secciones 1 y 2 deja tres hechos, y cada uno se traduce en una decisión de diseño:

1. **El nivel de conversión de cada día es un proceso persistente** (autocorrelación del logit diario de la sección 2.2). Entonces, el intercepto del modelo debe ser un estado latente que evoluciona entre días y que se estima de forma conjunta con el efecto de las variables, en lugar de un número fijo para todo el periodo (componente **C1**), y que en producción se actualiza con las etiquetas de cada día cerrado (componente **C2**).
2. **La relación de varias variables con la suscripción cambia entre 2008 y 2009** (deriva de concepto en `job`, `education`, `default`, `day_of_week`, `marital` y `age`; sección 3.6.5 del tercer entregable), pero mucho más despacio que el nivel. Así mismo, las pendientes deben poder derivar, en una escala de tiempo más lenta que la del nivel (componente **C3**). Separar las dos escalas es lo que distingue la propuesta de una logística dinámica de una sola tasa de olvido.
3. **Las etiquetas llegan por días y con demora.** El resultado de una llamada se conoce al cerrar la jornada, y para predecir el día *d* solo están disponibles las etiquetas de los días anteriores. En ese sentido, la regla de predicción debe respetar esa demora por construcción, y como el nivel de un día nuevo es incierto, la probabilidad predictiva debe integrar esa incertidumbre (componente **C4**).

El modelo resultante no usa ninguna variable nueva: recibe exactamente las mismas predictoras preprocesadas que los benchmarks, más el identificador del día de contacto que ya se reconstruía en el tercer entregable para la validación cronológica.'''

S52_MD = r'''### 5.2 Formulación matemática

**Modelo de observación.** Para el registro $i$, contactado el día $t(i)$ y con predictoras preprocesadas $x_i \in \mathbb{R}^p$,

$$P(y_i = 1 \mid \theta_{t(i)}) = \sigma\!\left(u_{t(i)} + x_i^\top \beta_{t(i)}\right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \theta_t = (u_t, \beta_t). \tag{1}$$

**Dinámica en dos escalas.** Si $t'$ es el día de contacto anterior y $\Delta_t = t - t'$ el número de días transcurridos,

$$u_t = u_{t'} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,\, q_u \Delta_t); \qquad \beta_t = \beta_{t'} + \nu_t,\quad \nu_t \sim \mathcal{N}(0,\, q_\beta \Delta_t I_p); \qquad \beta \sim \mathcal{N}(0,\, \lambda^{-1} I_p), \tag{2}$$

con $q_\beta \ll q_u$: el nivel de conversión se mueve de un día a otro (escala rápida) y las pendientes cambian a lo largo de meses (escala lenta). $\lambda$, $q_u$ y $q_\beta$ son los tres hiperparámetros del modelo.

**Fase 1: ajuste por moda posterior.** Dentro de la ventana de entrenamiento, con días $1, \dots, T$, las pendientes se toman constantes (aproximación de dos escalas) y se maximiza la log-posterior penalizada

$$J(u, \beta) = \sum_{i=1}^{n} \left[ y_i \eta_i - \log\!\left(1 + e^{\eta_i}\right) \right] - \sum_{k=2}^{T} \frac{(u_k - u_{k-1})^2}{2 q_u \Delta_k} - \frac{\epsilon}{2}\lVert u \rVert^2 - \frac{\lambda}{2} \lVert \beta \rVert^2, \qquad \eta_i = u_{t(i)} + x_i^\top \beta, \tag{3}$$

donde $\epsilon = 10^{-6}$ es una precisión difusa sobre el nivel. Con $\pi_i = \sigma(\eta_i)$ y $w_i = \pi_i (1 - \pi_i)$, el gradiente es $g_u = \left(\sum_{i : t(i) = t} (y_i - \pi_i)\right)_t - (D + \epsilon I) u$ y $g_\beta = X^\top (y - \pi) - \lambda \beta$, y el Hessiano negativo tiene la estructura por bloques

$$H = \begin{pmatrix} A & B \\ B^\top & C \end{pmatrix}, \qquad A = \operatorname{diag}(W_t) + D + \epsilon I, \quad B_{t\cdot} = \sum_{i : t(i) = t} w_i x_i^\top, \quad C = X^\top W X + \lambda I, \tag{4}$$

donde $W_t = \sum_{i : t(i) = t} w_i$ y $D = L^\top \operatorname{diag}\!\left(1/(q_u \Delta_k)\right) L$, con $L$ la matriz de primeras diferencias: $A$ es **tridiagonal**. El paso de Newton se obtiene eliminando el nivel con el complemento de Schur,

$$S = C - B^\top A^{-1} B, \qquad \delta_\beta = S^{-1}\left(g_\beta - B^\top A^{-1} g_u\right), \qquad \delta_u = A^{-1}\left(g_u - B \delta_\beta\right), \tag{5}$$

y $A^{-1}$ solo se aplica a vectores mediante una resolución tridiagonal. Al converger, la aproximación de Laplace da la media y la covarianza del estado al cierre del entrenamiento, sin invertir $H$ completo: con $v = A^{-1} e_T$,

$$\operatorname{Cov}(\beta) = S^{-1}, \qquad \operatorname{Cov}(u_T, \beta) = -v^\top B\, S^{-1}, \qquad \operatorname{Var}(u_T) = v_T + v^\top B\, S^{-1} B^\top v. \tag{6}$$

**Fase 2: filtrado con etiquetas diferidas.** Sea $z_i = (1, x_i)$ y $(m, P)$ la media y la covarianza vigentes del estado. Para un día nuevo $d$, la predicción del estado es

$$m^- = m, \qquad P^- = P + \Delta_d \operatorname{diag}(q_u, q_\beta, \dots, q_\beta). \tag{7}$$

Cuando se cierran las etiquetas del día $d'$ (solo si $d' \le d - L$, con $L$ el retardo), el estado se actualiza con la moda de la posterior de ese día y su curvatura (aproximación de Laplace):

$$\theta^+ = \arg\max_\theta \sum_{i : t(i) = d'} \left[ y_i z_i^\top \theta - \log\!\left(1 + e^{z_i^\top \theta}\right) \right] - \tfrac{1}{2} (\theta - m^-)^\top (P^-)^{-1} (\theta - m^-), \qquad P^+ = \left( \sum_{i : t(i) = d'} w_i z_i z_i^\top + (P^-)^{-1} \right)^{-1}. \tag{8}$$

**Probabilidad predictiva.** Integrando la incertidumbre del estado con la aproximación probit (MacKay, 1992),

$$\hat p_i = \sigma\!\left( \frac{\mu_i}{\sqrt{1 + \pi s_i^2 / 8}} \right), \qquad \mu_i = z_i^\top m^-, \qquad s_i^2 = z_i^\top P^- z_i. \tag{9}$$'''

S53_MD = r'''### 5.3 Pseudocódigo

```text
Algoritmo 1. Ajuste por moda posterior (fit)
Entradas : X ∈ R^{n×p}, días t ∈ Z^n (no decrecientes), y ∈ {0,1}^n, hiperparámetros λ > 0, q_u > 0
Salidas  : nivel suavizado û_1..û_T, pendientes β̂, estado (m, P) de (u_T, β) al cierre del entrenamiento
 1. Días únicos d_1 < … < d_T; índice τ(i) ∈ {1..T}; Δ_k = d_k − d_{k−1}
 2. u ← logit(ȳ)·1_T ;  β ← 0_p
 3. repetir para iter = 1..max_iter
 4.     η_i ← u_{τ(i)} + x_iᵀβ ;  π_i ← σ(η_i) ;  w_i ← π_i(1 − π_i)
 5.     g_u, g_β ← gradientes de la ecuación (3)
 6.     A ← diag(W_t) + D + εI (tridiagonal) ;  B ← Σ_i w_i e_{τ(i)} x_iᵀ ;  C ← XᵀWX + λI
 7.     resolver A [Ã_B, ã_g] = [B, g_u] (tridiagonal, p + 1 lados derechos)
 8.     S ← C − Bᵀ Ã_B ;  δ_β ← S⁻¹ (g_β − Bᵀ ã_g) (Cholesky) ;  δ_u ← ã_g − Ã_B δ_β
 9.     λ²_N ← g_uᵀδ_u + g_βᵀδ_β                               (decremento de Newton)
10.     si λ²_N / 2 ≤ tol: terminar (convergencia)
11.     s ← 1 ; mientras J(u + sδ_u, β + sδ_β) < J(u, β) + 0,25·s·λ²_N : s ← s/2   (Armijo)
12.     u ← u + sδ_u ;  β ← β + sδ_β
13. m ← (u_T, β) ;  P ← covarianza de Laplace de la ecuación (6)

Algoritmo 2. Predicción precuencial con etiquetas diferidas (predict_proba_secuencial)
Entradas : estado (m, P) y último día asimilado t*, registros nuevos (X, días, y), retardo L ≥ 1, q_u, q_β
Salidas  : probabilidad p̂_i para cada registro; el estado del estimador no se modifica
 1. pendientes ← cola vacía
 2. para cada día d de los registros, en orden cronológico
 3.     mientras pendientes ≠ ∅ y primero(pendientes) ≤ d − L          (solo días ya cerrados)
 4.         d' ← extraer(pendientes) ;  P⁻ ← P + (d' − t*)·diag(q_u, q_β, …, q_β)
 5.         (m, P) ← moda y curvatura de la ecuación (8) (Newton en 1 + p dimensiones) ;  t* ← d'
 6.     P_d ← P + (d − t*)·diag(q_u, q_β, …, q_β)
 7.     p̂_i ← σ(μ_i / sqrt(1 + π s²_i / 8)) para los registros del día d   (ecuación 9)
 8.     agregar d a pendientes
Criterio de parada: el algoritmo termina al recorrer el último día; cada actualización (paso 5) termina
cuando el paso de Newton es menor que 1e−10 en norma infinito (a lo sumo 50 iteraciones).
```'''

S54_MD = r'''### 5.4 Propiedades teóricas

**P1. Existencia y unicidad del ajuste.** El objetivo (3) es estrictamente cóncavo y tiene un único maximizador. *Demostración.* La log-verosimilitud de Bernoulli es cóncava en $\eta$ y $\eta$ es lineal en $(u, \beta)$, así que su Hessiano es $-Z^\top W Z \preceq 0$. Las penalizaciones aportan $-\operatorname{blockdiag}(D + \epsilon I,\ \lambda I)$, con $D \succeq 0$ y $\epsilon, \lambda > 0$, que es definida negativa. La suma es definida negativa en todo punto, luego $J$ es estrictamente cóncava. Además $J \le -\tfrac{\epsilon}{2}\lVert u \rVert^2 - \tfrac{\lambda}{2}\lVert \beta \rVert^2$, porque la log-verosimilitud es no positiva, así que $J \to -\infty$ cuando $\lVert (u, \beta) \rVert \to \infty$ (coercividad) y el máximo existe; por la concavidad estricta es único. $\blacksquare$

**P2. Convergencia del ajuste.** $J$ es dos veces continuamente diferenciable, fuertemente cóncava (módulo $\ge \min(\epsilon, \lambda)$) y su Hessiano es Lipschitz en el conjunto de nivel inicial, porque $\sigma''$ es acotada. Bajo esas condiciones, Newton con búsqueda lineal de retroceso (Armijo) converge globalmente al maximizador y, cerca de él, con orden cuadrático (Boyd y Vandenberghe, 2004, sección 9.5.3). El criterio de parada usa el decremento de Newton $\lambda_N^2 = g^\top H^{-1} g$, cuyo valor $\lambda_N^2/2$ estima la distancia al óptimo $J^\star - J$ (ecuación 9.29 de la misma obra). Por construcción, la secuencia $J^{(k)}$ es no decreciente; la prueba unitaria `test_objetivo_no_decrece_y_converge` y la sección 5.8 lo verifican.

**P3. Complejidad.** En cada iteración de la fase 1, formar $X^\top W X$ cuesta $O(np^2)$; $B$, $O(np)$; la resolución tridiagonal de $A$ con $p + 1$ lados derechos, $O(Tp)$; el producto $B^\top A^{-1} B$, $O(Tp^2)$, y la factorización de Cholesky de $S$, $O(p^3)$. El costo total es $O(I \cdot (np^2 + Tp^2 + p^3))$ en tiempo y $O(np + Tp)$ en memoria, con $I$ iteraciones de Newton (entre 5 y 10 en la práctica), frente a $O(I \cdot (T + p)^3)$ y $O((T + p)^2)$ de resolver el sistema completo. En la fase 2, cada día con $n_d$ registros cuesta $O(k_d (n_d p^2 + p^3))$, con $k_d$ iteraciones de Newton, de modo que predecir $n$ registros en $D$ días cuesta $O(np^2 + Dp^3)$: lineal en el número de registros.

**P4. Unicidad de cada actualización diaria.** El objetivo de la ecuación (8) es la suma de una log-verosimilitud cóncava y de $-\tfrac{1}{2}(\theta - m^-)^\top (P^-)^{-1}(\theta - m^-)$, con $(P^-)^{-1} \succ 0$; es estrictamente cóncavo y tiene un único maximizador, aunque el día tenga un solo registro o una sola clase. Esto es lo que permite actualizar el estado con días de pocos contactos, frecuentes en 2009-2010 (sección 2.2), donde un estimador de máxima verosimilitud del día no existe.

**P5. Casos límite: el L2E anida a los modelos anteriores.**

- *Logística con penalización L2.* Con un solo intercepto ($q_u \to 0$ en el ajuste) y sin actualización en línea, el L2E es la logística L2 de scikit-learn con $C = 1/\lambda$; la prueba `test_caso_limite_equivale_a_logistica_l2` lo verifica con tolerancia de $10^{-4}$.
- *Efectos fijos por día.* Con $q_u \to \infty$, la penalización del paseo aleatorio desaparece y cada día tiene su propio intercepto; el componente C1 interpola entre la logística agrupada y los efectos fijos.
- *Recalibración mensual del intercepto (entregables 2 y 3).* Con las pendientes fijas, una varianza a priori del nivel $P^- \to \infty$, lotes mensuales y un retardo de un mes, la ecuación (8) se reduce a $\max_u \sum_i [y_i(u + o_i) - \log(1 + e^{u + o_i})]$, con $o_i = x_i^\top \beta$, cuya condición de primer orden $\sum_i \sigma(u + o_i) = \sum_i y_i$ es exactamente la ecuación que resolvía la bisección de la recalibración mensual: la probabilidad media del mes cerrado igual a su tasa observada. *Es decir*, el método que sugirió el docente es el caso particular del filtro del L2E sin memoria, sin incertidumbre y sin dinámica de las pendientes.
- *Logística dinámica de una sola escala.* Con $q_\beta = q_u$ se obtiene una regresión logística dinámica con una sola tasa de deriva, en la línea de McCormick et al. (2012); es la variante `escala_unica` de la ablación.

**P6. Causalidad (ausencia de fuga).** En el algoritmo 2, la predicción del día $d$ es función únicamente del estado ajustado y de las etiquetas de los días $d' \le d - L$, porque el paso 3 solo extrae de la cola días que cumplen esa condición antes de predecir. Ninguna etiqueta influye en su propia predicción ni en las de los $L - 1$ días siguientes. La prueba `test_sin_fuga_de_etiquetas` altera las etiquetas desde un día $k$ y verifica que las predicciones hasta $k$ quedan idénticas bit a bit.'''

S55_MD = r'''### 5.5 Implementación

La clase `LogisticaDosEscalas` hereda de `ClassifierMixin` y `BaseEstimator`, implementa `fit`, `predict`, `predict_proba` y `decision_function`, y agrega `predict_proba_secuencial` (algoritmo 2) y `actualizar`, que asimila un día cerrado en el estado del estimador para su uso en producción. El cálculo numérico usa solo `numpy` y `scipy` (`solveh_banded` para el sistema tridiagonal y `cho_factor` para el de Schur); de scikit-learn solo se usan las clases base y las funciones de validación de entradas. Las banderas `nivel_latente`, `actualizar_nivel`, `actualizar_pendientes`, `escala_unica` e `integrar_incertidumbre` apagan cada componente para la ablación sin duplicar código, y las opciones `semivida` y `q_nivel_empirico` implementan las mejoras M3 y M2 evaluadas en la sección 6.13; por defecto están desactivadas y el modelo es exactamente el descrito en las secciones 5.2 a 5.4. El protocolo (pipeline, puntaje precuencial y validación anidada) está en `src/bm/protocolo_original.py`. A continuación, el código completo del modelo:'''

S55_CODE = r'''codigo = Path("src/bm/modelo_original.py").read_text(encoding="utf-8")
display(Markdown(f"```python\n{codigo}\n```"))
print(f"{len(codigo.splitlines())} líneas · dependencias numéricas: numpy y scipy")'''

S56_MD = r'''### 5.6 Pruebas unitarias

Las pruebas (`tests/test_modelo_original.py`, con `pytest`) cubren los cuatro mínimos de la guía (sección 5.4) y las propiedades de la sección 5.4: (i) que el modelo aprende en datos sintéticos con solución conocida, (ii) la reproducibilidad, (iii) el manejo de entradas inválidas y (iv) la compatibilidad con `sklearn.base.clone` y con la validación cruzada, además de la ausencia de fuga, la monotonía del objetivo, la equivalencia con la logística L2 y la coherencia entre la predicción secuencial y la actualización en producción.'''

S56_CODE = r'''import xml.etree.ElementTree as ET

salida = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider",
                         "--junitxml=resultados/pytest.xml"], capture_output=True, text=True)
raiz = ET.parse("resultados/pytest.xml").getroot()
PROPIEDAD = {
    "test_recupera_pendientes_y_nivel": "(i) Aprende: recupera pendientes y nivel simulados",
    "test_filtro_supera_al_modelo_estatico_cuando_el_nivel_deriva": "(i) Aprende: el filtro mejora cuando el nivel deriva",
    "test_ajuste_determinista": "(ii) Reproducibilidad",
    "test_hiperparametros_invalidos": "(iii) Hiperparámetros inválidos",
    "test_datos_invalidos": "(iii) Datos inválidos (NaN, una clase, día no entero, sin ajustar)",
    "test_prediccion_invalida": "(iii) Predicción con entradas inválidas",
    "test_clone_y_parametros": "(iv) clone, get_params y set_params",
    "test_integra_con_validacion_cruzada": "(iv) cross_val_score con TimeSeriesSplit",
    "test_salidas_bien_formadas": "(iv) Forma y rango de las salidas",
    "test_sin_fuga_de_etiquetas": "P6. Ausencia de fuga con retardo 1 y 3",
    "test_objetivo_no_decrece_y_converge": "P1 y P2. Objetivo no decreciente y convergencia",
    "test_caso_limite_equivale_a_logistica_l2": "P5. Caso límite: logística L2 de scikit-learn",
    "test_sin_actualizacion_igual_a_prediccion_estatica": "P5. Sin actualización = predicción estática",
    "test_actualizar_equivale_a_la_prediccion_secuencial": "Coherencia entre actualizar y la predicción secuencial",
    "test_filtro_aislado_sigue_el_nivel": "Componente C2 aislado: sigue un salto de régimen",
    "test_semivida_por_defecto_no_cambia_el_ajuste": "Mejora M3: sin semivida el ajuste no cambia",
    "test_semivida_da_mas_peso_al_final": "Mejora M3: la recencia sigue un cambio de pendiente",
    "test_bayes_empirico_recupera_el_orden_de_q": "Mejora M2: Bayes empírico recupera el orden de q_u",
}
filas = []
for caso in raiz.iter("testcase"):
    base = caso.get("name").split("[")[0]
    estado = "falla" if caso.find("failure") is not None or caso.find("error") is not None else "pasa"
    filas.append({"Prueba": caso.get("name"), "Propiedad verificada": PROPIEDAD.get(base, "—"), "Resultado": estado,
                  "Tiempo (s)": float(caso.get("time"))})
pruebas_u = pd.DataFrame(filas)
tabla(pruebas_u, dec=3, indice=False)
print(salida.stdout.strip().splitlines()[-1])
n_pasan = int((pruebas_u["Resultado"] == "pasa").sum())
para_guion("pruebas_unitarias", {"pasan": n_pasan, "total": len(pruebas_u)})
interpretar(
    f"Pasan {n_pasan} de {len(pruebas_u)} casos de prueba en {num(pruebas_u['Tiempo (s)'].sum(), 1)} s. En los datos "
    "sintéticos, el modelo recupera las pendientes con un error máximo menor que 0,15 y la trayectoria del nivel con "
    "una correlación mayor que 0,9; con el nivel a la deriva, la predicción secuencial mejora la pérdida logarítmica "
    "y el AUC de la predicción estática; y en el caso límite reproduce la logística L2 de scikit-learn.",
    conectar("De esta forma,", "Las propiedades P1, P2, P5 y P6 de la sección 5.4 no quedan solo como demostraciones en "
             "papel: cada una tiene una verificación automática que se vuelve a ejecutar con cada cambio del código. La "
             "prueba de ausencia de fuga es la más importante para la validez de la sección 6, porque garantiza que la "
             "ventaja del modelo no proviene de mirar etiquetas que en producción todavía no existen."),
    conectar("Por lo anterior, conviene", "Mantener estas pruebas como condición para cualquier modificación del "
             "algoritmo, en particular durante la sustentación, donde la guía prevé cambios en vivo."))'''

S57_MD = r'''### 5.7 Complejidad empírica frente a la teórica

La propiedad P3 predice que el tiempo de ajuste crece linealmente con $n$ y con $T$, y de forma cuadrática a cúbica con $p$. Se mide con datos sintéticos (`scripts/ejecutar_analisis.py`), variando un factor a la vez, y se compara con la resolución densa del mismo sistema de Newton, de orden $(T + p)^3$.'''

S57_CODE = r'''comp = pd.read_parquet(ANA / "complejidad.parquet")
pend = {}
fig, axs = plt.subplots(1, 3, figsize=(14, 3.7))
for ax, (exp_, g) in zip(axs, comp.groupby("experimento", sort=False)):
    x, t_ = g[exp_].to_numpy(dtype=float), g["t_ajuste"].to_numpy()
    b_ = np.polyfit(np.log(x), np.log(t_), 1)[0]
    pend[exp_] = b_
    ax.loglog(x, t_, marker="o", color=TINTA, label=f"L2E (Schur + tridiagonal), pendiente {num(b_, 2)}")
    if "t_denso" in g and g["t_denso"].notna().any():
        gd = g.dropna(subset=["t_denso"])
        b_d = np.polyfit(np.log(gd[exp_]), np.log(gd["t_denso"]), 1)[0]
        pend["T (denso)"] = b_d
        ax.loglog(gd[exp_], gd["t_denso"], marker="s", color=NARANJA, label=f"sistema denso, pendiente {num(b_d, 2)}")
    ax.set(title=f"Tiempo de ajuste frente a {exp_}", xlabel=exp_, ylabel="segundos")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
tabla(comp, dec=3, indice=False)
gT = comp[comp["experimento"] == "T"].dropna(subset=["t_denso"])
casi_constante = pend["T"] < 0.3
para_guion("complejidad", {**{k: float(v) for k, v in pend.items()}, "ratio_denso": float(gT["t_denso"].iloc[-1] / gT["t_ajuste"].iloc[-1]), "T_max_denso": int(gT["T"].iloc[-1])})
interpretar(
    f"En escala log-log, el tiempo de ajuste crece con pendiente {num(pend['n'], 2)} frente a $n$ y "
    f"{num(pend['p'], 2)} frente a $p$, y {'prácticamente no cambia' if casi_constante else 'crece'} frente a $T$ "
    f"(pendiente {num(pend['T'], 2)}): con {num(gT['T'].iloc[0])} días tarda {num(gT['t_ajuste'].iloc[0], 3)} s y con "
    f"{num(gT['T'].iloc[-1])}, {num(gT['t_ajuste'].iloc[-1], 3)} s. La resolución densa del mismo sistema pasa de "
    f"{num(gT['t_denso'].iloc[0], 3)} s a {num(gT['t_denso'].iloc[-1], 3)} s en ese rango y, con {num(gT['T'].iloc[-1])} "
    f"días, tarda {num(gT['t_denso'].iloc[-1] / gT['t_ajuste'].iloc[-1], 1)} veces más que el L2E.",
    conectar("Así mismo,", "Las mediciones son coherentes con la propiedad P3. El costo es lineal en $n$, como predice "
             "el término $np^2$. Frente a $T$ casi no cambia, porque con n = 20 000 domina $np^2$ y el término "
             "$Tp^2$ de la eliminación por bloques es pequeño, mientras que el sistema denso empieza a crecer con el "
             "cubo de $T + p$ en cuanto el número de días supera unos cientos. Frente a $p$, la pendiente medida queda "
             "por debajo del orden teórico 2, porque para p ≤ 400 las operaciones matriciales vectorizadas no "
             "alcanzan su régimen asintótico; el orden teórico funciona como cota superior. La eliminación por bloques "
             "es lo que vuelve viable el componente C1, que necesita un intercepto por día."),
    conectar("Con base en esto, se propone", "Usar el L2E sin restricciones de tamaño en este problema (166 días y 49 "
             "predictoras en el entrenamiento) y documentar que su costo es comparable al de la logística del "
             "tercer entregable, como confirma la tabla resumen de la sección 6.2."))'''

S58_MD = r'''### 5.8 Optimización de hiperparámetros en la validación anidada

Los tres hiperparámetros se optimizan con el método bayesiano del tercer entregable (Optuna con el estimador de Parzen en árbol, las mismas 5 configuraciones iniciales aleatorias y 24 candidatos por paso) dentro del bucle interno, con el mismo presupuesto de 20 configuraciones y los mismos pliegues internos. Los espacios de búsqueda son log-uniformes: $\lambda \in [0{,}1;\ 1000]$ (de casi sin penalización a una contracción fuerte, el rango donde el Entregable 2 encontró su óptimo con C = 0,01, es decir, $\lambda = 100$), $q_u \in [10^{-3};\ 1]$ (de un nivel casi constante a uno que puede duplicar su odds de un día a otro) y $q_\beta \in [10^{-7};\ 10^{-2}]$ (de pendientes fijas a pendientes que cambian en semanas). La métrica del bucle interno es el AUC-PR de las predicciones precuenciales del bloque de validación interna.'''

S58_CODE = r'''filas = []
for c in corr["l2e"]:
    filas.append({"Pliegue": c["pliegue"] + 1, "λ": c["params"]["modelo__lam"], "q_u": c["params"]["modelo__q_nivel"],
                  "q_β": c["params"]["modelo__q_pendiente"], "AUC-PR interno": c["puntaje_interno"],
                  "AUC-PR externo": c["auc_pr"], "AUC externo": c["auc"], "Iteraciones de Newton": c["n_iter_newton"],
                  "Optimización (s)": c["t_opt"]})
hp = pd.DataFrame(filas).set_index("Pliegue")
hp.loc["Modelo final (todo el entrenamiento)"] = [L2E["params"]["modelo__lam"], L2E["params"]["modelo__q_nivel"],
                                                   L2E["params"]["modelo__q_pendiente"], L2E["puntaje_interno"],
                                                   np.nan, np.nan, L2E["pipeline"][-1].n_iter_, L2E["t_opt"]]
sty = hp.style.format({"λ": num_c, "q_u": num_c, "q_β": num_c, "AUC-PR interno": "{:.3f}", "AUC-PR externo": "{:.3f}",
                       "AUC externo": "{:.3f}", "Iteraciones de Newton": "{:.0f}", "Optimización (s)": "{:.1f}"},
                      na_rep="—", decimal=",")
display(sty)

fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
for c in corr["l2e"]:
    h = pd.DataFrame(c["historial"])
    axs[0].plot(h["n"], h["mejor"], marker=".", lw=1.3, label=f"pliegue {c['pliegue'] + 1}")
axs[0].set(title="Optuna: mejor AUC-PR interno en cada evaluación", xlabel="Configuraciones evaluadas",
           ylabel="Mejor AUC-PR interno")
axs[0].legend(ncol=2)
obj = np.array(L2E["pipeline"][-1].objetivo_)
axs[1].plot(range(len(obj)), obj, marker="o", color=TINTA)
axs[1].set(title="Convergencia de Newton en el modelo final (objetivo de la ecuación 3)", xlabel="Iteración",
           ylabel="Log-posterior penalizada")
plt.tight_layout()
plt.show()

conv = all(c["convergio"] for v in corr.values() for c in v)
para_guion("hiperparametros", hp.reset_index().astype({"Pliegue": str}))
razon_q = hp["q_u"].iloc[:5] / hp["q_β"].iloc[:5]
salto_q = hp["q_u"].iloc[4] / hp["q_u"].iloc[:4].max()
interpretar(
    f"En los cinco pliegues, Optuna elige $q_u$ entre {num_c(hp['q_u'].iloc[:5].min())} y "
    f"{num_c(hp['q_u'].iloc[:5].max())} y $\\lambda$ entre {num_c(hp['λ'].iloc[:5].min())} y "
    f"{num_c(hp['λ'].iloc[:5].max())}; el modelo final, optimizado con todo el entrenamiento, usa $\\lambda$ = "
    f"{num_c(hp.iloc[-1]['λ'])}, $q_u$ = {num_c(hp.iloc[-1]['q_u'])} y $q_\\beta$ = {num_c(hp.iloc[-1]['q_β'])}. "
    f"Newton converge {'en todas las corridas' if conv else 'en casi todas las corridas'} con entre "
    f"{int(hp['Iteraciones de Newton'].min())} y {int(hp['Iteraciones de Newton'].max())} iteraciones, y el objetivo "
    "del modelo final crece en cada iteración hasta estabilizarse.",
    conectar("Hay que tener en cuenta que", "Los pliegues de 2008 eligen un nivel casi constante ($q_u$ pequeño), porque "
             "dentro de 2008 la tasa diaria varía poco y la validación interna no premia moverla, mientras que el "
             "último pliegue y el modelo final, que ven el cambio de régimen de 2009, eligen un $q_u$ "
             f"{num(salto_q, 0)} veces mayor. La optimización no está imponiendo la dinámica: la descubre solo cuando los datos la contienen. "
             f"En todos los pliegues $q_\\beta$ queda por debajo de $q_u$, entre {num(razon_q.min(), 0)} y "
             f"{num(razon_q.max(), 0)} veces, lo que respalda con los datos la separación en dos escalas de la "
             "ecuación (2)."),
    conectar("Por tal motivo, se opta por", "Conservar el espacio de búsqueda y el presupuesto del tercer entregable, sin "
             "darle al modelo propio más evaluaciones que a los benchmarks, y analizar en la sección 6.9 cuánto cambia "
             "el resultado si se fijan los hiperparámetros en otros valores."))'''

S59_MD = r'''### 5.9 Estudio de ablación

Cada componente se retira por separado y la variante resultante se **optimiza de nuevo** con el mismo protocolo (validación anidada, Optuna, 20 configuraciones), de modo que ninguna variante queda en desventaja por usar hiperparámetros ajustados para otra. Las variantes son: sin nivel latente en el ajuste (C1: un solo intercepto en la fase 1), sin deriva del nivel ($q_u = 0$ en las dos fases), sin actualizar las pendientes (C3: solo se filtra el nivel), sin actualización en línea (C2 y C3: modelo estático con el nivel del último día de entrenamiento), una sola escala ($q_\beta = q_u$) y sin integrar la incertidumbre (C4).'''

S59_CODE = r'''filas = []
for v in VARIANTES:
    ext_ = np.array([c["auc_pr"] for c in corr[v]])
    f_ = fin_l2e[v]
    filas.append({"Variante": NOMBRE_VARIANTE[v], "clave": v, "AUC-PR externo": ext_.mean(), "DE externo": ext_.std(ddof=1),
                  "AUC externo": np.mean([c["auc"] for c in corr[v]]), "AUC-PR prueba": f_["auc_pr"],
                  "AUC prueba": f_["auc"], "Brier prueba": f_["brier"]})
abl = pd.DataFrame(filas)
ref_abl = abl.iloc[0]
abl["Δ AUC-PR externo"] = abl["AUC-PR externo"] - ref_abl["AUC-PR externo"]
abl["Δ AUC-PR prueba"] = abl["AUC-PR prueba"] - ref_abl["AUC-PR prueba"]
tabla(abl.drop(columns="clave"), dec=4, indice=False)

fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
yy = np.arange(len(abl))
for ax, col, tit in ((axs[0], "AUC-PR externo", "Bucle externo (media de 5 pliegues)"),
                     (axs[1], "AUC-PR prueba", "Prueba (2009-2010)")):
    colores = [TINTA if v == "l2e" else GRIS for v in abl["clave"]]
    ax.barh(yy, abl[col], color=colores)
    ax.axvline(ref_abl[col], color=TINTA, lw=1, ls="--")
    ax.set(title=tit, xlabel="AUC-PR")
    ax.set_xlim(abl[col].min() * 0.9, abl[col].max() * 1.03)
axs[0].set_yticks(yy, abl["Variante"])
axs[0].invert_yaxis()
mejor_b = pliegues_e3.merge(fin_ext.iloc[[0]][["modelo", "balanceo", "optimizador"]])["auc_pr"].mean()
axs[0].axvline(mejor_b, color=AZUL, lw=1.2, ls=":", label=f"mejor benchmark ({CORTO[MEJOR]})")
axs[1].axvline(ap_te[mejor_te], color=AZUL, lw=1.2, ls=":", label=f"mejor benchmark en prueba ({CORTO[mejor_te]})")
axs[0].legend(loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=7.5)
axs[1].legend(loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=7.5)
plt.tight_layout()
plt.show()

a = abl.set_index("clave")
para_guion("ablacion", abl)
# La única variante que supera al modelo completo en el bucle externo: ¿la diferencia se distingue del ruido?
ap_sp = np.array([c_["auc_pr"] for c_ in corr["sin_pendientes"]])
ap_l2 = np.array([c_["auc_pr"] for c_ in corr["l2e"]])
nb_sp = st4.t_nadeau_bengio(ap_sp, ap_l2, [c_["n_ent"] for c_ in corr["l2e"]], [c_["n_val"] for c_ in corr["l2e"]])
bay_sp = st4.rangos_signo_bayes(ap_sp, ap_l2, rope=0.01)
gana_sp = int((ap_sp > ap_l2).sum())
para_guion("ablacion_sp", {"dif": ap_sp.mean() - ap_l2.mean(), "gana": gana_sp, "p_nb": nb_sp["p"],
                           "p_rope": bay_sp["p_rope"], "p_mejor": bay_sp["p_a_mejor"]})
rango_ext = a["AUC-PR externo"].rank(ascending=False).astype(int)
rango_te = a["AUC-PR prueba"].rank(ascending=False).astype(int)
bajo_ext = [v for v in ("estatico", "sin_deriva_nivel", "escala_unica") if a.loc[v, "AUC-PR externo"] < mejor_b]
bajo_te = [v for v in ("estatico", "sin_deriva_nivel", "escala_unica") if a.loc[v, "AUC-PR prueba"] < ap_te[mejor_te]]
NV = {"estatico": "sin actualización en línea", "sin_deriva_nivel": "sin deriva del nivel",
      "escala_unica": "con una sola escala"}
interpretar(
    f"Retirar la actualización en línea (C2 y C3) baja el AUC-PR de {num(a.loc['l2e', 'AUC-PR externo'], 3)} a "
    f"{num(a.loc['estatico', 'AUC-PR externo'], 3)} en el bucle externo y de {num(a.loc['l2e', 'AUC-PR prueba'], 3)} a "
    f"{num(a.loc['estatico', 'AUC-PR prueba'], 3)} en la prueba; quitar la deriva del nivel lo deja en "
    f"{num(a.loc['sin_deriva_nivel', 'AUC-PR externo'], 3)} y {num(a.loc['sin_deriva_nivel', 'AUC-PR prueba'], 3)}, y "
    f"usar una sola escala en {num(a.loc['escala_unica', 'AUC-PR externo'], 3)} y "
    f"{num(a.loc['escala_unica', 'AUC-PR prueba'], 3)}. Sin actualizar las pendientes, el bucle externo queda en "
    f"{num(a.loc['sin_pendientes', 'AUC-PR externo'], 3)} y la prueba en {num(a.loc['sin_pendientes', 'AUC-PR prueba'], 3)}; "
    f"sin nivel latente en el ajuste, en {num(a.loc['sin_nivel_latente', 'AUC-PR externo'], 3)} y "
    f"{num(a.loc['sin_nivel_latente', 'AUC-PR prueba'], 3)}; y sin integrar la incertidumbre, en "
    f"{num(a.loc['sin_incertidumbre', 'AUC-PR externo'], 3)} y {num(a.loc['sin_incertidumbre', 'AUC-PR prueba'], 3)}.",
    conectar("Lo anterior indica que", "El componente central es el nivel dinámico. En el bucle externo quedan por "
             f"debajo del mejor benchmark las variantes {enumerar([NV[v] for v in bajo_ext]) or 'ninguna'}, y en la "
             f"prueba, {'solo la variante ' if len(bajo_te) == 1 else 'las variantes '}"
             f"{enumerar([NV[v] for v in bajo_te]) or 'ninguna'}. Sin actualización en línea el modelo pierde su "
             "ventaja en los dos escenarios. Sin deriva del nivel o con una sola escala, la caída se concentra en el "
             f"último pliegue externo ({num(corr['sin_deriva_nivel'][4]['auc_pr'], 3)} y "
             f"{num(corr['escala_unica'][4]['auc_pr'], 3)}, frente a {num(corr['l2e'][4]['auc_pr'], 3)} del modelo "
             "completo), que contiene el cambio de régimen de 2009: con una sola tasa de deriva, la optimización tiene "
             "que elegir entre un nivel ágil con pendientes inestables o pendientes estables con un nivel rígido, que "
             "es la limitación de la logística dinámica clásica que motiva las dos escalas. La actualización de las pendientes aporta en la prueba, donde la "
             "deriva de concepto de 2009-2010 es mayor, pero no en el bucle externo, donde la variante sin pendientes "
             f"es {'mejor' if a.loc['sin_pendientes', 'AUC-PR externo'] > a.loc['l2e', 'AUC-PR externo'] else 'peor'} "
             "que el modelo completo. El nivel latente en el ajuste y la integración de la incertidumbre tienen efectos "
             "pequeños, del orden de la variabilidad entre semillas del optimizador (sección 6.9)."),
    conectar("En consecuencia, se decide", "Mantener el modelo completo, que es el declarado antes de ejecutar los "
             "experimentos (sección 5.1). La variante sin actualizar las pendientes es la única que supera al L2E en el "
             f"bucle externo, por {num(a.loc['sin_pendientes', 'AUC-PR externo'] - a.loc['l2e', 'AUC-PR externo'], 4)}, "
             f"y gana {gana_sp} de los {cfg.K_EXTERNO} pliegues; esa diferencia no se distingue de cero (t corregida de "
             f"Nadeau-Bengio, p = {num(nb_sp['p'], 3)}) y la prueba bayesiana de rangos con signo da una probabilidad de "
             f"{num(bay_sp['p_rope'], 2)} de equivalencia práctica (ROPE de 0,01) frente a {num(bay_sp['p_a_mejor'], 2)} "
             "de que la variante sea mejor. La ablación mide el aporte de cada componente, no elige el modelo, y con el "
             "bucle externo empatado no hay evidencia para retirar un componente que el diseño justifica con la deriva "
             "de concepto del entrenamiento; la prueba no interviene en esta decisión. Así mismo, se reporta de forma "
             "explícita que C1 y C4 son componentes de coherencia del modelo (el mismo paseo aleatorio en el ajuste y en "
             "la predicción, y probabilidades que reflejan la incertidumbre del nivel) más que fuentes de desempeño."))'''
