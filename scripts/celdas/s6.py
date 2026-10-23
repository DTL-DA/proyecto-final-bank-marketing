"""Sección 6: optimización computacional."""

S61_MD = r'''## 6. Optimización computacional

Los experimentos de esta sección se ejecutaron con la máquina libre (después de la validación anidada), con un hilo por proceso salvo en el análisis de paralelización, y cada tiempo es el promedio de dos o tres repeticiones. Los modelos se miden sobre la matriz ya preprocesada (49 columnas densas), para aislar el costo del algoritmo. La máquina tiene 12 núcleos lógicos y no tiene GPU NVIDIA.

### 6.1 Complejidad algorítmica teórica

La tabla resume la complejidad de entrenamiento e inferencia de cada modelo en función de n (observaciones), p (características), B (árboles), k (vecinos), m (componentes de Nyström) y n_sv (vectores de soporte). Las secciones siguientes la contrastan con el tiempo medido y con el **exponente empírico**, la pendiente de log(tiempo) frente a log(n): un exponente cercano a 1 es lineal y cercano a 2, cuadrático.'''

S61_CODE = r'''RC = RES / "computacional"
cp = {f.stem: pd.read_parquet(f) for f in RC.glob("*.parquet")}
with open(RC / "perfil.pkl", "rb") as f:
    perfil = pickle.load(f)
tabla(comp.COMPLEJIDAD, indice=False)
exportar("comp_complejidad", comp.COMPLEJIDAD)
interpretar(
    'La tabla agrupa los algoritmos en tres órdenes de costo de entrenamiento: lineales en n (Naive Bayes, la logística y Ridge/Lasso con SAGA o SGD, el descenso por coordenadas de Lasso, la SVM lineal, Nyström + SVM lineal y XGBoost con histogramas), casi lineales del orden n log n (árbol de decisión, Random Forest, XGBoost exacto y la construcción del KD-Tree y el Ball Tree) y superlineales, de O(n²) a O(n³), en la SVM con núcleo. En inferencia, los modelos paramétricos cuestan O(p) por consulta sin importar el tamaño del entrenamiento, mientras que la KNN exhaustiva cuesta O(n · p) y la SVM con núcleo O(n_sv · p), porque ambas guardan los datos de entrenamiento o parte de ellos.',
    conectar('En ese sentido,', 'Con n = 32 940 observaciones de entrenamiento y p = 49 columnas, los cuellos de botella esperados son el entrenamiento de la SVM con núcleo y la consulta de la KNN, principalmente porque son los únicos cuyo costo crece más que linealmente con n o se paga de nuevo en cada predicción. Así mismo, el término p³ de Cholesky en Ridge es despreciable con 49 columnas, de modo que la ventaja asintótica de SAGA solo aparecería con p grande o con datos dispersos.'),
    conectar('Con base en esto, se propone', 'Concentrar la optimización en tres frentes, que son las estructuras de búsqueda y los vecinos aproximados para la KNN, la aproximación de Nyström con LinearSVC para evitar el núcleo exacto y los histogramas en XGBoost, y contrastar cada orden teórico con el exponente empírico de las secciones siguientes, porque la notación O(·) omite las constantes, el uso de BLAS y el paralelismo interno, que en tamaños moderados pueden pesar más que el orden asintótico.'),
)
for k_, v_ in cp.items():
    if k_ != "perfil_tabla":
        exportar(f"comp_{k_}", v_)
exportar("comp_perfil", {"total_s": perfil["total_s"], "memoria_pico_mb": perfil["memoria_pico_mb"]})


def exponentes(df, x="n", y="Ajuste (s)", grupo="Método"):
    return df.groupby(grupo).apply(lambda g: comp.exponente(g[x], g[y]), include_groups=False).rename("Exponente empírico")


def grafico_escalamiento(ax, df, y, titulo, grupo="Método"):
    for i, (m_, g) in enumerate(df.groupby(grupo, sort=False)):
        ax.loglog(g["n"], g[y], marker="o", color=CAT[i], label=m_)
    ax.set(title=titulo, xlabel="n (escala log)", ylabel=f"{y} (escala log)")
    ax.legend(fontsize=7)'''

S62_MD = r'''### 6.2 KNN: búsqueda exhaustiva, KD-Tree, Ball Tree y vecinos aproximados con FAISS

Se buscan los 40 vecinos de las 8 236 observaciones de prueba entre las 32 940 de entrenamiento. Además del tiempo, se mide el *recall*@40 de cada método frente a la búsqueda exacta y el AUC-PR del clasificador que resulta de la proporción de vecinos positivos, que es el costo en exactitud de la aproximación.'''

S62_CODE = r'''kb = cp["knn_busqueda"]
tabla(kb, dec=4, indice=False)
ke = cp["knn_escalamiento"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
grafico_escalamiento(ax, ke, "Consulta (s)", "KNN: tiempo de consulta de 2 000 puntos frente a n")
plt.tight_layout()
plt.show()
exp_knn = exponentes(ke, y="Consulta (s)")
tabla(exp_knn.to_frame(), dec=2)

bru = kb[kb["Método"] == "sklearn brute"].iloc[0]
kd = kb[kb["Método"] == "sklearn kd_tree"].iloc[0]
hn = kb[kb["Método"].str.contains("HNSW")].iloc[0]
interpretar(
    f"La búsqueda exhaustiva tarda {num(bru['Consulta prueba (s)'], 2)} s en consultar la prueba, el KD-Tree "
    f"{num(kd['Consulta prueba (s)'], 2)} s y FAISS HNSW {num(hn['Consulta prueba (s)'], 2)} s, con un *recall*@40 de "
    f"{num(hn['Recall@k vs exacta'], 3)} frente a la búsqueda exacta y un AUC-PR de {num(hn['AUC-PR (prueba)'], 4)} "
    f"(exacta: {num(bru['AUC-PR (prueba)'], 4)}). El exponente empírico de la consulta es "
    f"{enumerar([num(v, 2) + ' (' + k + ')' for k, v in exp_knn.items()], codigo=False)}.",
    conectar("Esto significa que", "Con 49 dimensiones, la mayoría binarias, los árboles de partición espacial pierden su "
             "ventaja teórica de O(log n): la maldición de la dimensionalidad obliga a visitar muchas hojas y el "
             "KD-Tree se acerca al costo exhaustivo, que además aprovecha BLAS. El grafo HNSW de FAISS sí escala de forma "
             "sublineal porque sacrifica exactitud, y en este problema esa pérdida casi no cambia la métrica."
             if kd["Consulta prueba (s)"] > 0.5 * bru["Consulta prueba (s)"] else
             "Incluso con 49 dimensiones el KD-Tree mantiene una ventaja sobre la búsqueda exhaustiva, porque muchas "
             "columnas binarias reducen la dimensión efectiva; FAISS HNSW la amplía a costa de una pérdida mínima de "
             "exactitud."),
    conectar("Por lo anterior, conviene", "Usar la búsqueda exhaustiva vectorizada en la validación anidada (lo que se "
             "hizo, con `algorithm='brute'`) y reservar FAISS para escenarios de inferencia masiva, donde el ahorro por "
             "consulta compensa la construcción del índice y la pérdida de exactitud es aceptable."))'''

S63_MD = r'''### 6.3 Ridge, Lasso y regresión logística: solucionadores

La guía sugiere SAGA, cuyo costo por época es lineal en n y p y que admite matrices dispersas. Se compara con los solucionadores directos (Cholesky, SVD), con el descenso por coordenadas del Lasso y con SGD, midiendo el tiempo de ajuste y la métrica en prueba.'''

S63_CODE = r'''lin = cp["lineales"]
tabla(lin, dec=4, indice=False)
le = cp["lineales_escalamiento"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
grafico_escalamiento(ax, le, "Ajuste (s)", "Modelos lineales: tiempo de ajuste frente a n")
plt.tight_layout()
plt.show()
exp_lin = exponentes(le)
tabla(exp_lin.to_frame(), dec=2)

rc = lin[(lin["Modelo"] == "Ridge") & (lin["Solucionador"] == "cholesky")].iloc[0]
rs = lin[(lin["Modelo"] == "Ridge") & (lin["Solucionador"] == "saga")].iloc[0]
interpretar(
    f"Ridge con Cholesky se ajusta en {num(rc['Ajuste (s)'], 3)} s y con SAGA en {num(rs['Ajuste (s)'], 3)} s, con un RMSE "
    f"de {num(rc['Valor'], 4)} y {num(rs['Valor'], 4)}. Los exponentes empíricos son "
    f"{enumerar([num(v, 2) + ' (' + k + ')' for k, v in exp_lin.items()], codigo=False)}.",
    conectar("Hay que tener en cuenta que", "Con p = 49, el término p³ de Cholesky es despreciable y su costo O(n · p²) "
             "es una sola pasada muy optimizada por BLAS; SAGA necesita varias épocas para converger, así que su ventaja "
             "asintótica solo aparece cuando p es grande o los datos son dispersos. Ambos resuelven el mismo problema "
             "convexo; la pequeña diferencia de RMSE se debe a que SAGA se detiene por tolerancia antes de converger del "
             "todo."),
    conectar("De ahí que se opte por", "Mantener el solucionador directo para Ridge en este tamaño de problema y SAGA en "
             "la logística L1/L2, donde además es el que admite ambas penalizaciones con un mismo algoritmo."))'''

S64_MD = r'''### 6.4 Naive Bayes: aprendizaje incremental con `partial_fit`

Para simular un conjunto que no cabe en memoria, el archivo se lee por lotes de 2 000 filas, cada lote se transforma con el preprocesador ya ajustado y se descarta después de actualizar las medias y varianzas del modelo.'''

S64_CODE = r'''nb_ = cp["naive_bayes"]
tabla(nb_, dec=4, indice=False)
f0, f1 = nb_.iloc[0], nb_.iloc[1]
interpretar(
    f"El ajuste completo usa {num(f0['Memoria pico (MB)'], 1)} MB de memoria pico y el incremental "
    f"{num(f1['Memoria pico (MB)'], 1)} MB, con tiempos de {num(f0['Tiempo (s)'], 3)} s y {num(f1['Tiempo (s)'], 3)} s; el "
    f"AUC-PR en prueba es {num(f0['AUC-PR prueba'], 4)} con `fit` y {num(f1['AUC-PR prueba'], 4)} con `partial_fit`.",
    conectar("Esto quiere decir que", "Las medias y varianzas de Naive Bayes son estadísticos suficientes que se "
             "actualizan exactamente por lotes; la pequeña diferencia viene del suavizado de varianza, que `partial_fit` "
             "calcula con la varianza máxima del primer lote y `fit` con la de todos los datos. El tiempo "
             "adicional es el de leer y transformar el archivo por partes, y la memoria queda acotada por el tamaño del "
             "lote y no por n."),
    conectar("Por ello, se recomienda", "Usar `partial_fit` cuando los datos crezcan más allá de la memoria disponible; con "
             "41 000 registros no es necesario, pero el resultado demuestra que el pipeline lo admite sin cambios."))'''

S65_MD = r'''### 6.5 XGBoost: histogramas frente al método exacto, parada temprana y GPU'''

S65_CODE = r'''import shutil
xg = cp["xgboost"].copy()
gpu_txt = ("disponible" if (xgboost.build_info().get("USE_CUDA") and shutil.which("nvidia-smi")) else
           "no disponible, porque no hay GPU NVIDIA (con device='cuda', XGBoost recurre a la CPU sin error)")
xg["GPU"] = gpu_txt
tabla(xg, dec=4, indice=False)
xe = cp["xgboost_escalamiento"]
xp = cp["xgboost_parada"]
tabla(xp, dec=4, indice=False)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
grafico_escalamiento(ax, xe, "Ajuste (s)", "XGBoost (200 árboles): tiempo de ajuste frente a n")
plt.tight_layout()
plt.show()
exp_xgb = exponentes(xe)
tabla(exp_xgb.to_frame(), dec=2)

ex = xg[xg["Configuración"].str.contains("exact")].iloc[0]
hi_ = xg[xg["Configuración"].str.contains("hist")].iloc[0]
interpretar(
    f"Con 300 árboles, el método exacto tarda {num(ex['Ajuste (s)'], 2)} s y el de histogramas {num(hi_['Ajuste (s)'], 2)} s "
    f"({num(ex['Ajuste (s)'] / hi_['Ajuste (s)'], 1)} veces menos), con un AUC-PR de {num(ex['AUC-PR prueba'], 4)} y "
    f"{num(hi_['AUC-PR prueba'], 4)}. La parada temprana usa {int(xp['Árboles usados'].iloc[1])} de "
    f"{int(xp['Árboles usados'].iloc[0])} árboles y reduce el ajuste de {num(xp['Ajuste (s)'].iloc[0], 1)} s a "
    f"{num(xp['Ajuste (s)'].iloc[1], 1)} s, con un AUC-PR de {num(xp['AUC-PR prueba'].iloc[1], 4)} frente a "
    f"{num(xp['AUC-PR prueba'].iloc[0], 4)}.",
    conectar("Lo anterior indica que", "Discretizar cada característica en a lo sumo 256 bins elimina el ordenamiento "
             "O(n log n) por nodo del método exacto, y la pérdida de exactitud es nula en la práctica porque la mayoría "
             "de las columnas son binarias o tienen pocos valores distintos. " + ("La parada temprana ahorra cómputo, "
             "pero aquí se detuvo con muy pocos árboles: el conjunto de validación (el final del entrenamiento, ya en "
             "2009) deja de mejorar casi de inmediato porque lo aprendido con 2008 no se traslada, y en prueba pierde "
             "AUC-PR. Su efecto de regularización depende de que la validación represente el periodo de uso."
             if xp["AUC-PR prueba"].iloc[1] < xp["AUC-PR prueba"].iloc[0] else
             "La parada temprana ahorra cómputo y actúa como regularización, siempre que el conjunto de validación sea "
             "independiente y posterior en el tiempo.")),
    conectar("Por tal motivo, se opta por", "`tree_method='hist'` en todos los experimentos (como se hizo) y por no usar "
             "GPU. La máquina no tiene GPU NVIDIA y, con unos 41 000 registros, tampoco se justifica: con histogramas, el "
             f"ajuste en CPU ya tarda {num(hi_['Ajuste (s)'], 2)} s, y scikit-learn está diseñado para CPU (su soporte de "
             "GPU es experimental y cubre pocos estimadores), de modo que el preprocesamiento, el balanceo y los demás "
             "modelos seguirían en CPU."))'''

S66_MD = r'''### 6.6 SVM: núcleo exacto frente a LinearSVC, SGD y aproximación de Nyström'''

S66_CODE = r'''sv = cp["svm_escalamiento"]
tabla(sv, dec=4, indice=False)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
grafico_escalamiento(axs[0], sv, "Ajuste (s)", "SVM: tiempo de ajuste frente a n")
for i, (m_, g) in enumerate(sv.groupby("Método", sort=False)):
    axs[1].semilogx(g["n"], g["AUC-PR prueba"], marker="o", color=CAT[i], label=m_)
axs[1].set(title="SVM: AUC-PR en prueba frente a n", xlabel="n (escala log)", ylabel="AUC-PR")
axs[1].legend(fontsize=7)
plt.tight_layout()
plt.show()
exp_svm = exponentes(sv)
tabla(exp_svm.to_frame(), dec=2)

nmax = sv[sv["Método"].str.startswith("SVC")]["n"].max()
s_k = sv[(sv["Método"].str.startswith("SVC")) & (sv["n"] == nmax)].iloc[0]
s_n = sv[(sv["Método"].str.startswith("Nyström")) & (sv["n"] == nmax)].iloc[0]
interpretar(
    f"El exponente empírico del SVC con núcleo exacto es {num(exp_svm.filter(like='SVC').iloc[0], 2)}, frente a "
    f"{enumerar([num(v, 2) + ' (' + k + ')' for k, v in exp_svm.items() if not k.startswith('SVC')], codigo=False)}. Con "
    f"n = {num(nmax)}, el núcleo exacto tarda {num(s_k['Ajuste (s)'], 1)} s y Nyström + LinearSVC "
    f"{num(s_n['Ajuste (s)'], 2)} s, con AUC-PR de {num(s_k['AUC-PR prueba'], 4)} y {num(s_n['AUC-PR prueba'], 4)}.",
    ("Así mismo, el crecimiento superlineal del núcleo exacto confirma el costo cuadrático a cúbico de SMO. "
     if exp_svm.filter(like="SVC").iloc[0] >= 1.8 else
     "Así mismo, el exponente del núcleo exacto queda por debajo del cuadrático teórico en este rango de n, porque "
     "SMO trabaja sobre los vectores de soporte y la caché del núcleo; aun así, su tiempo absoluto es varias veces el "
     "de la aproximación y su costo teórico crece de forma cuadrática a cúbica cuando n aumenta. ")
    + "Además, con gamma='scale' y C = 1 sin ajustar, el núcleo exacto discrimina peor en prueba que la aproximación de "
      "Nyström con gamma = 0,03: la elección de gamma importa más que la exactitud del núcleo. Con 20 configuraciones "
      "por pliegue en la validación anidada, el núcleo exacto multiplicaría el costo sin ventaja demostrada.",
    conectar("En consecuencia, se decide", "Usar Nyström + LinearSVC como la SVM del proyecto (y Nyström + LinearSVR como "
             "la SVR), con el número de componentes y gamma como hiperparámetros, tal como recomienda la guía para "
             "problemas de gran tamaño de muestra."))'''

S67_MD = r'''### 6.7 Paralelización con `joblib`

Se mide la validación cruzada de un Random Forest (200 árboles) paralelizando entre pliegues con los *backends* `loky` (procesos) y `threading` (hilos), y paralelizando entre árboles dentro del propio bosque. La aceleración es el tiempo con un núcleo dividido por el tiempo con n_jobs núcleos; la eficiencia es la aceleración dividida por n_jobs.'''

S67_CODE = r'''pa = cp["paralelizacion"]
tabla(pa, dec=2, indice=False)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for i, ((niv, be), g) in enumerate(pa.groupby(["Nivel", "Backend"], sort=False)):
    ax.plot(g["n_jobs"], g["Aceleración"], marker="o", color=CAT[i], label=f"{niv} · {be}")
ax.plot(pa["n_jobs"].unique(), pa["n_jobs"].unique(), color=GRIS, ls="--", lw=1, label="Aceleración ideal")
ax.set(title="Aceleración por paralelización", xlabel="n_jobs", ylabel="Aceleración")
ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

mej = pa.loc[pa["Aceleración"].idxmax()]
interpretar(
    f"La mayor aceleración es de {num(mej['Aceleración'], 1)} veces con {int(mej['n_jobs'])} núcleos "
    f"({mej['Nivel']}, {mej['Backend']}), una eficiencia del {pct(mej['Eficiencia'], 0)}. Con 3 pliegues internos, "
    "paralelizar entre pliegues no puede superar una aceleración de 3, porque no hay más tareas que repartir.",
    conectar("Es decir,", "La granularidad decide la escalabilidad: entre pliegues hay pocas tareas grandes y entre árboles "
             "hay muchas tareas pequeñas. Los hilos funcionan bien en scikit-learn y XGBoost porque el núcleo de cómputo "
             "libera el GIL; los procesos de `loky` pagan el arranque de cada trabajador y la copia de los datos, y con "
             "tareas de pocos segundos ese costo fijo anula la ganancia"
             + (" (aquí `loky` entre pliegues fue incluso más lento que un solo núcleo)."
                if (pa[(pa["Backend"] == "loky") & (pa["n_jobs"] > 1)]["Aceleración"] < 1).any() else ".")),
    conectar("Por tal motivo, se opta por", "Paralelizar al nivel más grueso disponible, que en este proyecto son las 700 "
             "tareas (combinación × pliegue externo): 10 procesos independientes con un hilo cada uno, sin sobresuscribir "
             "los núcleos. Herramientas distribuidas como Dask o Ray no se justifican con 41 000 registros que caben en "
             "memoria y una sola máquina; serían la opción si el número de combinaciones o el tamaño de los datos "
             "exigiera varios nodos."))'''

S68_MD = r'''### 6.8 Perfilamiento: dónde está el costo real

Se perfila con `cProfile` (tiempo de CPU por función) y `memory_profiler` (memoria pico) una combinación completa del bucle anidado: XGBoost con SMOTE y optimización bayesiana en el pliegue externo más grande, es decir, 20 configuraciones × 3 pliegues internos más el reajuste final.'''

S68_CODE = r'''pc = cp["perfil_componentes"]
tabla(pc, dec=2, indice=False)
print(f"Tiempo total perfilado: {num(perfil['total_s'], 1)} s · memoria pico: {num(perfil['memoria_pico_mb'], 0)} MB")
dom = pc.iloc[0]
interpretar(
    f"El componente dominante es \"{dom['Componente']}\", con el {num(dom['% del total'], 1)} % del tiempo; le siguen "
    f"{enumerar([f'{r.Componente} ({num(r._3, 1)} %)' for r in pc.iloc[1:3].itertuples()], codigo=False)}. La memoria "
    f"pico de la combinación es de {num(perfil['memoria_pico_mb'], 0)} MB.",
    conectar("Esto significa que", "El costo propio del optimizador (el muestreo de TPE) es despreciable frente al "
             "ajuste de los modelos. "
             + (f"El preprocesamiento sí pesa ({num(pc.set_index('Componente').loc['Preprocesamiento (ColumnTransformer)', '% del total'], 1)} %): "
                "se reajusta en cada uno de los 60 ajustes internos aunque sus parámetros apenas cambian entre "
                "configuraciones, de modo que precalcularlo por pliegue (con `memory` en el `Pipeline`) es la "
                "optimización con mejor relación costo-beneficio después del modelo. "
                if pc.set_index('Componente').loc['Preprocesamiento (ColumnTransformer)', '% del total'] > 10 else
                "El preprocesamiento también es marginal, así que optimizarlo sería una optimización prematura. ")
             + "La memoria no es una restricción en este tamaño de datos."),
    conectar("Por lo anterior, conviene", "Concentrar las mejoras en reducir el número de ajustes (multi-fidelidad, "
             "parada temprana, menos configuraciones malas gracias a métodos adaptativos) y en el método de "
             "construcción de los árboles (`hist`), que son las palancas que actúan sobre el cuello de botella real."))'''

S69_MD = r'''### 6.9 Tabla comparativa: modelo estándar frente a optimizado

La tabla que pide la guía resume, para cada par, la complejidad teórica, el tiempo de ajuste, el tiempo de inferencia sobre la prueba, la memoria del ajuste y la variación del desempeño predictivo (AUC-PR o RMSE en prueba). La memoria es el incremento de la memoria residente del proceso durante el ajuste, medido con `memory_profiler` en un proceso nuevo para cada versión: así incluye las reservas internas de las librerías escritas en C/C++ (XGBoost, FAISS y libsvm), que `tracemalloc` no registra, y no se mezcla con la memoria liberada por ajustes anteriores.'''

S69_CODE = r'''co = cp["comparativa"]
tabla(co, dec=4, indice=False)
pares_c = co.groupby("Modelo", sort=False)
ahorro = pares_c.apply(lambda g: g["Ajuste (s)"].iloc[0] / g["Ajuste (s)"].iloc[1], include_groups=False)
ahorro_i = pares_c.apply(lambda g: g["Inferencia (s)"].iloc[0] / g["Inferencia (s)"].iloc[1], include_groups=False)
dif_m = pares_c["Variación de la métrica"].last()
mem = pares_c["Memoria del ajuste (MB)"].agg(["first", "last"])
def veces(v, fem=False):
    if 0.9 <= v <= 1.1:
        return "similar"
    a = "a" if fem else "o"
    return f"{num(v, 1)} veces más rápid{a}" if v > 1 else f"{num(1 / v, 1 if 1 / v < 10 else 0)} veces más lent{a}"

interpretar(
    "Frente a la versión estándar, el ajuste de la versión optimizada es "
    + enumerar([f"{veces(v)} ({k})" for k, v in ahorro.items()], codigo=False)
    + "; la inferencia es " + enumerar([f"{veces(v, fem=True)} ({k})" for k, v in ahorro_i.items()], codigo=False)
    + ". La memoria del ajuste pasa de " + enumerar([f"{num(r['first'], 1)} a {num(r['last'], 1)} MB ({k})" for k, r in mem.iterrows()], codigo=False)
    + ", y la métrica cambia en " + enumerar([f"{num(v, 4)} ({k})" for k, v in dif_m.items()], codigo=False) + ".",
    conectar("En síntesis,", "Cada optimización actúa sobre una fase distinta: los histogramas abaratan el ajuste de XGBoost "
             "sin perder desempeño; FAISS HNSW encarece la construcción del índice pero abarata cada consulta, con una "
             "pérdida pequeña de AUC-PR; Nyström abarata la SVM y, con un gamma adecuado, incluso mejora la métrica. "
             "SAGA, que solo mejora la complejidad asintótica, es mucho más lento que Cholesky con 49 columnas. En memoria, "
             "los histogramas también reducen el consumo de XGBoost, mientras que el índice de FAISS y la matriz de "
             "componentes de Nyström cambian velocidad por memoria."),
    conectar("Así las cosas, se decide", "Mantener en el pipeline las versiones optimizadas que ganan en tiempo sin perder "
             "desempeño y documentar las que no aplican a este tamaño de problema, en lugar de adoptarlas por defecto."))'''
