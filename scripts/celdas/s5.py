"""Sección 5: optimización de hiperparámetros."""

S51_MD = r'''## 5. Optimización de hiperparámetros

### 5.1 Espacios de búsqueda

Cada modelo tiene **un único espacio de búsqueda, compartido por todos los optimizadores**, con a lo sumo cuatro hiperparámetros (los de mayor efecto documentado en el modelo) y rangos tomados de la práctica habitual y del Entregable 2, no arbitrarios. Las escalas son logarítmicas para `C`, `alpha`, `learning_rate`, `gamma`, `var_smoothing`, `min_samples_leaf` y el número de vecinos, porque su efecto es multiplicativo. Grid Search recorre una rejilla explícita de a lo sumo 20 puntos dentro del mismo espacio, de modo que todos los métodos tienen el mismo presupuesto de **20 configuraciones por pliegue externo**.'''

S51_CODE = r'''filas = []
for (t_, m_), esp in ESPACIOS.items():
    for n_, e in esp.items():
        nombre = n_.split("__")[-1]
        if e["tipo"] == "cat":
            rango, escala = enumerar(e["opciones"], codigo=False), "categórica"
        else:
            rango = f"[{num_c(e['bajo'])}; {num_c(e['alto'])}]"
            escala = ("log" if e["log"] else "lineal") + (", entera" if e["tipo"] == "int" else "")
        rej = REJILLAS[(t_, m_)][n_]
        filas.append({"Tarea": "Clasificación" if t_ == "clf" else "Regresión", "Modelo": NM[m_],
                      "Hiperparámetro": nombre, "Rango": rango, "Escala": escala,
                      "Rejilla (Grid Search)": enumerar([num_c(v) if isinstance(v, float) else v for v in rej],
                                                        codigo=False)})
tabla(pd.DataFrame(filas), indice=False)

just = pd.DataFrame([{"Modelo": NM[m_], "Puntos de la rejilla (clf / reg)":
                      " / ".join(str(n_puntos_rejilla(t_, m_)) for t_ in ("clf", "reg") if (t_, m_) in ESPACIOS),
                      "Justificación del espacio": JUSTIFICACION[m_]} for m_ in JUSTIFICACION])
tabla(just, indice=False)

exportar("espacios", pd.DataFrame(filas))
exportar("justificacion", just)
'''

S52_MD = r'''### 5.2 Métodos de optimización y su justificación

Todos los métodos llaman al mismo evaluador (`src/bm/optimizadores.py`), que ajusta el `Pipeline` completo (preprocesamiento, balanceo y modelo) **dentro de cada pliegue interno**; así, Optuna y DEAP, que no integran esa funcionalidad, tampoco tienen fuga de información. El evaluador cuenta como presupuesto las configuraciones distintas (las repetidas salen de un caché) y registra cada evaluación con su tiempo de pared para las curvas *anytime*.

**Grid Search.** Recorre la rejilla en orden. Garantiza cobertura uniforme de los valores elegidos, pero el número de puntos crece como el producto de los niveles y desperdicia evaluaciones en dimensiones poco influyentes (Bergstra y Bengio, 2012).

**Random Search.** Muestrea el espacio continuo de forma uniforme (log-uniforme en las escalas logarítmicas). Con el mismo presupuesto explora más valores distintos de cada hiperparámetro que la rejilla, que es lo que importa cuando solo unos pocos hiperparámetros son relevantes.

**Optimización bayesiana (Optuna).**

- *Modelo sustituto:* **Tree-structured Parzen Estimator (TPE)** (Bergstra et al., 2011). En lugar de modelar p(y | x) como un proceso gaussiano, TPE divide las configuraciones evaluadas por el cuantil γ de la métrica y estima dos densidades: l(x) con las buenas y g(x) con las demás. Se prefiere al proceso gaussiano porque maneja de forma natural hiperparámetros enteros, categóricos y en escala log, y su costo crece de forma lineal con el número de evaluaciones.
- *Función de adquisición:* **Mejora Esperada (Expected Improvement, EI)**. Bajo TPE, maximizar la EI equivale a maximizar el cociente l(x)/g(x); Optuna lo evalúa en 24 candidatos muestreados de l(x) y elige el de mayor cociente.
- *Exploración y explotación:* las 5 primeras configuraciones son aleatorias (exploración pura, para tener con qué estimar las densidades). Después, la EI privilegia la **explotación** de las regiones con alta densidad de buenas configuraciones, y la exploración se mantiene porque los candidatos se muestrean de l(x), que tiene colas, y porque g(x) penaliza solo las regiones ya evaluadas como malas.

**Algoritmo genético (DEAP).** Cada individuo es un vector de genes en [0, 1] que se decodifica al espacio de búsqueda (escala log, enteros y tramos iguales para las categóricas).

- *Selección:* **torneo de tamaño 3**, que mantiene una presión selectiva moderada y no depende de la escala de la métrica (a diferencia de la ruleta).
- *Cruce:* **binario simulado acotado (SBX, η = 15)** con probabilidad **0,7**, diseñado para genes reales (Deb y Agrawal, 1995): los hijos quedan cerca de los padres con una dispersión controlada por η.
- *Mutación:* **polinomial acotada (η = 20)** con probabilidad **0,3** por individuo y **1/d** por gen.
- *Elitismo y población:* el **mejor individuo** pasa intacto a la siguiente generación y la población es de **6 individuos**. Con un presupuesto de 20 configuraciones, una población mayor dejaría muy pocas generaciones con individuos nuevos; con 6, la población inicial consume 6 evaluaciones y las generaciones siguientes se suceden hasta agotar las 14 restantes. Como los clones de individuos ya evaluados no consumen presupuesto, el número de generaciones crece cuando la población converge (hasta un tope de 50).
- *Diversidad genética:* se registra en cada generación la distancia euclídea media entre individuos, normalizada por √d (sección 5.6).

**Multi-fidelidad: Successive Halving (técnica adicional).** Para Random Forest y XGBoost sin balanceo se implementa `HalvingRandomSearchCV` con el **número de árboles como fidelidad**: 27 configuraciones aleatorias se evalúan con un noveno del máximo de árboles, sobrevive un tercio (**factor de reducción 3**) con el triple de árboles y así hasta el máximo. El número de árboles es una fidelidad natural porque el desempeño de un *ensemble* con pocos árboles está fuertemente correlacionado con el que alcanza con muchos, y el costo crece de forma lineal con ellos; el factor 3 es el valor por defecto de Hyperband (Li et al., 2018) y da tres rondas con el rango de árboles de cada modelo.'''

S53_MD = r'''### 5.3 Comparación de los métodos en el bucle externo

La comparación se hace en tres niveles: la métrica del bucle externo, el costo (tiempo total, tiempo por evaluación y número de evaluaciones) y la prueba de Friedman con bloques = (modelo, balanceo, pliegue externo), que compara los optimizadores dentro de cada bloque y elimina la variación entre modelos.'''

S53_CODE = r'''NUCLEO = ["grid", "random", "bayes", "genetico"]
clf_pl, reg_pl = pliegues[pliegues["tarea"] == "clf"], pliegues[pliegues["tarea"] == "reg"]


def resumen_optimizadores(df, metrica, orden):
    g = df.groupby("optimizador")
    salida = pd.DataFrame({
        "Métrica externa (media ± DE)": g[metrica].apply(ms),
        "Evaluaciones por pliegue": g["n_eval"].mean(), "Evaluaciones (total)": g["n_eval"].sum(),
        "Tiempo total (min)": g["t_opt"].sum() / 60, "Tiempo por evaluación (s)": g["t_por_eval"].mean(),
        "Pliegues": g.size()})
    salida.index = [NO[o] for o in salida.index]
    return salida.loc[[NO[o] for o in orden if NO[o] in salida.index]]


def friedman_optimizadores(df, metrica, orden, signo=1):
    mat = df.pivot_table(index=["modelo", "balanceo", "pliegue"], columns="optimizador", values=metrica)[orden]
    return st.friedman(signo * mat.dropna())


print("Clasificación: AUC-PR del bucle externo, 4 métodos de la guía en los 28 pares modelo-balanceo")
tabla(resumen_optimizadores(clf_pl[clf_pl["optimizador"].isin(NUCLEO)], "auc_pr", NUCLEO), dec=3)
print("Regresión: RMSE del bucle externo (escala log), 4 métodos en los 7 modelos")
tabla(resumen_optimizadores(reg_pl[reg_pl["optimizador"].isin(NUCLEO)], "rmse", NUCLEO), dec=4)

fr_clf = friedman_optimizadores(clf_pl[clf_pl["optimizador"].isin(NUCLEO)], "auc_pr", NUCLEO)
fr_reg = friedman_optimizadores(reg_pl[reg_pl["optimizador"].isin(NUCLEO)], "rmse", NUCLEO, signo=-1)
tabla(pd.DataFrame({
    "Tarea": ["Clasificación (AUC-PR)", "Regresión (−RMSE)"],
    "Bloques": [fr_clf["n_bloques"], fr_reg["n_bloques"]],
    "χ² de Friedman": [fr_clf["chi2"], fr_reg["chi2"]], "g. l.": [fr_clf["gl"], fr_reg["gl"]],
    "Valor p": [pval(fr_clf["p"]), pval(fr_reg["p"])],
    "Rangos promedio (1 = mejor)": ["; ".join(f"{NO[o]} {num(r_, 2)}" for o, r_ in fr["rangos"].items())
                                    for fr in (fr_clf, fr_reg)]}), indice=False)

res_clf = resumen_optimizadores(clf_pl[clf_pl["optimizador"].isin(NUCLEO)], "auc_pr", NUCLEO)
mejor_rango = fr_clf["rangos"].index[0]
t_min = res_clf["Tiempo por evaluación (s)"]
interpretar(
    f"En clasificación, el AUC-PR externo promedio de los cuatro métodos va de "
    f"{num(clf_pl[clf_pl['optimizador'].isin(NUCLEO)].groupby('optimizador')['auc_pr'].mean().min(), 3)} a "
    f"{num(clf_pl[clf_pl['optimizador'].isin(NUCLEO)].groupby('optimizador')['auc_pr'].mean().max(), 3)}, y la prueba de "
    f"Friedman sobre {fr_clf['n_bloques']} bloques da χ² = {num(fr_clf['chi2'], 2)} ({p_txt(fr_clf['p'])}); el mejor "
    f"rango promedio es el de {NO[mejor_rango]} ({num(fr_clf['rangos'].iloc[0], 2)}). En regresión, Friedman da "
    f"χ² = {num(fr_reg['chi2'], 2)} ({p_txt(fr_reg['p'])}). El tiempo por evaluación va de {num(t_min.min(), 2)} s "
    f"({t_min.idxmin()}) a {num(t_min.max(), 2)} s ({t_min.idxmax()}).",
    ("Esto significa que, con el mismo presupuesto de 20 configuraciones, el método de búsqueda cambia de forma "
     "sistemática la calidad del modelo que llega al bucle externo, aunque la diferencia es pequeña frente a la que "
     "hay entre modelos (sección 7). " if fr_clf["p"] < 0.05 else
     "Esto significa que, con el mismo presupuesto de 20 configuraciones, no hay evidencia de que el método de "
     "búsqueda cambie la calidad del modelo en el bucle externo: las diferencias entre métodos son menores que la "
     "variación entre bloques. ")
    + "Las diferencias de tiempo por evaluación reflejan qué regiones del espacio visita cada método (por ejemplo, "
      "bosques con más árboles), no el costo propio del optimizador, que es despreciable frente al ajuste de los modelos.",
    conectar("Por lo anterior, conviene", "Juzgar la eficiencia no solo por el valor final sino por la rapidez con que "
             "cada método llega a buenas soluciones (curvas *anytime*, secciones 5.4 y 5.5), que es donde la teoría "
             "predice la ventaja de los métodos bayesiano y genético."))

exportar("opt_clf", resumen_optimizadores(clf_pl[clf_pl["optimizador"].isin(NUCLEO)], "auc_pr", NUCLEO).reset_index())
exportar("opt_reg", resumen_optimizadores(reg_pl[reg_pl["optimizador"].isin(NUCLEO)], "rmse", NUCLEO).reset_index())
exportar("friedman_opt", {"clf": {"chi2": fr_clf["chi2"], "p": fr_clf["p"], "n": fr_clf["n_bloques"],
                                  "rangos": {NO[k]: v for k, v in fr_clf["rangos"].items()}},
                          "reg": {"chi2": fr_reg["chi2"], "p": fr_reg["p"], "n": fr_reg["n_bloques"],
                                  "rangos": {NO[k]: v for k, v in fr_reg["rangos"].items()}}})
'''

S54_MD = r'''### 5.4 Convergencia: curvas de desempeño en cualquier momento (*anytime*)

Para cada bloque (tarea, modelo, balanceo, pliegue externo) se registra el mejor puntaje de validación interna encontrado hasta la evaluación *t*. Para promediar bloques con escalas distintas, el puntaje se normaliza dentro del bloque entre el peor y el mejor valor que encontró **cualquier** método: 1 significa haber llegado al mejor valor conocido del bloque. La curva frente al tiempo usa el tiempo acumulado relativo al método más lento del bloque.'''

S54_CODE = r'''T = cfg.PRESUPUESTO
h = historial.copy()
h["bloque"] = h["tarea"] + "|" + h["modelo"] + "|" + h["balanceo"] + "|" + h["pliegue"].astype(str)
finito = h[np.isfinite(h["puntaje"])]
ref = finito.groupby("bloque")["puntaje"].agg(ref_max="max", ref_min="min")
T_bloque = h.groupby(["bloque", "optimizador"])["t_acum"].max().groupby("bloque").max()
REJ_TAU = np.linspace(0.05, 1, 20)

curvas, curvas_t, eficiencia = {}, {}, []
for (b_, o_), g in h.groupby(["bloque", "optimizador"]):
    lo, hi = ref.loc[b_, "ref_min"], ref.loc[b_, "ref_max"]
    rango = hi - lo if hi > lo else 1.0
    mejor = g.set_index("n")["mejor"].reindex(range(1, T + 1)).ffill().replace(-np.inf, np.nan).fillna(lo)
    norm = ((mejor - lo) / rango).clip(0, 1).to_numpy()
    curvas[(b_, o_)] = norm
    tt = g["t_acum"].to_numpy() / T_bloque.loc[b_]
    mt = np.array([g["mejor"][tt <= tau].max() if (tt <= tau).any() else lo for tau in REJ_TAU])
    curvas_t[(b_, o_)] = np.clip((np.where(np.isfinite(mt), mt, lo) - lo) / rango, 0, 1)
    tarea_ = b_.split("|")[0]
    alcanza = (mejor >= 0.99 * hi) if tarea_ == "clf" else (-mejor <= 1.01 * -hi)
    eficiencia.append({"bloque": b_, "tarea": tarea_, "modelo": b_.split("|")[1], "optimizador": o_,
                       "AUAC evaluaciones": norm.mean(), "AUAC tiempo": curvas_t[(b_, o_)].mean(),
                       "Final normalizado": norm[-1],
                       "Evaluaciones al 99 %": int(np.argmax(alcanza.to_numpy())) + 1 if alcanza.any() else np.nan,
                       "Tiempo total (s)": g["t_acum"].max()})
eficiencia = pd.DataFrame(eficiencia)


def banda(ax, claves, x, opt, matriz):
    M = np.vstack([matriz[c] for c in claves])
    ax.plot(x, np.median(M, axis=0), color=COLOR_OPT[opt], label=NO[opt])
    ax.fill_between(x, np.quantile(M, 0.25, axis=0), np.quantile(M, 0.75, axis=0), color=COLOR_OPT[opt], alpha=0.12,
                    lw=0)


fig, axs = plt.subplots(1, 3, figsize=(15, 4))
bloques_clf = [b_ for b_ in ref.index if b_.startswith("clf")]
bloques_reg = [b_ for b_ in ref.index if b_.startswith("reg")]
for o_ in NUCLEO:
    banda(axs[0], [(b_, o_) for b_ in bloques_clf], np.arange(1, T + 1), o_, curvas)
    banda(axs[1], [(b_, o_) for b_ in bloques_clf], REJ_TAU, o_, curvas_t)
    banda(axs[2], [(b_, o_) for b_ in bloques_reg], np.arange(1, T + 1), o_, curvas)
axs[0].set(title="Clasificación: frente a evaluaciones (28 pares × 5 pliegues)", xlabel="Evaluación t",
           ylabel="Mejor puntaje normalizado (mediana y RIC)")
axs[1].set(title="Clasificación: frente al tiempo relativo", xlabel="Tiempo acumulado / tiempo del método más lento")
axs[2].set(title="Regresión: frente a evaluaciones (7 modelos × 5 pliegues)", xlabel="Evaluación t")
for ax in axs:
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

med = {o_: np.median(np.vstack([curvas[(b_, o_)] for b_ in bloques_clf]), axis=0) for o_ in NUCLEO}
t90 = {o_: (int(np.argmax(med[o_] >= 0.9)) + 1 if (med[o_] >= 0.9).any() else None) for o_ in NUCLEO}
interpretar(
    "En la mediana de los bloques de clasificación, el puntaje normalizado llega a 0,9 en la evaluación "
    + "; ".join(f"{t90[o_] if t90[o_] else 'nunca (en 20)'} con {NO[o_]}" for o_ in NUCLEO)
    + f". Tras 5 evaluaciones, la mediana es de {enumerar([num(med[o_][4], 2) + ' (' + NO[o_] + ')' for o_ in NUCLEO], codigo=False)}; "
    f"al final del presupuesto, de {enumerar([num(med[o_][-1], 2) + ' (' + NO[o_] + ')' for o_ in NUCLEO], codigo=False)}.",
    conectar("Hay que tener en cuenta que", "Las cinco primeras evaluaciones de Optuna son aleatorias, así que su curva "
             "solo puede separarse de la de Random Search a partir de la sexta; el genético tampoco puede mejorar antes "
             "de completar la población inicial de seis individuos. La forma de las curvas, más que su punto final, "
             "muestra cuánto presupuesto necesita cada método para llegar a una buena región."),
    conectar("De ahí que se opte por", "Cuantificar esa diferencia con el área bajo la curva *anytime* y con las "
             "evaluaciones necesarias para llegar al 99 % del mejor valor (sección 5.5), y probarla estadísticamente."))

filas_any = []
for t_, bloques_t in (("clf", bloques_clf), ("reg", bloques_reg)):
    for o_ in NUCLEO:
        for eje, x, mat in (("evaluaciones", np.arange(1, T + 1), curvas), ("tiempo", REJ_TAU, curvas_t)):
            M = np.vstack([mat[(b_, o_)] for b_ in bloques_t])
            for xi, me, q1, q3 in zip(x, np.median(M, 0), np.quantile(M, .25, 0), np.quantile(M, .75, 0)):
                filas_any.append({"grupo": t_, "optimizador": NO[o_], "eje": eje, "x": float(xi), "mediana": me,
                                  "q1": q1, "q3": q3})
exportar("anytime", pd.DataFrame(filas_any))
'''

S54B_CODE = r'''# Historia de Optuna y regiones visitadas: XGBoost sin balanceo, último pliegue externo
b_ej = "clf|xgboost|ninguno|4"
ej = h[h["bloque"] == b_ej]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for o_ in ["random", "bayes"]:
    g = ej[ej["optimizador"] == o_]
    axs[0].scatter(g["n"], g["puntaje"], s=22, color=COLOR_OPT[o_], alpha=0.7, label=f"{NO[o_]}: evaluación")
    axs[0].step(g["n"], g["mejor"], where="post", color=COLOR_OPT[o_], lw=2, label=f"{NO[o_]}: mejor hasta t")
axs[0].axvline(5.5, color=GRIS, lw=1, ls="--")
axs[0].set(title="Historia de la optimización (AUC-PR interno)", xlabel="Evaluación t", ylabel="AUC-PR interno")
axs[0].legend(fontsize=7)
for ax, o_ in zip(axs[1:], ["random", "bayes"]):
    g = ej[ej["optimizador"] == o_]
    pr = pd.DataFrame([json.loads(p_) for p_ in g["params"]])
    sc = ax.scatter(pr["modelo__learning_rate"], pr["modelo__n_estimators"], c=g["puntaje"].to_numpy(), cmap=CMAP_SEQ,
                    s=20 + 4 * g["n"].to_numpy(), edgecolor=FONDO, lw=1.5)
    ax.set_xscale("log")
    ax.set(title=f"{NO[o_]}: configuraciones visitadas", xlabel="learning_rate (log)", ylabel="n_estimators")
    fig.colorbar(sc, ax=ax, label="AUC-PR interno")
plt.tight_layout()
plt.show()
print("El tamaño del punto crece con el orden de la evaluación; la línea discontinua marca el fin de la fase aleatoria de TPE.")

exportar("optuna_ejemplo", ej[["optimizador", "n", "puntaje", "mejor", "params"]].assign(optimizador=lambda d: d["optimizador"].map(NO)))
'''

S55_MD = r'''### 5.5 Eficiencia: calidad por unidad de presupuesto

La guía pide demostrar con evidencia cuantitativa, y no solo visual, si los métodos bayesiano y genético son más eficientes que Grid y Random Search. Para cada bloque y método se calculan tres indicadores: el **área bajo la curva *anytime*** frente a evaluaciones y frente al tiempo (promedio del puntaje normalizado; 1 = llegar al mejor valor desde la primera evaluación), y las **evaluaciones necesarias para alcanzar el 99 % del mejor valor** del bloque. Las diferencias se prueban con Friedman sobre los bloques, con comparaciones de Wilcoxon por pares ajustadas por Holm y con el delta de Cliff como tamaño de efecto.'''

S55_CODE = r'''from scipy.stats import wilcoxon

ef = eficiencia[eficiencia["optimizador"].isin(NUCLEO)]
for tarea_, nombre in (("clf", "Clasificación"), ("reg", "Regresión")):
    e_ = ef[ef["tarea"] == tarea_]
    res = e_.groupby("optimizador").agg(**{
        "AUAC evaluaciones": ("AUAC evaluaciones", "mean"), "AUAC tiempo": ("AUAC tiempo", "mean"),
        "Final normalizado": ("Final normalizado", "mean"),
        "% de bloques que llega al 99 %": ("Evaluaciones al 99 %", lambda s: 100 * s.notna().mean()),
        "Evaluaciones al 99 % (mediana)": ("Evaluaciones al 99 %", "median")}).loc[NUCLEO]
    res.index = [NO[o] for o in res.index]
    print(nombre)
    tabla(res, dec=3)

e_clf = ef[ef["tarea"] == "clf"]
mat_auac = e_clf.pivot(index="bloque", columns="optimizador", values="AUAC evaluaciones")[NUCLEO]
mat_auac_t = e_clf.pivot(index="bloque", columns="optimizador", values="AUAC tiempo")[NUCLEO]
fr_auac, fr_auac_t = st.friedman(mat_auac), st.friedman(mat_auac_t)
pares = [("bayes", "grid"), ("bayes", "random"), ("genetico", "grid"), ("genetico", "random"), ("bayes", "genetico"),
         ("random", "grid")]
filas = []
for a_, b2 in pares:
    for nombre, M in (("evaluaciones", mat_auac), ("tiempo", mat_auac_t)):
        w = wilcoxon(M[a_], M[b2])
        filas.append({"Comparación": f"{NO[a_]} vs {NO[b2]}", "AUAC frente a": nombre,
                      "Diferencia media": (M[a_] - M[b2]).mean(), "p crudo": w.pvalue,
                      "delta de Cliff": st.delta_cliff(M[a_], M[b2])})
comp_ef = pd.DataFrame(filas)
comp_ef["p Holm"] = np.nan
for nombre in ("evaluaciones", "tiempo"):
    m_ = comp_ef["AUAC frente a"] == nombre
    comp_ef.loc[m_, "p Holm"] = st.ajustar_p(comp_ef.loc[m_, "p crudo"], "holm")
comp_ef["Magnitud"] = comp_ef["delta de Cliff"].apply(st.magnitud_cliff)
print(f"Friedman sobre el AUAC (clasificación, {fr_auac['n_bloques']} bloques): frente a evaluaciones "
      f"χ² = {num(fr_auac['chi2'], 2)}, {p_txt(fr_auac['p'])}; frente al tiempo χ² = {num(fr_auac_t['chi2'], 2)}, "
      f"{p_txt(fr_auac_t['p'])}")
tabla(comp_ef.assign(**{"p crudo": comp_ef["p crudo"].map(pval), "p Holm": comp_ef["p Holm"].map(pval)}), dec=3,
      indice=False)

auac_m = mat_auac.mean()
sig = comp_ef[(comp_ef["AUAC frente a"] == "evaluaciones") & (comp_ef["p Holm"] < 0.05)]
b_vs = comp_ef[(comp_ef["AUAC frente a"] == "evaluaciones") & comp_ef["Comparación"].str.startswith(NO["bayes"])]
g_vs = comp_ef[(comp_ef["AUAC frente a"] == "evaluaciones") & comp_ef["Comparación"].str.startswith(NO["genetico"])]
interpretar(
    f"El área bajo la curva *anytime* frente a evaluaciones es de "
    f"{enumerar([num(auac_m[o_], 3) + ' (' + NO[o_] + ')' for o_ in NUCLEO], codigo=False)}; Friedman da "
    f"{p_txt(fr_auac['p'])} frente a evaluaciones y {p_txt(fr_auac_t['p'])} frente al tiempo. Frente a Grid y Random "
    f"Search, el delta de Cliff de la optimización bayesiana es de "
    f"{enumerar([num(d_, 2) for d_ in b_vs['delta de Cliff'].iloc[:2]], codigo=False)} y el del genético de "
    f"{enumerar([num(d_, 2) for d_ in g_vs['delta de Cliff'].iloc[:2]], codigo=False)}; "
    f"{len(sig)} de las {len(pares)} comparaciones por pares son significativas tras el ajuste de Holm.",
    conectar("Esto quiere decir que", (
        f"Frente a Grid Search la ventaja de los métodos adaptativos es clara (efecto {st.magnitud_cliff(b_vs['delta de Cliff'].iloc[0])} "
        f"en la bayesiana y {st.magnitud_cliff(g_vs['delta de Cliff'].iloc[0])} en el genético, según el delta de Cliff): la "
        "rejilla gasta las primeras evaluaciones en combinaciones fijas, a menudo en regiones malas, mientras que los "
        "métodos adaptativos usan las evaluaciones anteriores para concentrar el presupuesto. "
        if (b_vs["Diferencia media"].iloc[0] > 0 and g_vs["Diferencia media"].iloc[0] > 0) else
        "Frente a Grid Search no hay una ventaja uniforme de los métodos adaptativos. ")
        + (f"Frente a Random Search, en cambio, el efecto es {st.magnitud_cliff(b_vs['delta de Cliff'].iloc[1])} en la "
           f"bayesiana y {st.magnitud_cliff(g_vs['delta de Cliff'].iloc[1])} en el genético: la fase inicial aleatoria (5 "
           "evaluaciones en Optuna, 6 en el genético) consume una parte grande de un presupuesto de 20, y en espacios de "
           "dos a cuatro dimensiones el muestreo aleatorio ya cubre bien las regiones relevantes. La magnitud del efecto, "
           "y no solo el valor p, es la que indica si la diferencia importa en la práctica.")),
    conectar("Así las cosas, se decide", "Interpretar la eficiencia como un resultado condicionado al presupuesto y a la "
             "dimensión del espacio: con espacios de mayor dimensión o presupuestos mayores, la teoría y la literatura "
             "predicen que la ventaja de TPE crece (Bergstra et al., 2011), algo que este diseño no alcanza a medir."))

exportar("eficiencia", pd.concat([ef[ef["tarea"] == t_].groupby("optimizador")[["AUAC evaluaciones", "AUAC tiempo",
         "Final normalizado"]].mean().assign(tarea=t_).rename(index=NO) for t_ in ("clf", "reg")]).reset_index())
exportar("eficiencia_pruebas", comp_ef)
'''

S56_MD = r'''### 5.6 Algoritmo genético: diversidad y convergencia prematura

Una población que pierde diversidad antes de encontrar buenas soluciones converge de forma prematura: el cruce entre individuos casi iguales ya no explora. La diversidad se mide como la distancia euclídea media entre los genes de los individuos, normalizada por √d, en cada generación de todas las corridas del genético.'''

S56_CODE = r'''gen = pd.DataFrame([{**{k: c[k] for k in ("tarea", "modelo", "balanceo", "pliegue")}, **g_}
                    for c in corridas if c["optimizador"] == "genetico" and c["seed"] == SEED
                    for g_ in c["info"]["generaciones"]])
div = gen.groupby("generacion").agg(corridas=("diversidad", "size"), diversidad_mediana=("diversidad", "median"),
                                    q25=("diversidad", lambda s: s.quantile(0.25)),
                                    q75=("diversidad", lambda s: s.quantile(0.75)),
                                    pct_baja=("diversidad", lambda s: 100 * (s < 0.1).mean()),
                                    n_eval=("n_eval", "median"))
div = div[div["corridas"] >= 10]
tabla(div.rename(columns={"corridas": "Corridas", "diversidad_mediana": "Diversidad (mediana)", "q25": "Q1", "q75": "Q3",
                          "pct_baja": "% con diversidad < 0,1", "n_eval": "Evaluaciones acumuladas (mediana)"}), dec=3)

fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
axs[0].plot(div.index, div["diversidad_mediana"], color=CAT[3], marker="o")
axs[0].fill_between(div.index, div["q25"], div["q75"], color=CAT[3], alpha=0.15, lw=0)
axs[0].set(title="Diversidad genética por generación (mediana y RIC)", xlabel="Generación",
           ylabel="Distancia media / √d")
gen["brecha"] = gen["mejor"] - gen["media"]
br = gen[gen["generacion"].isin(div.index)].groupby(["tarea", "generacion"])["brecha"].median().unstack(0)
for t_, col in zip(br.columns, [AZUL, NARANJA]):
    axs[1].plot(br.index, br[t_], marker="o", color=col, label="Clasificación" if t_ == "clf" else "Regresión")
axs[1].set(title="Brecha entre el mejor y la media de la población", xlabel="Generación", ylabel="Mejor − media")
axs[1].legend()
plt.tight_layout()
plt.show()

d0, dU = div["diversidad_mediana"].iloc[0], div["diversidad_mediana"].iloc[-1]
n_gen = gen.groupby(["tarea", "modelo", "balanceo", "pliegue"])["generacion"].max()
interpretar(
    f"La diversidad mediana pasa de {num(d0, 3)} en la población inicial a {num(dU, 3)} en la generación "
    f"{div.index[-1]} (una caída del {pct(1 - dU / d0, 0)}), y el porcentaje de corridas con diversidad menor que 0,1 "
    f"sube del {num(div['pct_baja'].iloc[0], 1)} % al {num(div['pct_baja'].iloc[-1], 1)} %. La población inicial de "
    f"un genoma uniforme tiene una diversidad esperada cercana a 0,41. Cada corrida usó una mediana de "
    f"{num(n_gen.median())} generaciones (entre {num(n_gen.min())} y {num(n_gen.max())}).",
    conectar("Lo anterior indica que", ("La población se concentra rápido, como se espera con elitismo y torneo de "
             "tamaño 3: el genético pasa casi de inmediato a explotar la región del mejor individuo y, cuando la "
             "población converge, las generaciones siguientes producen sobre todo clones ya evaluados (por eso hay "
             "corridas con muchas generaciones y pocas evaluaciones nuevas). Es una señal de convergencia prematura, "
             "con riesgo de quedarse en un óptimo local."
             if dU / d0 < 0.5 else "La población conserva buena parte de su diversidad en las pocas generaciones que "
             "permite el presupuesto, así que el genético sigue explorando y no hay señales de convergencia prematura; "
             "el costo es que tampoco alcanza a explotar a fondo la mejor región.")),
    conectar("Por esta razón, lo indicado es", "Leer el buen desempeño del genético en las curvas *anytime* (sección "
             "5.5) como el de una búsqueda que explota pronto una región buena, no como una exploración completa del "
             "espacio. Con más presupuesto convendría una población mayor, una mutación adaptativa que reinyecte "
             "diversidad cuando caiga por debajo de un umbral o reemplazar los clones por individuos aleatorios."))

exportar("genetico_diversidad", div.reset_index())
'''

S57_MD = r'''### 5.7 Multi-fidelidad: Successive Halving

Successive Halving se compara con Random Search y con Optuna en las mismas combinaciones (Random Forest y XGBoost sin balanceo) y en los mismos pliegues externos. Su costo se expresa en **evaluaciones equivalentes**: suma de árboles entrenados en todas las configuraciones dividida por el número máximo de árboles.'''

S57_CODE = r'''hv = pd.DataFrame([{k: c[k] for k in ("modelo", "pliegue", "auc_pr", "auc", "t_opt", "n_eval", "eval_equivalentes")}
                   for c in corridas if c["optimizador"] == "halving"])
ref_hv = pliegues[(pliegues["tarea"] == "clf") & (pliegues["balanceo"] == "ninguno") &
                  pliegues["modelo"].isin(["bosque", "xgboost"]) & pliegues["optimizador"].isin(["random", "bayes"])]
filas = []
for m_ in ("bosque", "xgboost"):
    for o_, d_ in [("halving", hv[hv["modelo"] == m_])] + [(o_, ref_hv[(ref_hv["modelo"] == m_) &
                                                                       (ref_hv["optimizador"] == o_)])
                                                           for o_ in ("random", "bayes")]:
        filas.append({"Modelo": NM[m_], "Método": NO[o_], "AUC-PR externo": ms(d_["auc_pr"]),
                      "AUC externo": ms(d_["auc"]), "Configuraciones evaluadas por pliegue": d_["n_eval"].mean(),
                      "Evaluaciones equivalentes por pliegue": d_["eval_equivalentes"].mean() if o_ == "halving"
                      else d_["n_eval"].mean(), "Tiempo por pliegue (s)": d_["t_opt"].mean()})
tabla(pd.DataFrame(filas), dec=1, indice=False)

rondas_hv = pd.DataFrame([{**r_, "modelo": c["modelo"]} for c in corridas if c["optimizador"] == "halving"
                          for r_ in c["rondas"]])
tabla(rondas_hv.groupby(["modelo", "iter"]).agg(Candidatos=("candidatos", "first"), Árboles=("arboles", "first"),
                                                **{"Mejor AUC-PR interno (media)": ("mejor", "mean")}), dec=4)

t_hv = hv.groupby("modelo")["t_opt"].mean()
t_rs = ref_hv[ref_hv["optimizador"] == "random"].groupby("modelo")["t_opt"].mean()
ap_hv = hv.groupby("modelo")["auc_pr"].mean()
ap_rs = ref_hv[ref_hv["optimizador"] == "random"].groupby("modelo")["auc_pr"].mean()
interpretar(
    f"Successive Halving evalúa {int(hv['n_eval'].mean())} configuraciones por pliegue con un costo equivalente a "
    f"{num(hv['eval_equivalentes'].mean(), 1)} evaluaciones completas, frente a las 20 de Random Search. Su tiempo por "
    f"pliegue es {num(t_hv['xgboost'], 0)} s en XGBoost y {num(t_hv['bosque'], 0)} s en Random Forest, frente a "
    f"{num(t_rs['xgboost'], 0)} s y {num(t_rs['bosque'], 0)} s de Random Search; el AUC-PR externo cambia en "
    f"{num(ap_hv['xgboost'] - ap_rs['xgboost'], 3)} y {num(ap_hv['bosque'] - ap_rs['bosque'], 3)}, respectivamente.",
    "Es decir, con " + pct(hv["eval_equivalentes"].mean() / cfg.PRESUPUESTO, 0) + " de las evaluaciones "
    "equivalentes de Random Search, la eliminación sucesiva examina el doble de configuraciones porque descarta "
    "temprano las que rinden mal con pocos árboles. El ahorro de tiempo (" + pct(1 - t_hv["xgboost"] / t_rs["xgboost"], 0)
    + " en XGBoost y " + pct(1 - t_hv["bosque"] / t_rs["bosque"], 0) + " en Random Forest) es menor que el de árboles, "
    "porque cada evaluación tiene costos que no dependen del número de árboles (preprocesamiento y tres pliegues "
    "internos). La condición para que funcione es que el orden de las configuraciones con pocos árboles se parezca al "
    "orden con muchos; la tabla de rondas muestra si el mejor puntaje sube o baja al aumentar la fidelidad.",
    conectar("Por ello, se recomienda", "Usar la multi-fidelidad cuando el costo por evaluación domina (bosques y "
             "*boosting* con muchos árboles) y reasignar el presupuesto ahorrado a más configuraciones, tal como pide la "
             "guía; en modelos baratos como Naive Bayes o la logística la ganancia no compensa la complejidad."))

exportar("halving", pd.DataFrame(filas))
'''

S58_MD = r'''### 5.8 Estabilidad de las soluciones ante distintas semillas

Los métodos estocásticos (Random Search, Optuna y el genético) se repiten con tres semillas adicionales (1, 2 y 3) en XGBoost sin balanceo, en los cinco pliegues externos. Grid Search es determinista dado el espacio, así que su variación por semilla del optimizador es nula.'''

S58_CODE = r'''sem = pd.DataFrame([{k: c[k] for k in ("optimizador", "pliegue", "seed", "auc_pr", "puntaje_interno")}
                    | {"lr": c["params"]["modelo__learning_rate"], "arboles": c["params"]["modelo__n_estimators"]}
                    for c in corridas if c["tarea"] == "clf" and c["modelo"] == "xgboost" and c["balanceo"] == "ninguno"
                    and c["optimizador"] in ("random", "bayes", "genetico")])
por_sem = sem.groupby(["optimizador", "seed"])["auc_pr"].mean().unstack()
estab = pd.DataFrame({
    "AUC-PR externo por semilla (media de 5 pliegues)": por_sem.apply(lambda r_: "; ".join(num(v, 4) for v in r_), axis=1),
    "Desv. est. entre semillas": por_sem.std(axis=1, ddof=1),
    "Desv. est. entre pliegues (semilla 42)": sem[sem["seed"] == SEED].groupby("optimizador")["auc_pr"].std(),
    "Desv. est. de log10(learning_rate) entre semillas (media por pliegue)":
        sem.assign(llr=np.log10(sem["lr"])).groupby(["optimizador", "pliegue"])["llr"].std().groupby("optimizador").mean(),
    "Desv. est. del puntaje interno entre semillas":
        sem.groupby(["optimizador", "pliegue"])["puntaje_interno"].std().groupby("optimizador").mean()})
estab.index = [NO[o] for o in estab.index]
tabla(estab, dec=4)

r_ = estab["Desv. est. entre semillas"] / estab["Desv. est. entre pliegues (semilla 42)"]
interpretar(
    f"La desviación del AUC-PR externo entre semillas es de "
    f"{enumerar([num(v, 4) + ' (' + i + ')' for i, v in estab['Desv. est. entre semillas'].items()], codigo=False)}, "
    f"entre {num(r_.min(), 2)} y {num(r_.max(), 2)} veces la desviación entre pliegues. La tasa de aprendizaje elegida "
    f"varía en promedio {enumerar([num(v, 2) for v in estab.iloc[:, 3]], codigo=False)} órdenes de magnitud (log10) "
    f"entre semillas.",
    conectar("Esto significa que", "Las soluciones de los métodos estocásticos no son únicas: semillas distintas llevan a "
             "hiperparámetros diferentes con desempeños parecidos, lo que indica una superficie de respuesta plana "
             "cerca del óptimo. " + ("La variación por semilla es pequeña frente a la que hay entre periodos, así que no "
             "cambia las conclusiones." if r_.max() < 0.5 else "La variación por semilla no es despreciable frente a la "
             "que hay entre periodos, así que una sola corrida puede ordenar mal dos métodos cercanos.")),
    conectar("Por tanto, corresponde", "Reportar los hiperparámetros elegidos como una de varias soluciones equivalentes y "
             "no sobreinterpretar su valor exacto; para la comparación entre métodos, la prueba por bloques de la "
             "sección 5.5 ya promedia esta fuente de variación."))

exportar("semillas_optimizador", estab.reset_index())
'''
