"""Secciones 8 (comparación estadística), 9 (conclusiones) y 10 (referencias)."""

S81_MD = r'''## 8. Comparación estadística

Con 140 combinaciones hay más de 9 000 pares posibles; probarlos uno a uno con α = 0,05 produciría cientos de falsos positivos. Por eso la comparación sigue una secuencia jerárquica: **clasificación**, Friedman → Nemenyi y diagrama de diferencia crítica → DeLong dirigido con corrección por comparaciones múltiples → tamaño del efecto; **regresión**, Model Confidence Set → Giacomini-White y Clark-West → Diebold-Mariano con corrección HLN y *bootstrap* estacionario → tamaño del efecto.

### 8.1 Clasificación: prueba ómnibus de Friedman, Nemenyi y diagrama de diferencia crítica

Los bloques son los cinco pliegues del bucle externo y la métrica es el AUC-PR. Primero se prueban todas las combinaciones de clasificación y después los siete finalistas (el mejor de cada modelo). Como cinco bloques dan poca potencia, se agrega un análisis de sensibilidad con bloques mensuales: el AUC-PR de las predicciones fuera de muestra de cada finalista en cada mes de validación externa.'''

S81_CODE = r'''mat_all = clf_pl.pivot_table(index="pliegue", columns=["modelo", "balanceo", "optimizador"], values="auc_pr")
fr_all = st.friedman(mat_all)
fin_ext = evm.elegir_finalistas(maestra, "clf")
claves_fin = [(r_.modelo, r_.balanceo, r_.optimizador) for r_ in fin_ext.itertuples()]
mat_fin = mat_all[claves_fin]
mat_fin.columns = [NM[c[0]] for c in claves_fin]
fr_fin = st.friedman(mat_fin)

# Sensibilidad: bloques = meses de validación externa (predicciones fuera de muestra)
o_c = oof[oof["tarea"] == "clf"].copy()
o_c["periodo"] = tiempo.loc[o_c["indice"], "periodo"].to_numpy()
filas = {}
for (m_, b_, op_), nombre in zip(claves_fin, mat_fin.columns):
    d_ = o_c[(o_c["modelo"] == m_) & (o_c["balanceo"] == b_) & (o_c["optimizador"] == op_)]
    filas[nombre] = d_.groupby("periodo").apply(
        lambda g: average_precision_score(g["y"], g["puntaje"]) if 0 < g["y"].sum() < len(g) and len(g) >= 100 else np.nan,
        include_groups=False)
mat_mes = pd.DataFrame(filas).dropna()
fr_mes = st.friedman(mat_mes)

omni = pd.DataFrame([
    ("Todas las combinaciones de clasificación", fr_all), ("Finalistas (bloques = pliegues externos)", fr_fin),
    ("Finalistas (bloques = meses de validación)", fr_mes)], columns=["Comparación", "r"])
omni = pd.DataFrame({"Comparación": omni["Comparación"], "Modelos (k)": [r_["k_modelos"] for r_ in omni["r"]],
                     "Bloques (N)": [r_["n_bloques"] for r_ in omni["r"]], "χ²_F": [r_["chi2"] for r_ in omni["r"]],
                     "g. l.": [r_["gl"] for r_ in omni["r"]], "p (χ²)": [pval(r_["p"]) for r_ in omni["r"]],
                     "F de Iman-Davenport": [r_["F_ID"] for r_ in omni["r"]],
                     "p (F)": [pval(r_["p_F_ID"]) for r_ in omni["r"]],
                     "Diferencia crítica de Nemenyi": [st.diferencia_critica(r_["k_modelos"], r_["n_bloques"])
                                                       for r_ in omni["r"]],
                     "Conclusión": ["Se rechaza H0" if r_["p"] < 0.05 else "No se rechaza H0" for r_ in omni["r"]]})
tabla(omni, dec=3, indice=False)

fig, axs = plt.subplots(2, 1, figsize=(10, 5.6))
for ax, M, fr, titulo in ((axs[0], mat_fin, fr_fin, "Bloques = pliegues externos"),
                          (axs[1], mat_mes, fr_mes, "Bloques = meses de validación externa")):
    nem = sp.posthoc_nemenyi_friedman(M.to_numpy())
    nem.index = nem.columns = M.columns
    plt.sca(ax)
    sp.critical_difference_diagram(fr["rangos"], nem, ax=ax, label_fmt_left="{label} ({rank:.2f})  ",
                                   label_fmt_right="  ({rank:.2f}) {label}")
    ax.set_title(f"Diagrama de diferencia crítica (Nemenyi, α = 0,05) · {titulo} · CD = "
                 f"{num(st.diferencia_critica(fr['k_modelos'], fr['n_bloques']), 2)}", fontsize=9.5)
plt.tight_layout()
plt.show()

interpretar(
    f"Sobre todas las combinaciones, Friedman da χ² = {num(fr_all['chi2'], 1)} con {fr_all['gl']} g. l. "
    f"({p_txt(fr_all['p'])}); entre los siete finalistas con bloques por pliegue, χ² = {num(fr_fin['chi2'], 2)} "
    f"({p_txt(fr_fin['p'])}) y diferencia crítica de {num(st.diferencia_critica(7, fr_fin['n_bloques']), 2)} rangos; con "
    f"{fr_mes['n_bloques']} bloques mensuales, χ² = {num(fr_mes['chi2'], 2)} ({p_txt(fr_mes['p'])}) y diferencia crítica "
    f"de {num(st.diferencia_critica(7, fr_mes['n_bloques']), 2)}. El mejor rango promedio entre finalistas es el de "
    f"{fr_fin['rangos'].index[0]} ({num(fr_fin['rangos'].iloc[0], 2)}) por pliegues y el de {fr_mes['rangos'].index[0]} "
    f"({num(fr_mes['rangos'].iloc[0], 2)}) por meses.",
    conectar("Esto significa que", (f"Hay evidencia de que al menos una de las {fr_all['gl'] + 1} combinaciones difiere de las demás "
             "(sobre todo por las peores, con balanceos o espacios que fallan en algunos pliegues)"
             if fr_all["p"] < 0.05 else "No hay evidencia de que alguna combinación difiera de las demás")
             + (", pero entre los siete finalistas no hay evidencia de diferencias, ni con bloques por pliegue ni con "
                "bloques mensuales: todos quedan dentro de la diferencia crítica de Nemenyi y son estadísticamente "
                "equivalentes. "
                if fr_fin["p"] >= 0.05 and fr_mes["p"] >= 0.05 else
                "; entre los finalistas, el diagrama muestra qué pares se separan más que la diferencia crítica. ")
             + "Con pocos bloques la diferencia crítica es amplia, así que la equivalencia también refleja la poca "
             "potencia de la prueba, no solo la igualdad real de los modelos."),
    conectar("Por tanto, corresponde", "Comparar de forma dirigida solo los finalistas que encabezan el ranking, con "
             "DeLong sobre la prueba y corrección por comparaciones múltiples (sección 8.2), y no declarar un ganador a "
             "partir de diferencias que el diagrama muestra como no significativas."))

exportar("friedman_omnibus", omni)
exportar("rangos_cd", pd.concat([fr_fin["rangos"].rename("rango").rename_axis("modelo").reset_index().assign(bloques="pliegues",
          cd=st.diferencia_critica(fr_fin["k_modelos"], fr_fin["n_bloques"])),
          fr_mes["rangos"].rename("rango").rename_axis("modelo").reset_index().assign(bloques="meses",
          cd=st.diferencia_critica(fr_mes["k_modelos"], fr_mes["n_bloques"]))]))
'''

S82_MD = r'''### 8.2 Clasificación: comparaciones dirigidas con DeLong, *bootstrap* y tamaño del efecto

Se comparan en la prueba los tres finalistas con mayor AUC entre sí y el mejor contra el modelo base del Entregable 2. DeLong prueba la igualdad de dos AUC correlacionadas (mismas observaciones); los valores p se ajustan con Holm-Bonferroni y con Benjamini-Hochberg, y la diferencia de AUC lleva un intervalo *bootstrap* BCa pareado. El delta de Cliff sobre el AUC-PR de los cinco pliegues externos mide el tamaño del efecto en la validación anidada.'''

S82_CODE = r'''top3 = res_te[~res_te["clave"].str.endswith("e2")].sort_values("AUC", ascending=False)["clave"].iloc[:3].tolist()
pares_d = [(top3[0], top3[1]), (top3[0], top3[2]), (top3[1], top3[2]), (top3[0], "clf_referencia_e2")]
nombre_k = lambda k: "Referencia E2" if k.endswith("e2") else NM[fin_clf[k]["modelo"]]
ap_pl = {NM[c[0]]: mat_fin[NM[c[0]]].to_numpy() for c in claves_fin}
filas = []
for a_, b_ in pares_d:
    sa, sb = fin_clf[a_]["puntaje_te"], fin_clf[b_]["puntaje_te"]
    dl = st.delong(yte, sa, sb)
    ic = st.ic_bootstrap(lambda y_, s1, s2: roc_auc_score(y_, s1) - roc_auc_score(y_, s2), yte, sa, sb, n_rep=1000)
    cl = st.delta_cliff(ap_pl[nombre_k(a_)], ap_pl[nombre_k(b_)]) if not b_.endswith("e2") else np.nan
    filas.append({"Modelo A": nombre_k(a_), "Modelo B": nombre_k(b_), "AUC A": dl["auc_a"], "AUC B": dl["auc_b"],
                  "Diferencia de AUC": dl["diferencia"], "IC 95 % BCa": f"[{num(ic[0], 4)}; {num(ic[1], 4)}]",
                  "z": dl["z"], "p crudo": dl["p"], "delta de Cliff (AUC-PR externo)": cl})
dl_t = pd.DataFrame(filas)
dl_t["p Holm"] = st.ajustar_p(dl_t["p crudo"], "holm")
dl_t["p BH"] = st.ajustar_p(dl_t["p crudo"], "fdr_bh")
dl_t["Magnitud (Cliff)"] = dl_t["delta de Cliff (AUC-PR externo)"].map(lambda d: st.magnitud_cliff(d) if np.isfinite(d) else "—")
tabla(dl_t.assign(**{c: dl_t[c].map(pval) for c in ("p crudo", "p Holm", "p BH")}), dec=4, indice=False)

r0 = dl_t.iloc[0]
rE = dl_t.iloc[-1]
interpretar(
    f"Entre {r0['Modelo A']} y {r0['Modelo B']}, la diferencia de AUC en prueba es de {num(r0['Diferencia de AUC'], 4)} "
    f"{r0['IC 95 % BCa']} (p de Holm {pval(r0['p Holm'])}); frente a la referencia del Entregable 2, {rE['Modelo A']} "
    f"difiere en {num(rE['Diferencia de AUC'], 4)} {rE['IC 95 % BCa']} (p de Holm {pval(rE['p Holm'])}). "
    f"{int((dl_t['p Holm'] < 0.05).sum())} de las {len(dl_t)} comparaciones son significativas tras el ajuste.",
    (("Hay que tener en cuenta que, aun con más de 8 000 observaciones de prueba, ninguna diferencia de AUC es "
      "significativa tras el ajuste: los mejores finalistas y el modelo base del Entregable 2 son estadísticamente "
      "equivalentes en su capacidad de ordenar a los clientes, y las diferencias de pocas milésimas no cambian la "
      "lista de clientes a llamar. El intervalo de la diferencia y el delta de Cliff confirman que el efecto es pequeño."
      if (dl_t["p Holm"] >= 0.05).all() else
      "Hay que tener en cuenta que, con más de 8 000 observaciones de prueba, DeLong detecta diferencias de AUC muy "
      "pequeñas; por eso el intervalo de la diferencia y el delta de Cliff importan tanto como el valor p.")),
    conectar("Con base en esto, se propone", "Elegir entre los modelos estadísticamente equivalentes por criterios "
             "secundarios: costo de entrenamiento, calibración e interpretabilidad. Con esos criterios, la logística "
             "regularizada del Entregable 2 sigue siendo una opción muy competitiva, porque es la más barata y la más "
             "fácil de explicar, y su recalibración mensual solo desplaza el intercepto sin cambiar el orden."))

exportar("delong", dl_t)
'''

S83_MD = r'''### 8.3 Regresión: Model Confidence Set, Giacomini-White, Clark-West y Diebold-Mariano

El **Model Confidence Set** (Hansen, Lunde y Nason, 2011) parte de las 28 combinaciones de regresión más la media histórica y elimina de forma iterativa, con *bootstrap* estacionario, los modelos significativamente peores, hasta dejar un conjunto de modelos indistinguibles del mejor con confianza del 90 %. Las pérdidas son los errores cuadráticos de las predicciones fuera de muestra del bucle externo, en orden temporal. Sobre los finalistas se aplican, en la prueba:

- **Giacomini-White** (condicional, instrumentos 1 y la diferencia de pérdida rezagada) entre el mejor y el segundo, que no están anidados.
- **Clark-West** entre la media histórica y el mejor modelo: la media es un caso particular de cualquier modelo con intercepto (modelos anidados), donde Diebold-Mariano está sesgado.
- **Diebold-Mariano con corrección HLN** entre el mejor y el segundo, con el intervalo de la diferencia media de pérdida por *bootstrap* estacionario (Politis y Romano, 1994) y la d de Cohen.
- **SPA de Hansen** (2005): ¿algún finalista supera a la media histórica, controlando el *data snooping*?'''

S83_CODE = r'''o_r = oof[oof["tarea"] == "reg"].copy()
o_r["etiqueta"] = o_r["modelo"].map(NM) + " · " + o_r["optimizador"].map(NO)
L = (o_r.assign(e2=(o_r["y"] - o_r["pred"]) ** 2)
     .pivot_table(index="indice", columns="etiqueta", values="e2").sort_index())
bench = pd.Series(np.nan, index=X_tr.index)
for i_e, i_v in cv_externa(GAP).split(X_tr):
    bench.iloc[i_v] = rtr[i_e].mean()
L["Media histórica (referencia)"] = ((part.r_tr - bench) ** 2).loc[L.index]
mcs = st.model_confidence_set(L, alfa=0.10, n_rep=1000)
tabla(mcs.rename(columns={"perdida_media": "Pérdida media (MSE)", "rango_promedio": "Rango promedio de pérdida",
                          "p_MCS": "p MCS", "en_conjunto": "¿En el conjunto de confianza?"})
      .assign(**{"¿En el conjunto de confianza?": lambda d: d["¿En el conjunto de confianza?"].map({True: "Sí", False: "No"})}),
      dec=4)

orden_te = res_reg[res_reg["clave"] != "media"].sort_values("RMSE (log)")["clave"].tolist()
k1, k2 = orden_te[0], orden_te[1]
p1, p2 = fin_reg[k1]["pred_te"], fin_reg[k2]["pred_te"]
pm = np.full_like(rte, media_tr)
e1, e2, em = rte - p1, rte - p2, rte - pm
gw = st.giacomini_white(e2 ** 2, e1 ** 2)
cw = st.clark_west(rte, pm, p1)
dm = st.diebold_mariano(e2, e1)
dm_b = st.diebold_mariano(em, e1)
bs = st.ic_bootstrap_estacionario(e2 ** 2 - e1 ** 2)
spa_p = st.spa(em ** 2, pd.DataFrame({k: (rte - fin_reg[k]["pred_te"]) ** 2 for k in orden_te}))
dirigidas = pd.DataFrame([
    ("Giacomini-White condicional", f"{NM[fin_reg[k2]['modelo']]} vs {NM[fin_reg[k1]['modelo']]}", gw["GW"], gw["p"],
     gw["dif_media"]),
    ("Clark-West (anidados)", f"Media histórica ⊂ {NM[fin_reg[k1]['modelo']]}", cw["CW"], cw["p"], np.mean(em ** 2 - e1 ** 2)),
    ("Diebold-Mariano (HLN)", f"{NM[fin_reg[k2]['modelo']]} vs {NM[fin_reg[k1]['modelo']]}", dm["DM_HLN"], dm["p"],
     dm["dif_perdida"]),
    ("Diebold-Mariano (HLN)", f"Media histórica vs {NM[fin_reg[k1]['modelo']]}", dm_b["DM_HLN"], dm_b["p"],
     dm_b["dif_perdida"]),
], columns=["Prueba", "Comparación (pérdida 1 − pérdida 2)", "Estadístico", "p crudo", "Diferencia media de pérdida"])
dirigidas["p Holm"] = st.ajustar_p(dirigidas["p crudo"], "holm")
dirigidas["d de Cohen"] = [st.d_cohen(e2 ** 2 - e1 ** 2), st.d_cohen(em ** 2 - e1 ** 2), st.d_cohen(e2 ** 2 - e1 ** 2),
                           st.d_cohen(em ** 2 - e1 ** 2)]
tabla(dirigidas.assign(**{c: dirigidas[c].map(pval) for c in ("p crudo", "p Holm")}), dec=4, indice=False)
print(f"Bootstrap estacionario de la diferencia media de pérdida ({NM[fin_reg[k2]['modelo']]} − "
      f"{NM[fin_reg[k1]['modelo']]}): {num(bs['media'], 5)} [IC 95 %: {num(bs['ic_inf'], 5)}; {num(bs['ic_sup'], 5)}], "
      f"bloque medio {num(bs['bloque'], 1)}")
print(f"SPA de Hansen (referencia = media histórica): p = {num(spa_p['consistent'], 3)} (consistente), "
      f"{num(spa_p['lower'], 3)} (inferior), {num(spa_p['upper'], 3)} (superior)")

en_mcs = mcs[mcs["en_conjunto"]]
interpretar(
    f"El Model Confidence Set al 90 % conserva {len(en_mcs)} de los {len(mcs)} modelos"
    + (", incluida la media histórica" if "Media histórica (referencia)" in en_mcs.index else ", y excluye la media histórica")
    + f"; la menor pérdida media es la de {mcs.index[0]} ({num(mcs['perdida_media'].iloc[0], 4)}). En prueba, "
    f"Clark-West contra la media da {p_txt(cw['p'])}, Giacomini-White entre los dos mejores finalistas "
    f"{p_txt(gw['p'])}, Diebold-Mariano entre ellos {p_txt(dm['p'])} (p de Holm {pval(dirigidas['p Holm'].iloc[2])}, "
    f"d de Cohen {num(dirigidas['d de Cohen'].iloc[2], 3)}) y la SPA {p_txt(spa_p['consistent'])}.",
    conectar("En síntesis,", ("Hay modelos que superan de forma estadísticamente significativa a la media histórica, "
             "pero el tamaño del efecto es pequeño: la mejora en el error cuadrático es una fracción mínima de la "
             "variabilidad de la duración."
             if (cw["p"] < 0.05 or spa_p["consistent"] < 0.05) else
             "No hay evidencia de que algún modelo pronostique la duración mejor que la media histórica: el MCS no "
             "los distingue y las pruebas dirigidas no rechazan la igualdad de capacidad predictiva.")
             + (" Entre los dos mejores finalistas, Giacomini-White indica que su ventaja relativa depende del "
                "estado reciente (la diferencia de pérdida rezagada predice la siguiente)." if gw["p"] < 0.05 else
                " Entre los dos mejores finalistas, Giacomini-White no encuentra que la ventaja relativa dependa del "
                "estado reciente.")),
    conectar("Así las cosas, se decide", ("Concluir que los modelos sí superan a la media histórica de forma "
             "significativa, pero con una ganancia demasiado pequeña (R² menor que 0,1) para planificar la carga de los "
             "asesores llamada a llamada; como estimación del nivel medio, la media histórica es casi igual de útil y "
             "mucho más simple." if "Media histórica (referencia)" not in en_mcs.index else
             "Concluir que la rama de regresión no aporta un pronóstico útil de la duración con información previa a "
             "la llamada: la media histórica queda en el conjunto de confianza junto a los modelos optimizados.")))

exportar("mcs", mcs.reset_index(names="modelo"))
exportar("pruebas_regresion", dirigidas)
exportar("spa_bootstrap", {"spa": spa_p, "bootstrap": bs})
'''

S84_MD = r'''### 8.4 Intervalos de confianza al 95 % de las métricas principales

Intervalos *bootstrap* BCa con 1 000 réplicas sobre el conjunto de prueba (AUC y AUC-PR para clasificación, RMSE en escala log para regresión).'''

S84_CODE = r'''filas = [{"Modelo": r_["Modelo"], "Métrica": "AUC", "Valor": r_["AUC"], "IC 95 % inferior": r_["ic_auc"][0],
          "IC 95 % superior": r_["ic_auc"][1]} for _, r_ in res_te.iterrows()]
filas += [{"Modelo": r_["Modelo"], "Métrica": "AUC-PR", "Valor": r_["AUC-PR"], "IC 95 % inferior": r_["ic_ap"][0],
           "IC 95 % superior": r_["ic_ap"][1]} for _, r_ in res_te.iterrows()]
for k in orden_reg:
    ic = st.ic_bootstrap(root_mean_squared_error, rte, fin_reg[k]["pred_te"], n_rep=1000)
    filas.append({"Modelo": NM[fin_reg[k]["modelo"]], "Métrica": "RMSE (log)",
                  "Valor": root_mean_squared_error(rte, fin_reg[k]["pred_te"]), "IC 95 % inferior": ic[0],
                  "IC 95 % superior": ic[1]})
ic_tab = pd.DataFrame(filas)
tabla(ic_tab, dec=4, indice=False)
anchos = ic_tab.assign(ancho=ic_tab["IC 95 % superior"] - ic_tab["IC 95 % inferior"]).groupby("Métrica")["ancho"].mean()
print("Ancho medio del intervalo: " + "; ".join(f"{k} {num(v, 4)}" for k, v in anchos.items()))

exportar("intervalos", ic_tab)
'''

S9_CODE = r'''mejor_ext = m_clf.loc[m_clf["auc_pr_media"].idxmax()]
display(Markdown(f"""## 9. Conclusiones y limitaciones

### 9.1 Conclusiones

1. **Modelos y balanceo.** La mejor combinación en la validación anidada es {etiqueta(mejor_ext)}, con un AUC-PR externo de {num(mejor_ext['auc_pr_media'], 3)} ± {num(mejor_ext['auc_pr_de'], 3)}; dentro de 2008 ningún modelo discrimina y la señal aparece al validar en 2009, donde los modelos simples extrapolan igual o mejor que los flexibles. En prueba, el mayor AUC-PR entre los finalistas es el de {mejor_te['Modelo']} ({num(mejor_te['AUC-PR'], 3)} {mejor_te['AUC-PR IC 95 %']}), {'por debajo del' if refe['AUC-PR'] > mejor_te['AUC-PR'] else 'frente al'} {num(refe['AUC-PR'], 3)} del modelo base del Entregable 2 (logística L2 con C = 0,01). {'Es decir, en AUC-PR las 140 combinaciones no superan a una logística bien regularizada: en un problema con cambio de régimen, la regularización y la simplicidad pesan más que la flexibilidad y la búsqueda de hiperparámetros. ' if refe['AUC-PR'] >= mejor_te['AUC-PR'] else ''}Las diferencias entre los mejores finalistas son del orden de la variabilidad entre periodos: el diagrama de diferencia crítica no separa a ninguno con significancia {'y DeLong no confirma ninguna de las ' + str(len(dl_t)) + ' diferencias dirigidas' if (dl_t['p Holm'] >= 0.05).all() else 'y DeLong confirma ' + str(int((dl_t['p Holm'] < 0.05).sum())) + ' de ' + str(len(dl_t)) + ' diferencias dirigidas'} tras el ajuste de Holm{'' if (dl_t['p Holm'] >= 0.05).all() else ' (' + '; '.join(r['Modelo A'] + ' frente a ' + r['Modelo B'] + ' en AUC, ' + num(r['Diferencia de AUC'], 4) for _, r in dl_t[dl_t['p Holm'] < 0.05].iterrows()) + '), una ventaja en el orden de los clientes que no se traslada al AUC-PR'}.
2. **Balanceo.** SMOTE, ADASYN y `class_weight` elevan el recall y el F1 con el umbral por defecto, pero no mejoran el orden de los clientes (AUC-PR) y empeoran la calibración: actúan como un desplazamiento del umbral. El cambio de prevalencia entre el entrenamiento ({pct(y_tr.mean())}) y la prueba ({pct(y_te.mean())}) es un problema distinto, que solo la recalibración con los meses cerrados corrige.
3. **Optimización.** Con 20 configuraciones por pliegue, Friedman sobre la métrica externa da {p_txt(fr_clf['p'])} entre los cuatro métodos de la guía, y sobre el área bajo la curva *anytime* {p_txt(fr_auac['p'])}. El área bajo la curva *anytime* es de {num(auac_m['grid'], 3)} con Grid Search, {num(auac_m['random'], 3)} con Random Search, {num(auac_m['bayes'], 3)} con la bayesiana y {num(auac_m['genetico'], 3)} con el genético: los métodos adaptativos llegan antes a buenas soluciones que la rejilla (delta de Cliff de {num(b_vs['delta de Cliff'].iloc[0], 2)} y {num(g_vs['delta de Cliff'].iloc[0], 2)}), pero frente a Random Search la ventaja es pequeña ({num(b_vs['delta de Cliff'].iloc[1], 2)} y {num(g_vs['delta de Cliff'].iloc[1], 2)}), porque la fase aleatoria inicial consume buena parte de un presupuesto de 20 configuraciones. La mejor búsqueda tampoco se traduce en mejores modelos en el bucle externo de forma apreciable. Successive Halving examina {int(hv['n_eval'].mean())} configuraciones por pliegue con un costo equivalente a {num(hv['eval_equivalentes'].mean(), 1)} evaluaciones completas.
4. **Costo computacional.** El cuello de botella real es el ajuste de los modelos ({num(pc.set_index('Componente').loc['Ajuste de XGBoost (booster)', '% del total'], 0)} % del tiempo en el perfil de XGBoost), seguido del preprocesamiento ({num(pc.set_index('Componente').loc['Preprocesamiento (ColumnTransformer)', '% del total'], 0)} %); el costo propio del optimizador es despreciable. Los histogramas de XGBoost (unas {num(ex['Ajuste (s)'] / hi_['Ajuste (s)'], 0)} veces más rápidos sin perder desempeño), la aproximación de Nyström en la SVM y la paralelización al nivel de tarea son las optimizaciones con mayor efecto; SAGA y los índices espaciales de KNN no compensan con 49 columnas, y FAISS solo conviene cuando la inferencia es masiva.
5. **Regresión.** Con información previa a la llamada, la duración apenas se puede pronosticar: el mejor R² en prueba es {num(mejor_r['R²'], 3)}. Los modelos superan a la media histórica de forma significativa (Clark-West {p_txt(cw['p'])}, SPA {p_txt(spa_p['consistent'])}) y el Model Confidence Set conserva {len(en_mcs)} de {len(mcs)} modelos{', sin la media' if 'Media histórica (referencia)' not in en_mcs.index else ''}, pero la ganancia es demasiado pequeña para planificar llamada a llamada; es coherente con la sección 3.5: la duración depende sobre todo de lo que ocurre durante la llamada.

### 9.2 Limitaciones

- **Cambio de régimen.** Los pliegues de 2008 casi no permiten discriminar y la prueba (2009-2010) tiene otra prevalencia; las métricas del bucle externo y las de prueba no son comparables en nivel, y la media ± desviación estándar mezcla variabilidad de muestreo y cambio de régimen.
- **Presupuesto corto.** Veinte configuraciones por pliegue limitan la ventaja de los métodos adaptativos y la exploración del genético, cuya población converge en pocas generaciones; las conclusiones sobre eficiencia valen para este presupuesto y espacios de dos a cuatro dimensiones.
- **Pocos bloques.** Cinco pliegues externos dan poca potencia a Friedman y una diferencia crítica de Nemenyi amplia; los bloques mensuales complementan, pero no son independientes.
- **Sobremuestreo en el espacio *one-hot*.** SMOTE y ADASYN interpolan columnas binarias; SMOTE-NC sería más apropiado para variables categóricas.
- **Entorno.** No se usó GPU: la máquina no tiene GPU NVIDIA y, con unos 41 000 registros, no se justifica, porque XGBoost con histogramas ajusta en {num(hi_['Ajuste (s)'], 2)} s en CPU y scikit-learn está diseñado para CPU. Tampoco se usaron Dask ni Ray, porque los datos caben en memoria; los tiempos son de una sola máquina de 12 núcleos.
- **Objetivo de regresión.** `log1p(duration)` se eligió por ser la única variable continua con sentido a nivel de cliente; su falta de señal limita el valor de la rama de regresión como comparación entre modelos.
"""))'''

S10_MD = r'''## 10. Referencias

- Akiba, T., Sano, S., Yanase, T., Ohta, T. y Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD*, 2623–2631. https://doi.org/10.1145/3292500.3330701
- Benjamini, Y. y Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300.
- Bergstra, J., Bardenet, R., Bengio, Y. y Kégl, B. (2011). Algorithms for hyper-parameter optimization. *Advances in Neural Information Processing Systems, 24*.
- Bergstra, J. y Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3.
- Brock, W. A., Dechert, W. D., Scheinkman, J. A. y LeBaron, B. (1996). A test for independence based on the correlation dimension. *Econometric Reviews, 15*(3), 197–235.
- Cawley, G. C. y Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107.
- Chawla, N. V., Bowyer, K. W., Hall, L. O. y Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794. https://doi.org/10.1145/2939672.2939785
- Clark, T. E. y West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics, 138*(1), 291–311.
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin, 114*(3), 494–509.
- Deb, K. y Agrawal, R. B. (1995). Simulated binary crossover for continuous search space. *Complex Systems, 9*, 115–148.
- DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics, 44*(3), 837–845.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30.
- Diebold, F. X. y Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*(3), 253–263.
- Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the American Statistical Association, 82*(397), 171–185.
- Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M. y Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. *Journal of Machine Learning Research, 13*, 2171–2175.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association, 32*(200), 675–701.
- Giacomini, R. y White, H. (2006). Tests of conditional predictive ability. *Econometrica, 74*(6), 1545–1578.
- Gneiting, T. y Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477), 359–378.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics, 23*(4), 365–380.
- Hansen, P. R., Lunde, A. y Nason, J. M. (2011). The model confidence set. *Econometrica, 79*(2), 453–497.
- Harvey, D., Leybourne, S. y Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting, 13*(2), 281–291.
- He, H., Bai, Y., Garcia, E. A. y Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. *IEEE International Joint Conference on Neural Networks*, 1322–1328. https://doi.org/10.1109/IJCNN.2008.4633969
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.
- Iman, R. L. y Davenport, J. M. (1980). Approximations of the critical region of the Friedman statistic. *Communications in Statistics: Theory and Methods, 9*(6), 571–595.
- Jamieson, K. y Talwalkar, A. (2016). Non-stochastic best arm identification and hyperparameter optimization. *Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 240–248.
- Johnson, J., Douze, M. y Jégou, H. (2021). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data, 7*(3), 535–547.
- Lemaître, G., Nogueira, F. y Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research, 18*(17), 1–5.
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. y Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research, 18*(185), 1–52.
- Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.
- Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. y Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*, 56–67.
- Malkov, Y. A. y Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 42*(4), 824–836.
- Moro, S., Cortez, P. y Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
- Naeini, M. P., Cooper, G. F. y Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *Proceedings of the AAAI Conference on Artificial Intelligence, 29*(1), 2901–2907.
- Nemenyi, P. B. (1963). *Distribution-free multiple comparisons* (tesis doctoral). Princeton University.
- Niculescu-Mizil, A. y Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.
- Politis, D. N. y Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association, 89*(428), 1303–1313.
- Politis, D. N. y White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews, 23*(1), 53–70.
- Ribeiro, M. T., Singh, S. y Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD*, 1135–1144. https://doi.org/10.1145/2939672.2939778
- Romano, J., Kromrey, J. D., Coraggio, J. y Skowronek, J. (2006). Appropriate statistics for ordinal level data: Should we really be using t-test and Cohen's d for evaluating group differences on the NSSE and other surveys? *Annual Meeting of the Florida Association of Institutional Research*.
- Saerens, M., Latinne, P. y Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. *Neural Computation, 14*(1), 21–41.
- Saito, T. y Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
- Sun, X. y Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters, 21*(11), 1389–1393.
- Varma, S. y Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics, 7*, 91. https://doi.org/10.1186/1471-2105-7-91
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica, 48*(4), 817–838.
- White, H. (2000). A reality check for data snooping. *Econometrica, 68*(5), 1097–1126.
- Williams, C. K. I. y Seeger, M. (2001). Using the Nyström method to speed up kernel machines. *Advances in Neural Information Processing Systems, 13*, 682–688.
- Zadrozny, B. y Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *Proceedings of the 8th ACM SIGKDD*, 694–699.'''
