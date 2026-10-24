"""Sección 3: modelos benchmark y selección de las mejores combinaciones (bloque B)."""

S3_MD = r'''## 3. Modelos benchmark

### 3.1 Mejores combinaciones por modelo y mejor combinación global

Del universo de 140 combinaciones del tercer entregable, el criterio de la guía (sección 4.1) es estadístico y no visual: para cada uno de los 7 modelos base se toma la combinación de balanceo y optimizador con la mayor **métrica media del bucle externo** (AUC-PR), y la mejor de ellas es la mejor combinación global. Estos siete finalistas, más el modelo base del Entregable 2 como referencia, son los *benchmarks* del modelo original; están serializados con su `metadata.json` y su prueba de ida y vuelta en `artifacts/` del repositorio del dashboard, que es el que los carga en producción.'''

S3_CODE = r'''filas = []
for r_ in fin_ext.itertuples():
    pl_ = pl_fin[pl_fin["modelo"] == r_.modelo]
    filas.append({"Modelo": cfg.NOMBRE_MODELO[r_.modelo], "Balanceo": cfg.NOMBRE_BALANCEO[r_.balanceo],
                  "Optimizador": cfg.NOMBRE_OPTIMIZADOR[r_.optimizador], "AUC-PR externo": ms(pl_["auc_pr"]),
                  "AUC externo": ms(pl_["auc"]), "Ajuste medio (s)": r_.t_fit_medio,
                  "Inferencia media (s)": r_.t_pred_medio})
benchmarks = pd.DataFrame(filas)
tabla(benchmarks, dec=3, indice=False)

# Pertenencia de la mejor combinación global al conjunto de finalistas de Friedman → Nemenyi del E3
mat_fin = pl_fin.pivot_table(index="pliegue", columns="modelo", values="auc_pr")[MODELOS]
fr_e3 = st.friedman(mat_fin)
nem_e3 = sp.posthoc_nemenyi_friedman(mat_fin.to_numpy())
nem_e3.index = nem_e3.columns = MODELOS
cd_e3 = st.diferencia_critica(len(MODELOS), len(mat_fin))
interpretar(
    f"La mejor combinación global es {NOMBRE[MEJOR]}, con un AUC-PR externo de "
    f"{benchmarks.iloc[0]['AUC-PR externo']}; la última de los siete finalistas, {NOMBRE[MODELOS[-1]]}, obtiene "
    f"{benchmarks.iloc[-1]['AUC-PR externo']}. Entre los finalistas, Friedman da χ² = {num(fr_e3['chi2'], 2)} "
    f"({p_txt(fr_e3['p'])}) y la menor p de Nemenyi frente a la mejor combinación es de "
    f"{num(nem_e3.loc[MEJOR].drop(MEJOR).min(), 3)}, con una diferencia crítica de {num(cd_e3, 2)} rangos.",
    conectar("Así mismo,", "La mejor combinación global pertenece al conjunto de modelos estadísticamente equivalentes: "
             "ninguno de los siete se separa de los demás y las desviaciones estándar entre pliegues son del mismo orden "
             "que las medias, porque un solo pliegue concentra la señal (sección 1.1). Superar a estos benchmarks no "
             "exige un clasificador más flexible, sino capturar la estructura que todos dejan sin modelar."),
    conectar("Por ello, se recomienda", "Comparar el modelo original contra los siete finalistas y contra la "
             "referencia del Entregable 2, con la mejor combinación global como contraste principal, tal como define "
             "la guía la condición R3 (sección 6.1), y sin descartar a ninguno de la comparación global de la "
             "sección 6.3."))'''
