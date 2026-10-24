"""Secciones 7 a 9: artefacto del modelo original, conclusiones, reproducibilidad y referencias."""

S71_MD = r'''## 7. Dashboard y despliegue

### 7.1 Artefacto del modelo original y prueba de ida y vuelta

El dashboard nunca entrena modelos (guía, sección 4.2): carga artefactos ya ajustados. El L2E se serializa con `joblib` (compresión 3) en dos versiones: `modelo.joblib`, con el estado al cierre del entrenamiento, que es el que se evaluó en la prueba, y `modelo_vigente.joblib`, que además asimiló con `actualizar` las etiquetas de todos los días de la prueba y es el que el dashboard debe usar para predecir un día nuevo. Junto a ellos, `metadata.json` registra las versiones de las librerías, la semilla, los hiperparámetros, el esquema de entrada y las métricas del bucle externo y de la prueba. La prueba de ida y vuelta verifica que el modelo leído de disco produce exactamente las mismas predicciones que el modelo en memoria. Estos artefactos referencian el paquete de investigación `bm`; la versión que carga el dashboard la genera `scripts/exportar_dashboard_l2e.py` con la copia de producción de la clase y repite la misma verificación (sección 7.3).'''

S71_CODE = r'''import copy
import joblib

DEST = Path("artifacts") / "l2e"
DEST.mkdir(parents=True, exist_ok=True)
pipe = L2E["pipeline"]
X_te_d = con_dia(X_te, tiempo_te)
joblib.dump(pipe, DEST / "modelo.joblib", compress=3)
vigente = copy.deepcopy(pipe)
vigente[-1].actualizar(vigente[:-1].transform(X_te_d), yte)
joblib.dump(vigente, DEST / "modelo_vigente.joblib", compress=3)

esquema = {}
for c in X_tr.columns:
    if c in cfg.VARS_NUM:
        esquema[c] = {"tipo": "numérica", "min": float(X_tr[c].min()), "max": float(X_tr[c].max())}
    elif c in cfg.VARS_CAT:
        esquema[c] = {"tipo": "categórica", "valores": sorted(map(str, X_tr[c].unique()))}
esquema["dia"] = {"tipo": "entero", "descripcion": "día de contacto reconstruido (bm.datos.reconstruir_tiempo)"}
metadata = {
    "modelo": "LogisticaDosEscalas (L2E)", "clase": "bm.modelo_original.LogisticaDosEscalas", "semilla": SEED,
    "versiones": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                  "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    "hiperparametros": {k.split("__")[1]: float(v) for k, v in L2E["params"].items()},
    "parametros_fijos": {"retardo": 1, "nivel_latente": True, "actualizar_nivel": True, "actualizar_pendientes": True,
                         "integrar_incertidumbre": True},
    "esquema_entrada": esquema,
    "metricas_bucle_externo": {"auc_pr_media": float(ext["L2E"].mean()), "auc_pr_de": float(ext["L2E"].std(ddof=1)),
                               "auc_media": float(np.mean(ext_auc["L2E"]))},
    "metricas_prueba": {"auc_pr": float(ap_te["L2E"]), "auc": float(auc_te["L2E"]),
                        "brier": float(brier_score_loss(yte, PUNTAJE_TE["L2E"]))},
    "estado": {"modelo.joblib": {"ultimo_dia": int(pipe[-1].dia_final_), "periodo": str(periodo_tr.iloc[-1])},
               "modelo_vigente.joblib": {"ultimo_dia": int(vigente[-1].dia_final_), "periodo": str(periodo_te.iloc[-1])}},
}
(DEST / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1), encoding="utf-8")

leido = joblib.load(DEST / "modelo.joblib")
s_leido = puntaje_secuencial(leido, X_te_d, y_te)
dif_script = float(np.max(np.abs(s_leido - L2E["puntaje_te"])))
ida_vuelta = pd.DataFrame([
    {"Verificación": "predict_proba (estático): memoria frente a disco", "Criterio": "igualdad exacta",
     "Resultado": "idénticas" if np.array_equal(leido.predict_proba(X_te_d), pipe.predict_proba(X_te_d)) else "DIFERENTES"},
    {"Verificación": "predict_proba_secuencial: memoria frente a disco", "Criterio": "igualdad exacta",
     "Resultado": "idénticas" if np.array_equal(s_leido, puntaje_secuencial(pipe, X_te_d, y_te)) else "DIFERENTES"},
    {"Verificación": "frente al puntaje evaluado en la prueba (otro proceso)", "Criterio": "diferencia máxima < 1e−10",
     "Resultado": f"diferencia máxima {dif_script:.1e}" + (" (coinciden)" if dif_script < 1e-10 else " (DIFERENTES)")}])
tamanos = {f.name: f.stat().st_size / 1024 for f in DEST.iterdir()}
tabla(ida_vuelta, indice=False)
tabla(pd.Series(tamanos, name="Tamaño (KB)").to_frame(), dec=1)
assert ida_vuelta["Resultado"].iloc[:2].eq("idénticas").all() and dif_script < 1e-10
print("La comparación con el puntaje de la prueba se hace con tolerancia porque ese puntaje se calculó en otro proceso, "
      "con BLAS de un solo hilo; el orden de las sumas de punto flotante cambia en el último dígito.")'''

S72_MD = r'''### 7.2 Predicción en vivo con explicación local

Así se usa el modelo en el dashboard: se toma un cliente con sus variables previas a la llamada, se le asigna el día siguiente al último asimilado y el modelo vigente devuelve la probabilidad y su descomposición exacta en el logit: el nivel del día y la contribución de cada variable del cliente.'''

S72_CODE = r'''cliente = X_te.iloc[[-1]].copy()
dia_nuevo = int(vigente[-1].dia_final_) + 1
fila = cliente.assign(dia=dia_nuevo)
prob = float(vigente.predict_proba(fila)[0, 1])
Z = vigente[:-1].transform(fila)[:, :-1]
m_v = vigente[-1].estado_media_
xbar = vigente[:-1].transform(con_dia(X_tr, tiempo_tr))[:, :-1].mean(axis=0)
contrib = pd.Series(m_v[1:] * (Z[0] - xbar), index=nombres)
nivel_dia = float(m_v[0] + m_v[1:] @ xbar)
top_c = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(["nivel del día (logit esperado)"] + list(top_c.index), [nivel_dia] + list(top_c.values),
        color=[TINTA] + [NARANJA if v > 0 else AZUL for v in top_c.values])
ax.axvline(0, color=GRIS, lw=0.8)
ax.invert_yaxis()
ax.set(title=f"Explicación local del L2E · probabilidad estimada = {num(prob, 3)}", xlabel="Contribución al logit")
plt.tight_layout()
plt.show()
tabla(cliente[cfg.VARS_NUM + cfg.VARS_CAT].T.rename(columns={cliente.index[0]: "Valor"}), indice=True)
interpretar(
    f"Para este cliente, con el estado vigente al cierre de {periodo_te.iloc[-1]}, el L2E estima una probabilidad de "
    f"suscripción de {num(prob, 3)}. El nivel del día aporta {num(nivel_dia, 2)} al logit y la variable con mayor "
    f"contribución es `{top_c.index[0]}` ({num(top_c.iloc[0], 2)}).",
    conectar("De esta forma,", "La explicación local separa lo que se debe al momento (el nivel del día, común a todos "
             "los clientes que se llamen ese día) de lo que se debe al cliente, y responde a la pregunta que se hace un "
             "asesor: si conviene llamarlo hoy y por qué."),
    conectar("Por tanto, corresponde", "Incorporar esta salida, la trayectoria del nivel y la tabla de pruebas de la "
             "sección 6.7 a la subsección del modelo original de la página Modelos del dashboard (sección 7.3)."))'''

S73_MD = r'''### 7.3 Dashboard desplegado: arquitectura, decisiones de diseño y despliegue

El dashboard está en línea en Render: **<https://proyecto-final-bank-marketing.onrender.com>**. Tiene las tres páginas que pide la guía (Contexto, EDA y Modelos) y, desde este entregable, el L2E es un modelo más del dashboard: encabeza la tabla de líderes, se puede elegir en la predicción en vivo, tiene su diagnóstico y su SHAP global, entra al diagrama de diferencia crítica y cuenta con su propia subsección.

**Arquitectura.** El repositorio (<https://github.com/DTL-DA/proyecto-final-bank-marketing>) sigue la estructura de la sección 7.2 de la guía; este cuaderno y su código de investigación están en `cuarto_entregable/`:

```text
proyecto-final-bank-marketing/
├── app.py                  punto de entrada: crea la aplicación multipágina y expone server (gunicorn)
├── pages/                  contexto.py, eda.py y modelos.py (una página por sección, con sus callbacks)
├── src/
│   ├── original_model.py   clase del modelo original (copia de producción de bm.modelo_original)
│   ├── data.py             carga y caché de datos, artefactos y resultados precalculados
│   ├── plots.py            funciones que construyen las figuras, con una plantilla común de Plotly
│   ├── explicaciones.py    valores SHAP por permutaciones (sin cargar la librería shap en el servidor)
│   └── componentes.py      componentes propios de los pipelines de los benchmarks
├── artifacts/              un directorio por modelo con modelo.joblib y metadata.json (l2e incluido)
├── data/                   datos livianos para el EDA y resultados precalculados (data/resultados)
├── assets/                 hoja de estilos
├── tests/                  pruebas con pytest: aplicación, artefactos, explicaciones y modelo original
├── book/                   fuentes del Jupyter Book (_config.yml, _toc.yml, references.bib)
├── cuarto_entregable/      este cuaderno, src/bm, scripts, pruebas y resultados del modelo original
├── requirements.txt        versiones exactas
└── render.yaml             configuración del despliegue
```

**Integración del modelo original.** `scripts/exportar_dashboard_l2e.py` toma el modelo final de la sección 7.1, reemplaza las referencias al paquete de investigación `bm` por la copia de producción `src/original_model.py` (el algoritmo es idéntico línea por línea, y `tests/test_original_model.py` lo verifica junto con las 25 pruebas unitarias del modelo), repite la prueba de ida y vuelta, guarda los dos artefactos en `artifacts/l2e/` y agrega el L2E a las tablas precalculadas del dashboard. Como el dashboard nunca entrena ni optimiza, todo lo que muestra sale de esos artefactos y de esas tablas. Para predecir, el dashboard carga `modelo_vigente.joblib` y le asigna al cliente el día siguiente al último cerrado (sección 7.2).

**Decisiones de diseño.**

- *Un producto, no una colección de gráficos:* cada tarjeta responde una pregunta del usuario (a quién llamar primero, qué tan confiable es la probabilidad, qué modelo es mejor y por qué) y termina con una frase de interpretación.
- *Tema y color:* `dash-bootstrap-components` con el tema FLATLY, barra de navegación fija y una paleta de dos colores más grises, validada para visión normal y para las tres formas de daltonismo. El color sigue a la entidad: el L2E es siempre naranja y los benchmarks, azules, y la identidad nunca depende solo del color, porque cada serie tiene su etiqueta.
- *Rendimiento en el plan gratuito:* los datos y los ocho modelos se cargan una sola vez con `functools.lru_cache`; los valores SHAP se estiman con un estimador por permutaciones propio, porque la librería `shap` no cabe en la memoria junto con los modelos, y el L2E solo suma 37 KB de artefactos.
- *Robustez y accesibilidad:* el formulario de predicción se genera desde el esquema de variables y valida tipos y rangos; un error se informa con `dbc.Alert` sin detener la aplicación. Las tablas y las figuras usan la rejilla de `dbc.Row` y `dbc.Col` para escritorio y tableta, con etiquetas descriptivas y contraste suficiente.

**Guía de despliegue.** (1) `python app.py` levanta el dashboard en local en <http://127.0.0.1:8050>. (2) `requirements.txt` fija las versiones exactas, incluido `gunicorn`, y `render.yaml` fija Python 3.12.10. (3) Render construye con `pip install -r requirements.txt` y arranca con `gunicorn app:server --workers 1 --threads 4 --timeout 120`; cada *push* a la rama `main` del repositorio dispara un despliegue nuevo. (4) Después de cada despliegue se revisan los registros de construcción y se recorren las tres páginas en la URL pública. En el plan gratuito el servicio se suspende tras 15 minutos sin visitas y la primera carga tarda cerca de un minuto, por lo que conviene abrirlo unos minutos antes de la sustentación.

**Capturas.** Las capturas se tomaron con `scripts/capturas_dashboard.py` sobre el dashboard local, con los mismos artefactos y datos que la versión en línea.

![Página Contexto del dashboard: problema, pregunta de investigación, datos, metodología y limitaciones](imagenes/dashboard_contexto.png)

![Página EDA del dashboard: filtros globales, distribución del objetivo, explorador bivariado, correlación, faltantes y atípicos](imagenes/dashboard_eda.png)

![Página Modelos, primera parte: tabla de líderes con el L2E, predicción en vivo con su explicación SHAP local y diagnóstico del modelo seleccionado](imagenes/dashboard_modelos_1.png)

![Página Modelos, segunda parte: SHAP global, diagrama de diferencia crítica con el modelo original y subsección del L2E con su veredicto](imagenes/dashboard_modelos_2.png)'''

S8_CODE = r'''ganancia_mejor = bt_ap.set_index("modelo").loc[MEJOR, "diferencia"]
equivalentes = [CORTO[k] for k in COMP[1:] if CORTO[k] not in sig_te]
adic_cols = [c for c in med.columns if c != "bank marketing"]
n_ad_gana = int(sum(r_l2e[c] == 1 for c in adic_cols))
display(Markdown(f"""## 8. Conclusiones y limitaciones

### 8.1 Hallazgos

1. **El límite de los 140 modelos del tercer entregable era temporal, no de flexibilidad.** La tasa diaria de suscripción es un nivel latente con memoria larga (autocorrelación de {num(fac[1], 2)} al primer rezago en el entrenamiento), y un puntaje que solo conoce la tasa de los días cerrados alcanza en el bucle externo un AUC-PR de {num(solo_nivel.iloc[:3]['Media'].max(), 3)}, frente a {num(solo_nivel.iloc[-1]['Media'], 3)} de la mejor de las 140 combinaciones. Ningún modelo estático puede representar esa estructura, y por eso los finalistas eran estadísticamente equivalentes.
2. **El L2E convierte ese hallazgo en un modelo.** Separa un nivel diario que se mueve rápido de unas pendientes que derivan despacio, se ajusta con un Newton por bloques de costo lineal en el número de días y se actualiza con un filtro de Laplace que respeta la demora de las etiquetas. La recalibración mensual que sugirió el docente es un caso límite de su filtro (propiedad P5).
3. **Desempeño.** En el bucle externo, el L2E obtiene {ms(ext['L2E'])}, frente a {ms(ext[MEJOR])} de la mejor combinación; en la prueba, un AUC-PR de {num(ap_te['L2E'], 3)} y un AUC de {num(auc_te['L2E'], 3)}, con una ganancia de {num(ganancia_mejor, 3)} en AUC-PR sobre la mejor combinación y probabilidades calibradas sin recalibración aparte (Brier de {num(resumen.iloc[0]['Brier prueba'], 3)}).
4. **Dónde está la ventaja.** Dentro de cada mes de prueba, el L2E es significativamente mejor que {enumerar(sig_te)}, y en el ordenamiento global de la prueba supera a todos los benchmarks. Frente a {enumerar(equivalentes)} no es superior dentro de cada mes: la ventaja viene de seguir el nivel de conversión entre días, no de ordenar mejor a los clientes de un mismo día.
5. **El mecanismo es transferible.** El filtro de nivel aislado mejora a {c2_gana} de los {len(COMP) - 1} benchmarks cuando se les aplica como envoltorio y supera a la recalibración mensual en {len(c2_vs_mensual)} de {len(con_mensual)} modelos con probabilidades; el mejor de esos híbridos queda a {num(bt_c['diferencia'], 3)} del L2E en AUC-PR de prueba ({p_txt(bt_c['p'])}).
6. **Límites de validez.** La ventaja se reduce con la demora de las etiquetas, y {'en ninguno de los tres conjuntos adicionales el L2E es el mejor modelo' if n_ad_gana == 0 else f'el L2E es el mejor modelo en {n_ad_gana} de los tres conjuntos adicionales'}: donde el nivel latente es débil o lo explican las variables, no supera a los benchmarks. El modelo es una herramienta para problemas de propensión con un nivel persistente, como el telemercadeo bancario, no un clasificador general.
7. **Mejoras evaluadas.** De los cinco ajustes para los árboles y las cinco mejoras del L2E (sección 6.13), {'ninguna supera la regla de decisión declarada, así que el L2E se conserva sin cambios' if (t_mej['Decisión'] == 'Se adopta').sum() == 0 else 'se adopta ' + enumerar(t_mej.loc[t_mej['Decisión'] == 'Se adopta', 'Mejora'].tolist())}; los ajustes que más ayudan a los modelos de referencia son los que usan las etiquetas recientes, el mecanismo del modelo original.

### 8.2 Veredicto sobre R1, R2 y R3

R1 (novedad en la aplicación) y R2 (implementación algorítmica propia) se cumplen. R3 {'se cumple' if r3 else 'no se cumple'} con la definición operativa de la guía: la media externa del L2E supera a la de los siete finalistas, {n_r3} pruebas distintas rechazan la igualdad frente a la mejor combinación benchmark con p ajustado < 0,05 y el efecto es {st.magnitud_cliff(cliff_mejor)} (δ de Cliff = {num(cliff_mejor, 3)}). Por la regla de la guía (sección 12), el tope que corresponde es {'5,0' if r3 else '4,0'}, sujeto a la evaluación del docente.

### 8.3 Limitaciones

- **La significancia depende de la unidad de análisis.** Con los cinco pliegues externos o con los meses de la validación externa ninguna prueba es significativa, porque en 2008 ningún modelo discrimina; la evidencia de superioridad viene de la prueba (2009-2010), que es el régimen de uso pero también un solo periodo.
- **Sin ventaja dentro de cada periodo frente a las mejores logísticas.** El L2E no ordena mejor a los clientes de un mismo mes que una logística bien regularizada o la SVM, y la prueba bayesiana se inclina levemente a favor de ellas; su aporte es el nivel.
- **Condición operativa.** El modelo necesita que las etiquetas de cada día estén disponibles poco después; con demoras de semanas pierde parte de su ventaja.
- **Aproximaciones.** Las pendientes se toman constantes dentro de la ventana de entrenamiento, la posterior se aproxima por Laplace y la probabilidad predictiva por la aproximación probit; ninguna se evaluó frente a una inferencia exacta (por ejemplo, MCMC).
- **Estabilidad de los coeficientes entre pliegues.** El L2E es menos estable que la logística con C = 0,01, porque la validación interna elige un λ distinto en cada pliegue (sección 1.2).
- **Selección de hiperparámetros con pocos datos.** En el primer pliegue de *electricity* la validación interna eligió una dinámica que no existía; con ventanas de ajuste pequeñas, el modelo puede perseguir ruido.
- **Revisión de literatura.** Se hizo con APIs públicas; Scopus y Web of Science no se consultaron por falta de acceso programático, y las búsquedas por relevancia no permiten intersecciones booleanas.
- **Uso de la prueba durante el prototipado.** Durante el desarrollo se evaluaron algunas variantes exploratorias también en la prueba. Las decisiones finales están respaldadas por la evidencia del entrenamiento y del bucle externo (secciones 2.2, 5.8 y 5.9), pero esa consulta previa es una forma de filtración de la prueba hacia el diseño y se declara aquí.

### 8.4 Trabajo futuro

- Validar con datos posteriores a 2010 el ensamble del L2E con un Random Forest reentrenado cada mes (mejora M4), que sube el AUC-PR de la prueba pero no el del bucle externo, y por eso no se adoptó (sección 6.13).
- Buscar una forma de elegir $q_u$ que funcione con ventanas de ajuste cortas: el Bayes empírico (mejora M2) aprende la dinámica del pasado y falla cuando el régimen cambia más de lo que cambiaba antes.
- Extender el nivel a varios grupos (por canal de contacto o por campaña) con un modelo jerárquico, y combinar el filtro con un clasificador no lineal como término fijo, ya que el componente C2 es transferible.
- Evaluar el modelo en datos posteriores a 2010 o en otros problemas de propensión con etiquetas diferidas (abandono de clientes, cobranza, conversión publicitaria).

### 8.5 Declaración de autoría y uso de herramientas de inteligencia artificial

Este trabajo es de autoría de Donny Torres. En su desarrollo se usó Claude (Anthropic) como asistente de programación, análisis y redacción: apoyo en la evaluación crítica del tercer entregable, en la formulación del modelo L2E y la derivación de su algoritmo, en la implementación del código, de las pruebas unitarias y de los *scripts* de experimentación, en la búsqueda y verificación bibliográfica, en la integración del modelo al dashboard, en el armado de este Jupyter Book y en la redacción de los textos. Todas las cifras se calculan con el código incluido y se reproducen con los comandos de la sección 9. El autor revisó el código, las derivaciones y los resultados, y es responsable de comprender, explicar y modificar cualquier parte del algoritmo y de las pruebas estadísticas, como exige la guía (secciones 5.4 y 13).

**Firma:** Donny Torres · Maestría en Analítica de Datos, Universidad del Norte · Barranquilla, 24 de octubre de 2026."""))'''

S91_MD = r'''## 9. Reproducibilidad y bibliografía

### 9.1 Entorno, semillas y ejecución

- **Entorno:** `conda create -n bm_e3 python=3.12`, `pip install -r requirements.txt` y `python -m ipykernel install --user --name bm_e3`. Semilla global `SEED = 42` en `src/bm/config.py`; el L2E es determinista y la semilla solo afecta a Optuna.
- **Datos:** `data/bank-additional-full.csv` (UCI, CC BY 4.0) y los tres conjuntos de OpenML (identificadores 1169, 151 y 1487), que `sklearn.datasets.fetch_openml` descarga en `data/adicionales/` la primera vez.
- **Orden de ejecución** (desde la raíz del proyecto): `python scripts/importar_e3.py` (resultados del tercer entregable), `python scripts/ejecutar_original.py` (validación anidada, ablación, semillas y modelos finales; unos 6 minutos con 10 núcleos), `python scripts/ejecutar_adicionales.py` (conjuntos adicionales; unos 25 minutos), `python scripts/ejecutar_analisis.py` (sensibilidad, robustez, complejidad y SHAP), `python scripts/revision_literatura.py` y `python scripts/verificar_referencias.py` (revisión y bibliografía), `pytest -q tests` y `python scripts/construir_cuaderno.py --ejecutar`. Para el dashboard y el libro (en el repositorio, desde `cuarto_entregable/`): `python scripts/exportar_dashboard_l2e.py --app ..` (artefactos y resultados del L2E en el dashboard), `python scripts/capturas_dashboard.py --url <URL del dashboard>` (capturas del capítulo 7) y `python scripts/construir_libro.py --libro ../book` (fuentes del Jupyter Book, que se compila con `jupyter-book build book/`).'''

S92_CODE = r'''display(Markdown("### 9.2 Tabla maestra de experimentos"))
m_e3 = maestra[["tarea", "modelo", "balanceo", "optimizador", "auc_pr_media", "auc_pr_de", "auc_media", "auc_de",
                "rmse_media", "rmse_de", "r2_media", "t_opt_total", "t_fit_medio", "t_pred_medio", "seed"]].copy()
m_e3["origen"] = "Tercer entregable"
filas = []
for v in VARIANTES:
    c_ = corr[v]
    filas.append({"tarea": "clf", "modelo": f"l2e:{v}", "balanceo": "ninguno", "optimizador": "bayes",
                  "auc_pr_media": np.mean([c["auc_pr"] for c in c_]), "auc_pr_de": np.std([c["auc_pr"] for c in c_], ddof=1),
                  "auc_media": np.mean([c["auc"] for c in c_]), "auc_de": np.std([c["auc"] for c in c_], ddof=1),
                  "t_opt_total": sum(c["t_opt"] for c in c_), "t_fit_medio": np.mean([c["t_fit"] for c in c_]),
                  "t_pred_medio": np.mean([c["t_pred"] for c in c_]), "seed": SEED, "origen": "Modelo original"})
for s, c_ in corr_sem.items():
    filas.append({"tarea": "clf", "modelo": "l2e:l2e", "balanceo": "ninguno", "optimizador": "bayes",
                  "auc_pr_media": np.mean([c["auc_pr"] for c in c_]), "auc_pr_de": np.std([c["auc_pr"] for c in c_], ddof=1),
                  "auc_media": np.mean([c["auc"] for c in c_]), "auc_de": np.std([c["auc"] for c in c_], ddof=1),
                  "t_opt_total": sum(c["t_opt"] for c in c_), "t_fit_medio": np.mean([c["t_fit"] for c in c_]),
                  "t_pred_medio": np.mean([c["t_pred"] for c in c_]), "seed": s, "origen": "Modelo original (semilla)"})
maestra_e4 = pd.concat([m_e3, pd.DataFrame(filas)], ignore_index=True)
maestra_e4["prueba_auc_pr"] = maestra_e4["modelo"].map(lambda m: fin_l2e[m.split(":")[1]]["auc_pr"] if m.startswith("l2e:") else np.nan)
maestra_e4.loc[maestra_e4["origen"] == "Modelo original (semilla)", "prueba_auc_pr"] = [fin_sem[s]["auc_pr"] for s in corr_sem]
maestra_e4.to_csv(RES / "tabla_maestra_e4.csv", index=False)
maestra_e4.to_parquet(RES / "tabla_maestra_e4.parquet", index=False)
tabla(maestra_e4[maestra_e4["tarea"] == "clf"].sort_values("auc_pr_media", ascending=False).head(15), dec=4, indice=False)
n_e3 = maestra_e4["origen"].eq("Tercer entregable")
print(f"Tabla maestra: {len(maestra_e4)} filas ({n_e3.sum()} combinaciones del tercer entregable, "
      f"{(n_e3 & maestra_e4['tarea'].eq('clf')).sum()} de clasificación y {(n_e3 & maestra_e4['tarea'].eq('reg')).sum()} de "
      f"regresión, y {(~n_e3).sum()} corridas del modelo original) en resultados/tabla_maestra_e4.csv y .parquet")
assert n_e3.sum() == 140'''

S93_MD = r'''### 9.3 Referencias

Las entradas en formato BibTeX están en `resultados/literatura/references.bib`; las que tienen DOI se generaron desde doi.org con `scripts/verificar_referencias.py`.

- Akiba, T., Sano, S., Yanase, T., Ohta, T. y Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623-2631. <https://doi.org/10.1145/3292500.3330701>
- Benavoli, A., Corani, G., Demšar, J. y Zaffalon, M. (2017). Time for a change: A tutorial for comparing multiple classifiers through Bayesian analysis. *Journal of Machine Learning Research*, 18(77), 1-36.
- Bergstra, J., Bardenet, R., Bengio, Y. y Kégl, B. (2011). Algorithms for hyper-parameter optimization. *Advances in Neural Information Processing Systems*, 24.
- Boyd, S. y Vandenberghe, L. (2004). *Convex optimization*. Cambridge University Press. <https://doi.org/10.1017/CBO9780511804441>
- Cawley, G. C. y Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079-2107.
- Chowdary, B. V., Keerthi, K. L., Sandhya, D., Sunny, Ch. y Reddy, M. D. (2025). Advancing telemarketing success using ensemble-driven online machine learning. *2025 5th International Conference on Soft Computing for Security Applications (ICSCSA)*, 974-978. <https://doi.org/10.1109/icscsa66339.2025.11170856>
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, 114(3), 494-509. <https://doi.org/10.1037/0033-2909.114.3.494>
- DeLong, E. R., DeLong, D. M. y Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics*, 44(3), 837-845. <https://doi.org/10.2307/2531595>
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
- Diebold, F. X. y Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263. <https://doi.org/10.1080/07350015.1995.10524599>
- Durbin, J. y Koopman, S. J. (2012). *Time series analysis by state space methods* (2.ª ed.). Oxford University Press. <https://doi.org/10.1093/acprof:oso/9780199641178.001.0001>
- Fahrmeir, L. (1992). Posterior mode estimation by extended Kalman filtering for multivariate dynamic generalized linear models. *Journal of the American Statistical Association*, 87(418), 501-509. <https://doi.org/10.1080/01621459.1992.10475232>
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. y Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1-37. <https://doi.org/10.1145/2523813>
- Garg, S., Wu, Y., Balakrishnan, S. y Lipton, Z. C. (2020). A unified view of label shift estimation. *Advances in Neural Information Processing Systems*, 33.
- Grzenda, M., Gomes, H. M. y Bifet, A. (2020). Delayed labelling evaluation for data streams. *Data Mining and Knowledge Discovery*, 34(5), 1237-1266. <https://doi.org/10.1007/s10618-019-00654-y>
- Harries, M. (1999). *SPLICE-2 comparative evaluation: Electricity pricing* [Informe técnico]. University of New South Wales.
- Harvey, D., Leybourne, S. y Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281-291. <https://doi.org/10.1016/S0169-2070(96)00719-4>
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35-45. <https://doi.org/10.1115/1.3662552>
- Lipton, Z. C., Wang, Y.-X. y Smola, A. J. (2018). Detecting and correcting for label shift with black box predictors. *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 3122-3130.
- Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- MacKay, D. J. C. (1992). The evidence framework applied to classification networks. *Neural Computation*, 4(5), 720-736. <https://doi.org/10.1162/neco.1992.4.5.720>
- McCormick, T. H., Raftery, A. E., Madigan, D. y Burd, R. S. (2012). Dynamic logistic regression and dynamic model averaging for binary classification. *Biometrics*, 68(1), 23-30. <https://doi.org/10.1111/j.1541-0420.2011.01645.x>
- Moro, S., Cortez, P. y Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22-31. <https://doi.org/10.1016/j.dss.2014.03.001>
- Nadeau, C. y Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281. <https://doi.org/10.1023/A:1024068626366>
- Politis, D. N. y Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313. <https://doi.org/10.1080/01621459.1994.10476870>
- Romano, J., Kromrey, J. D., Coraggio, J. y Skowronek, J. (2006). Appropriate statistics for ordinal level data: Should we really be using t-test and Cohen's d for evaluating group differences on the NSSE and other surveys? *Annual Meeting of the Florida Association of Institutional Research*.
- Saerens, M., Latinne, P. y Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. *Neural Computation*, 14(1), 21-41. <https://doi.org/10.1162/089976602753284446>
- Varma, S. y Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91. <https://doi.org/10.1186/1471-2105-7-91>
- West, M., Harrison, P. J. y Migon, H. S. (1985). Dynamic generalized linear models and Bayesian forecasting. *Journal of the American Statistical Association*, 80(389), 73-83. <https://doi.org/10.1080/01621459.1985.10477131>
- Zhang, K. y Fan, W. (2008). Forecasting skewed biased stochastic ozone days: Analyses, solutions and beyond. *Knowledge and Information Systems*, 14(3), 299-326. <https://doi.org/10.1007/s10115-007-0095-1>'''
