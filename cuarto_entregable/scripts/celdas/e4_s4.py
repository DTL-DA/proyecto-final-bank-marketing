"""Sección 4: revisión de literatura, matriz de novedad, trabajos cercanos y declaración de novedad (sección 5.2)."""

S41_MD = r'''## 4. Revisión de literatura y novedad

La guía es explícita: la novedad no se declara, se demuestra con una revisión documentada. Esta sección sigue su protocolo (sección 5.2) con tres decisiones de diseño: las cadenas combinan la familia del modelo, el problema de aplicación y sinónimos en inglés y español; se agregan dos **cadenas de control** (solo la aplicación y solo la familia del modelo) que verifican que la sintaxis devuelve resultados y dimensionan cada literatura, de modo que un cero en las intersecciones no se confunda con una consulta mal escrita; y todo queda registrado por `scripts/revision_literatura.py` en `resultados/literatura/consultas.json`, con la fecha de cada consulta y los registros revisados, para que la revisión sea reproducible.

### 4.1 Protocolo

- **Bases consultadas:** OpenAlex (más de 250 millones de trabajos; búsqueda booleana en título y resumen), arXiv (preimpresiones de estadística y *machine learning*; búsqueda booleana en todos los campos), Crossref (registro de DOI de las editoriales; búsqueda por relevancia, sin operadores booleanos) y Semantic Scholar (búsqueda por relevancia en título y resumen). Scopus y Web of Science requieren acceso institucional y no tienen API pública; la misma lista de cadenas puede repetirse en ellas desde la biblioteca de la universidad.
- **Cadenas:** 2 de control y 8 de intersección (familia × aplicación), de las cuales una está en español.
- **Revisión:** se leen los primeros 25 registros de cada consulta (todos, si son menos). El tamizaje automático marca como **candidato** un registro cuyo título o resumen menciona a la vez el problema (telemercadeo bancario, depósito a término) y un mecanismo temporal (estado latente, Kalman, logística dinámica, deriva, cambio de prevalencia, flujo de datos, etiquetas diferidas); cada candidato se lee completo y se clasifica a mano como **relevante** si aplica un modelo de la misma familia al mismo problema.
- **Criterio de exclusión de la novedad:** existiría un antecedente si un trabajo aplicara a la predicción de la suscripción en telemercadeo bancario un modelo logístico con estado latente dinámico (intercepto o coeficientes que evolucionan en el tiempo y se actualizan con etiquetas nuevas).'''

S42_MD = r'''### 4.2 Matriz de novedad'''

S42_CODE = r'''# Clasificación manual de los candidatos del tamizaje, tras leer título, resumen y, cuando existe, el texto completo
CLASIFICACION = {
    "https://doi.org/10.1109/icscsa66339.2025.11170856": (
        True, "Chowdary et al. (2025): Gradient Boosting, Random Forest y XGBoost con SMOTE/ADASYN y actualización "
              "incremental sobre Bank Marketing; no modela un estado latente ni etiquetas diferidas"),
}
filas, candidatos = [], {}
for q in literatura:
    regs = q["registros"]
    rel = [CLASIFICACION.get(r["doi"], (False, ""))[0] for r in regs if r["candidato"]]
    for r in regs:
        if r["candidato"]:
            candidatos.setdefault(r["doi"] or r["titulo"], {**r, "base": q["base"]})
    if q["error"]:
        obs = "Consulta fallida: " + q["error"][:60]
    elif q.get("control"):
        obs = "Cadena de control: la sintaxis devuelve resultados"
    elif q["base"] in ("Crossref", "Semantic Scholar"):
        obs = "Búsqueda por relevancia: el total no filtra la intersección"
    elif q["resultados"] == 0:
        obs = "Sin resultados para la intersección"
    else:
        obs = "Ver candidatos"
    filas.append({"Base": q["base"], "Tipo": "Control" if q.get("control") else "Intersección",
                  "Cadena de búsqueda": q["cadena"], "Fecha (UTC)": q["fecha_utc"][:10],
                  "Result.": q["resultados"], "Revis.": q["revisados"], "Cand.": q["candidatos"], "Relev.": sum(rel),
                  "Observación": obs})
matriz = pd.DataFrame(filas)
tabla(matriz, dec=0, indice=False)

cand = pd.DataFrame([{"Base": c["base"], "Año": c["anio"], "Título": c["titulo"], "DOI o enlace": k,
                      "Relevante": "sí" if CLASIFICACION.get(k, (False,))[0] else "no",
                      "Clasificación": CLASIFICACION.get(k, (False, "Revisado: no aplica un modelo de estado latente "
                                                                    "al problema"))[1]} for k, c in candidatos.items()])
if len(cand):
    tabla(cand, indice=False)

inter = matriz[matriz["Tipo"] == "Intersección"]
bool_ = inter[inter["Base"].isin(["OpenAlex", "arXiv"]) & inter["Result."].notna()]
ctrl = matriz[(matriz["Tipo"] == "Control") & matriz["Base"].isin(["OpenAlex", "arXiv"])]
n_rev = int(matriz["Revis."].sum())
para_guion("literatura", {"consultas": len(matriz), "bases": int(matriz["Base"].nunique()), "revisados": n_rev,
                          "candidatos": int(len(cand)), "relevantes": int((cand["Relevante"] == "sí").sum()) if len(cand) else 0,
                          "fallidas": int(matriz["Observación"].str.startswith("Consulta fallida").sum())})
n_fallidas = int(matriz["Observación"].str.startswith("Consulta fallida").sum())
interpretar(
    f"Se hicieron {len(matriz)} consultas en {matriz['Base'].nunique()} bases ({n_fallidas} fallidas) y se revisaron "
    f"{num(n_rev)} registros. Las cadenas de control devuelven resultados en las bases booleanas (entre "
    f"{num(ctrl['Result.'].min())} y {num(ctrl['Result.'].max())} trabajos), mientras que las "
    f"{len(bool_)} intersecciones en OpenAlex y arXiv suman {num(bool_['Result.'].sum())} "
    f"resultado{'' if bool_['Result.'].sum() == 1 else 's'}. El tamizaje "
    f"marcó {len(cand)} candidato{'s' if len(cand) != 1 else ''} único{'s' if len(cand) != 1 else ''}, de los "
    f"cuales {int((cand['Relevante'] == 'sí').sum()) if len(cand) else 0} trata el mismo problema con un mecanismo "
    "temporal.",
    conectar("Lo anterior indica que", "Cada componente tiene su propia literatura (hay trabajos sobre telemercadeo "
             "bancario y sobre regresión logística dinámica), pero su intersección está prácticamente vacía. El único "
             "antecedente cercano en la aplicación adapta ensambles de árboles de forma incremental, que es una familia "
             "algorítmica distinta: no estima un nivel latente, no separa escalas de tiempo ni trata la demora de las "
             "etiquetas. Las búsquedas por relevancia (Crossref y Semantic Scholar) no permiten intersecciones "
             "booleanas, así que su aporte es la lectura de sus primeros registros, no el conteo."),
    conectar("Por tanto, corresponde", "Delimitar la novedad con precisión: no se reclama haber inventado los modelos "
             "de espacio de estados, sino una configuración algorítmica concreta (dos escalas, ajuste por bloques y "
             "filtro con etiquetas diferidas) y su primera aplicación a este problema, y diferenciarla de cada método "
             "afín en la sección 4.3."))'''

S43_MD = r'''### 4.3 Trabajos cercanos por familia algorítmica

La guía pide diferenciar la propuesta de métodos afines por familia, no solo por aplicación. La tabla compara el L2E con diez trabajos: dos de la misma aplicación y ocho de la misma familia (modelos lineales generalizados dinámicos, corrección del cambio de prevalencia y aprendizaje con deriva o etiquetas diferidas). Todas las referencias se verificaron por DOI o en sus actas (`scripts/verificar_referencias.py`).

| Trabajo | Familia | Qué conserva el L2E | Qué es nuevo en el L2E o qué limitación no resuelve el trabajo |
|---|---|---|---|
| Moro, Cortez y Rita (2014) | Modelos estáticos (red neuronal, SVM, árbol, logística) con ventana móvil, sobre telemercadeo bancario | El problema, las variables de contexto y la evaluación en orden temporal | La ventana móvil reajusta el modelo cada cierto tiempo y descarta el pasado; no estima el nivel de conversión ni lo actualiza día a día con etiquetas |
| Chowdary et al. (2025) | Ensambles de árboles con SMOTE y actualización incremental, sobre Bank Marketing | El uso de etiquetas nuevas para actualizar el modelo | Familia distinta (árboles, sin estado latente); no separa nivel y pendientes, no trata la demora de las etiquetas y su exactitud del 98 % sugiere el uso de `duration`, que aquí se excluye por fuga |
| West, Harrison y Migon (1985) | Modelos lineales generalizados dinámicos bayesianos | La idea de un estado que evoluciona y se actualiza con cada observación | Actualización conjugada aproximada observación por observación; el L2E usa Laplace por lote diario, dos escalas y un ajuste inicial por moda posterior |
| Fahrmeir (1992) | Moda posterior de MLG dinámicos con filtro de Kalman extendido | La estimación por moda posterior penalizada | Todos los coeficientes son dinámicos y el suavizado es O(T·p³); el L2E fija las pendientes en el ajuste y elimina el nivel con el complemento de Schur en O(T·p²) |
| McCormick et al. (2012) | Regresión logística dinámica con factor de olvido, una sola escala | El filtro de Laplace sobre los coeficientes de una logística | Una sola tasa de olvido para todos los coeficientes y etiquetas inmediatas; el L2E separa el nivel (rápido) de las pendientes (lentas), y la ablación de la sección 5.9 mide el costo de usar una sola escala |
| Kalman (1960); Durbin y Koopman (2012) | Filtrado y suavizado en espacio de estados | La recursión predicción-actualización | Formulados para observaciones gaussianas; el L2E trata observaciones de Bernoulli con aproximación de Laplace y probabilidad predictiva integrada (MacKay, 1992) |
| Saerens, Latinne y Decaestecker (2002) | Corrección del cambio de prevalencia sin etiquetas (EM) | La corrección del intercepto ante un cambio de la tasa base | Supone que p(x \| y) no cambia y no usa etiquetas; el L2E usa las etiquetas diferidas y además adapta las pendientes (deriva de concepto, sección 3.6.5 del tercer entregable) |
| Lipton, Wang y Smola (2018); Garg et al. (2020) | Estimación del cambio de prevalencia (BBSE, MLLS) | El enfoque de corregir la prevalencia del lote | Estiman una prevalencia fija por lote sin etiquetas; no hay dinámica entre lotes ni actualización de pendientes. MLLS se incluye como control en la sección 6.8 |
| Gama et al. (2014) | Adaptación a la deriva de concepto (ventanas, olvido, detectores) | La necesidad de olvidar el pasado de forma controlada | Las ventanas y los detectores son heurísticos; en el L2E el olvido sale de un modelo probabilístico, cuyas varianzas q_u y q_β se optimizan en la validación anidada |
| Grzenda, Gomes y Bifet (2020) | Evaluación con etiquetas diferidas en flujos de datos | El protocolo precuencial con demora de etiquetas | Es un protocolo de evaluación, no un modelo; el L2E lo incorpora en su regla de predicción (`retardo`) y en su prueba de ausencia de fuga |
| Recalibración mensual del intercepto (entregables 2 y 3) | Desplazamiento del logit para igualar la tasa del mes cerrado | La corrección del intercepto con datos recientes que sugirió el docente | Es el caso límite del filtro del L2E con varianza a priori infinita, lote mensual y retardo de un mes (propiedad P5, sección 5.4) |'''

S44_CODE = r'''n_bases = matriz.loc[~matriz["Observación"].str.startswith("Consulta fallida"), "Base"].nunique()
n_cadenas = len(matriz) // matriz["Base"].nunique()
MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
            "noviembre", "diciembre"]
_f = pd.Timestamp(matriz["Fecha (UTC)"].max())
fecha_rev = f"{_f.day} de {MESES_ES[_f.month - 1]} de {_f.year} (UTC)"
display(Markdown(f"""### 4.4 Declaración de novedad

Al mejor de nuestro conocimiento y según la búsqueda descrita ({n_bases} bases bibliográficas, {len(matriz)} consultas con {n_cadenas} cadenas en inglés y español, consulta del {fecha_rev}), **no se encontraron estudios que apliquen a la predicción de la suscripción a depósitos a término en campañas de telemercadeo bancario una regresión logística de estado latente con un nivel diario en paseo aleatorio, pendientes de deriva lenta y un filtro de Laplace con etiquetas diferidas**. Los trabajos más cercanos son, en la aplicación, Moro, Cortez y Rita (2014), que usan modelos estáticos con ventana móvil, y Chowdary et al. (2025), que actualizan ensambles de árboles de forma incremental; y, en la familia del modelo, McCormick et al. (2012), cuya regresión logística dinámica usa una sola escala de olvido y etiquetas inmediatas, y Fahrmeir (1992), cuya moda posterior trata todos los coeficientes como dinámicos sin la eliminación por bloques del nivel. La propuesta se diferencia de cada uno en la separación del nivel y las pendientes en dos escalas de tiempo, en el ajuste por bloques en O(T) y en la regla de predicción con etiquetas diferidas, cuya contribución medible se verifica con la ablación de la sección 5.9."""))'''
