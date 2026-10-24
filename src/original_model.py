"""Modelo original del proyecto: regresión logística de estado latente en dos escalas (L2E).

Formulación
-----------
Para el registro *i* contactado el día *t(i)*, con predictoras ``x_i`` (p columnas ya preprocesadas):

    P(y_i = 1 | θ_t) = σ(u_t + x_iᵀ β_t),                θ_t = (u_t, β_t)
    u_t = u_{t-1} + ε_t,   ε_t ~ N(0, q_u · Δ_t)          escala rápida: nivel de conversión del día
    β_t = β_{t-1} + ν_t,   ν_t ~ N(0, q_β · Δ_t · I_p)    escala lenta: deriva de las pendientes
    β_1 ~ N(0, λ⁻¹ I_p)

donde Δ_t es el número de días de contacto transcurridos desde el día anterior con datos. El algoritmo tiene dos fases:

1. **Ajuste por lotes** (``fit``): moda posterior del nivel ``u_1..u_T`` y de las pendientes en el periodo de
   entrenamiento, con β constante dentro de la ventana (aproximación de dos escalas, q_β ≪ q_u). El objetivo es
   estrictamente cóncavo y se maximiza con Newton-Raphson; el Hessiano tiene un bloque tridiagonal (el nivel) que se
   elimina con el complemento de Schur, de modo que cada iteración cuesta O(n·p² + T·p² + p³) y no O((T + p)³).
2. **Filtrado en línea** (``predict_proba_secuencial``): el estado (u, β) se propaga con el paseo aleatorio y, cuando se
   cierran las etiquetas de un día (con ``retardo`` días de demora), se actualiza con la aproximación de Laplace de la
   posterior (Newton sobre un objetivo estrictamente cóncavo de dimensión 1 + p). La probabilidad predictiva integra la
   incertidumbre del estado con la aproximación probit de MacKay (1992).

Casos límite: con q_u = q_β = 0 y sin actualización se recupera la logística con penalización L2; con q_β = q_u se
obtiene una regresión logística dinámica de una sola escala (McCormick et al., 2012); sin pendientes que se actualicen,
el filtro solo recalibra el intercepto día a día.

Solo depende de numpy y scipy para el cálculo numérico; scikit-learn aporta las clases base y la validación de entradas.

Copia de producción de ``bm.modelo_original`` (``cuarto_entregable/src/bm``): el algoritmo es idéntico línea por línea, lo
verifica ``tests/test_original_model.py``, y los artefactos de ``artifacts/l2e`` referencian esta clase, de modo que el
servidor del dashboard no depende del paquete de investigación.
"""

from collections import deque

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky_banded, solveh_banded
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

_PRECISION_NIVEL = 1e-6  # precisión a priori (difusa) del nivel, garantiza concavidad estricta en casos extremos
_ARMIJO, _REDUCCION = 0.25, 0.5  # búsqueda lineal con retroceso (Boyd y Vandenberghe, 2004, sec. 9.2)


def _logdet_banda(banda):
    """log|A| de una matriz tridiagonal simétrica definida positiva en forma de banda superior (Cholesky en O(T))."""
    if banda.shape[1] == 1:
        return float(np.log(banda[1, 0]))
    return float(2.0 * np.sum(np.log(cholesky_banded(banda)[-1])))


def _resolver_banda(banda, b):
    """Resuelve A x = b con A tridiagonal simétrica definida positiva en forma de banda (O(T) por columna)."""
    if banda.shape[1] == 1:
        return b / banda[1, 0]
    return solveh_banded(banda, b)


class LogisticaDosEscalas(ClassifierMixin, BaseEstimator):
    """Regresión logística de estado latente en dos escalas temporales, con filtro de Laplace para etiquetas diferidas.

    Parameters
    ----------
    lam : float, default=10.0
        Precisión a priori de las pendientes (penalización L2 sobre β en la suma de la log-verosimilitud).
    q_nivel : float, default=0.1
        Varianza diaria del paseo aleatorio del nivel u_t (escala rápida). En la fase 1 controla la suavidad de la
        trayectoria del nivel; en la fase 2, cuánto se le permite moverse entre días.
    q_pendiente : float, default=1e-4
        Varianza diaria del paseo aleatorio de cada pendiente (escala lenta). Solo actúa en la fase 2.
    retardo : int, default=1
        Días de contacto que tardan en conocerse las etiquetas: para predecir el día d solo se usan las de los días
        ≤ d − retardo.
    nivel_latente : bool, default=True
        Componente C1. Si es falso, el ajuste usa un solo intercepto para todo el entrenamiento (logística agrupada).
    actualizar_nivel : bool, default=True
        Componente C2. Si es falso, el nivel no se actualiza en línea.
    actualizar_pendientes : bool, default=True
        Componente C3. Si es falso, las pendientes quedan fijas en la estimación del ajuste.
    escala_unica : bool, default=False
        Si es verdadero, las pendientes derivan con la misma varianza que el nivel (q_β = q_u): regresión logística
        dinámica de una sola escala, usada en la ablación.
    integrar_incertidumbre : bool, default=True
        Componente C4. Si es falso, la probabilidad es σ(media del logit), sin integrar la incertidumbre del estado.
    semivida : float, optional
        Mejora M3. Semivida, en días de contacto, de los pesos por recencia de la fase 1; ``None`` no pondera.
    q_nivel_empirico : bool, default=False
        Mejora M2. Si es verdadero, q_u se estima por Bayes empírico (máxima evidencia de Laplace) en cada ajuste.
    max_iter : int, default=100
        Iteraciones máximas de Newton en el ajuste.
    tol : float, default=1e-9
        Tolerancia sobre el decremento de Newton, λ²/2, para declarar convergencia.
    col_tiempo : int, default=-1
        Columna de ``X`` con el identificador entero del día de contacto (no se usa como predictora).

    Attributes
    ----------
    classes_ : ndarray de forma (2,)
    coef_ : ndarray de forma (p,)
        Pendientes estimadas en el ajuste (moda posterior).
    nivel_ : ndarray de forma (T,)
        Trayectoria suavizada del nivel en los días de entrenamiento.
    dias_ : ndarray de forma (T,)
        Días de entrenamiento con al menos un registro.
    estado_media_, estado_cov_ : ndarray
        Media (1 + p) y covarianza (1 + p) × (1 + p) de la posterior de Laplace de (u_T, β) al cierre del ajuste.
    dia_final_ : int
        Último día asimilado por el estado.
    objetivo_ : list of float
        Valor del objetivo penalizado en cada iteración de Newton (no decreciente por construcción).
    n_iter_ : int
    convergio_ : bool
    q_nivel_ : float
        q_u usado en el ajuste y en el filtrado (igual a ``q_nivel`` salvo con ``q_nivel_empirico``).
    """

    def __init__(self, lam=10.0, q_nivel=0.1, q_pendiente=1e-4, retardo=1, nivel_latente=True,
                 actualizar_nivel=True, actualizar_pendientes=True, escala_unica=False,
                 integrar_incertidumbre=True, semivida=None, q_nivel_empirico=False, max_iter=100, tol=1e-9,
                 col_tiempo=-1):
        self.lam = lam
        self.q_nivel = q_nivel
        self.q_pendiente = q_pendiente
        self.retardo = retardo
        self.nivel_latente = nivel_latente
        self.actualizar_nivel = actualizar_nivel
        self.actualizar_pendientes = actualizar_pendientes
        self.escala_unica = escala_unica
        self.integrar_incertidumbre = integrar_incertidumbre
        self.semivida = semivida
        self.q_nivel_empirico = q_nivel_empirico
        self.max_iter = max_iter
        self.tol = tol
        self.col_tiempo = col_tiempo

    # ------------------------------------------------------------------------------------------ validación

    def _validar_hiperparametros(self):
        if not np.isfinite(self.lam) or self.lam <= 0:
            raise ValueError(f"lam debe ser positivo y finito; se recibió {self.lam!r}")
        for nombre in ("q_nivel", "q_pendiente"):
            v = getattr(self, nombre)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{nombre} debe ser no negativo y finito; se recibió {v!r}")
        if self.nivel_latente and self.q_nivel <= 0:
            raise ValueError("Con nivel_latente=True, q_nivel debe ser positivo (q_nivel = 0 equivale a "
                             "nivel_latente=False)")
        if int(self.retardo) != self.retardo or self.retardo < 1:
            raise ValueError(f"retardo debe ser un entero ≥ 1; se recibió {self.retardo!r}")
        if int(self.max_iter) != self.max_iter or self.max_iter < 1:
            raise ValueError(f"max_iter debe ser un entero ≥ 1; se recibió {self.max_iter!r}")
        if self.semivida is not None and (not np.isfinite(self.semivida) or self.semivida <= 0):
            raise ValueError(f"semivida debe ser None o un número positivo; se recibió {self.semivida!r}")

    def _separar(self, X):
        """Separa la columna de tiempo (entera) de las predictoras."""
        if X.shape[1] < 2:
            raise ValueError("X debe tener al menos una predictora además de la columna de tiempo")
        col = self.col_tiempo % X.shape[1]
        t = X[:, col]
        if not np.allclose(t, np.round(t)):
            raise ValueError("La columna de tiempo debe contener identificadores enteros de día")
        return np.round(t).astype(np.int64), np.delete(X, col, axis=1)

    def _ruido(self):
        """Varianza diaria del proceso para (u, β): vector de longitud 1 + p."""
        q_u = getattr(self, "q_nivel_", self.q_nivel)  # el estimado por Bayes empírico, si existe
        q_b = q_u if self.escala_unica else self.q_pendiente
        q = np.full(self.n_predictoras_ + 1, q_b if self.actualizar_pendientes else 0.0)
        q[0] = q_u if self.actualizar_nivel else 0.0
        return q

    # ------------------------------------------------------------------------------------------- fase 1

    def fit(self, X, y):
        """Ajusta la moda posterior del nivel latente y de las pendientes con Newton-Raphson.

        Con ``semivida``, cada registro pondera su log-verosimilitud por 0,5^((t_max − t) / semivida), de modo que los
        días recientes pesan más en las pendientes (deriva de concepto). Con ``q_nivel_empirico=True``, q_u se estima
        maximizando la evidencia de Laplace, log p(y | q_u) ≈ J(θ̂) + ½ log|Ω| − ½ log|H|, en lugar de fijarlo; el
        valor estimado queda en ``q_nivel_`` y se usa también en la fase 2.

        Parameters
        ----------
        X : array-like de forma (n, p + 1)
            Predictoras preprocesadas y, en ``col_tiempo``, el día de contacto (entero no decreciente en el tiempo).
        y : array-like de forma (n,)
            Etiquetas binarias.

        Returns
        -------
        self
        """
        self._validar_hiperparametros()
        X, y = check_X_y(X, y, dtype=np.float64, ensure_all_finite=True)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Se requieren exactamente dos clases; se encontraron {len(self.classes_)}")
        y01 = (y == self.classes_[1]).astype(np.float64)
        t, F = self._separar(X)
        self.n_features_in_ = X.shape[1]
        self.n_predictoras_ = F.shape[1]
        n = len(F)
        dias, idx = np.unique(t, return_inverse=True)
        if not self.nivel_latente:
            idx = np.zeros(n, dtype=np.int64)
            dias = dias[-1:]
        pesos = np.ones(n) if self.semivida is None else 0.5 ** ((t.max() - t) / float(self.semivida))
        datos = (F, idx, y01, pesos, np.diff(dias).astype(np.float64))
        lam = float(self.lam)
        if self.q_nivel_empirico and self.nivel_latente and len(dias) > 1:
            q, self.evidencia_ = self._q_bayes_empirico(datos, lam)
        else:
            q = float(self.q_nivel)
        self.q_nivel_ = q
        u, b, prec, (self.objetivo_, self.n_iter_, self.convergio_) = self._newton(datos, lam, q)
        self.coef_ = b
        self.nivel_ = u
        self.dias_ = dias
        self.dia_final_ = int(np.max(t))
        self.estado_media_, self.estado_cov_ = self._posterior_final(u, b, idx, F, prec, lam, pesos)
        return self

    def _newton(self, datos, lam, q, inicio=None):
        """Newton-Raphson amortiguado sobre la ecuación (3) para un q_u dado; devuelve (u, β, precisiones, info)."""
        F, idx, y01, pesos, delta = datos
        T, p = len(delta) + 1, F.shape[1]
        # Precisión del paseo aleatorio entre días consecutivos: 1 / (q_u · Δ_k)
        prec = 1.0 / (q * delta) if T > 1 else np.zeros(0)

        def objetivo(u, b):
            eta = u[idx] + F @ b
            ll = np.sum(pesos * (y01 * eta - np.logaddexp(0.0, eta)))
            return ll - 0.5 * np.sum(prec * np.diff(u) ** 2) - 0.5 * _PRECISION_NIVEL * u @ u - 0.5 * lam * b @ b

        if inicio is None:
            tasa = np.average(y01, weights=pesos)
            u, b = np.full(T, np.log((tasa + 1e-3) / (1 - tasa + 1e-3))), np.zeros(p)
        else:
            u, b = inicio[0].copy(), inicio[1].copy()
        f = objetivo(u, b)
        objetivos, convergio = [float(f)], False
        for it in range(1, int(self.max_iter) + 1):
            eta = u[idx] + F @ b
            pi = expit(eta)
            r = pesos * (y01 - pi)
            w = pesos * pi * (1.0 - pi)
            # Gradiente
            g_u = np.bincount(idx, r, T) - _PRECISION_NIVEL * u
            if T > 1:
                d = prec * np.diff(u)
                g_u[:-1] += d
                g_u[1:] -= d
            g_b = F.T @ r - lam * b
            # Hessiano negativo en bloques: A (T × T, tridiagonal), B (T × p), C (p × p)
            banda = self._banda(np.bincount(idx, w, T), prec)
            B = np.zeros((T, p))
            np.add.at(B, idx, w[:, None] * F)
            C = (F * w[:, None]).T @ F + lam * np.eye(p)
            Ainv = _resolver_banda(banda, np.column_stack([B, g_u]))
            AinvB, Ainv_g = Ainv[:, :p], Ainv[:, p]
            S = C - B.T @ AinvB  # complemento de Schur, simétrico definido positivo
            cS = cho_factor(S)
            d_b = cho_solve(cS, g_b - B.T @ Ainv_g)
            d_u = Ainv_g - AinvB @ d_b
            decremento = float(g_u @ d_u + g_b @ d_b)  # λ² de Newton = gᵀ H⁻¹ g
            if decremento / 2.0 <= self.tol:
                convergio = True
                break
            paso = 1.0
            while True:
                f_nuevo = objetivo(u + paso * d_u, b + paso * d_b)
                if f_nuevo >= f + _ARMIJO * paso * decremento or paso < 1e-10:
                    break
                paso *= _REDUCCION
            u, b, f = u + paso * d_u, b + paso * d_b, f_nuevo
            objetivos.append(float(f))
        return u, b, prec, (objetivos, it, convergio)

    def _log_evidencia(self, datos, lam, u, b, prec):
        """Evidencia de Laplace: J(θ̂) + ½ log|Ω| − ½ log|H|, con Ω la precisión a priori y H el Hessiano negativo.

        Los determinantes se obtienen sin matrices densas de tamaño T: log|A| y log|D + εI| con Cholesky por bandas y
        log|H| = log|A| + log|S| por el complemento de Schur.
        """
        F, idx, y01, pesos, _ = datos
        T, p = len(u), len(b)
        eta = u[idx] + F @ b
        pi = expit(eta)
        w = pesos * pi * (1.0 - pi)
        J = (np.sum(pesos * (y01 * eta - np.logaddexp(0.0, eta))) - 0.5 * np.sum(prec * np.diff(u) ** 2)
             - 0.5 * _PRECISION_NIVEL * u @ u - 0.5 * lam * b @ b)
        logdet_omega = _logdet_banda(self._banda(np.zeros(T), prec)) + p * np.log(lam)
        banda = self._banda(np.bincount(idx, w, T), prec)
        B = np.zeros((T, p))
        np.add.at(B, idx, w[:, None] * F)
        S = (F * w[:, None]).T @ F + lam * np.eye(p) - B.T @ _resolver_banda(banda, B)
        logdet_h = _logdet_banda(banda) + 2.0 * np.sum(np.log(np.diag(cho_factor(S)[0])))
        return float(J + 0.5 * logdet_omega - 0.5 * logdet_h)

    def _q_bayes_empirico(self, datos, lam, limites=(1e-4, 2.0)):
        """q_u que maximiza la evidencia de Laplace (búsqueda acotada en escala logarítmica, con arranque en caliente)."""
        from scipy.optimize import minimize_scalar

        previo = {"inicio": None}

        def menos_evidencia(log_q):
            u, b, prec, _ = self._newton(datos, lam, float(np.exp(log_q)), previo["inicio"])
            previo["inicio"] = (u, b)
            return -self._log_evidencia(datos, lam, u, b, prec)

        r = minimize_scalar(menos_evidencia, bounds=tuple(np.log(limites)), method="bounded", options={"xatol": 0.1})
        return float(np.exp(r.x)), float(-r.fun)

    @staticmethod
    def _banda(w_dia, prec):
        """Matriz tridiagonal A = diag(W_t) + penalización del paseo aleatorio, en forma de banda superior."""
        T = len(w_dia)
        diag = w_dia + _PRECISION_NIVEL
        if T > 1:
            diag[:-1] += prec
            diag[1:] += prec
        banda = np.zeros((2, T))
        banda[1] = diag
        banda[0, 1:] = -prec
        return banda

    def _posterior_final(self, u, b, idx, F, prec, lam, pesos=None):
        """Media y covarianza de Laplace de (u_T, β) al cierre del entrenamiento, sin invertir el Hessiano completo.

        Con H = [[A, B], [Bᵀ, C]] y S = C − Bᵀ A⁻¹ B: Cov(β) = S⁻¹, Cov(u_T, β) = −vᵀB S⁻¹ y
        Var(u_T) = v_T + vᵀB S⁻¹ Bᵀv, donde v = A⁻¹ e_T (una sola resolución tridiagonal).
        """
        T, p = len(u), len(b)
        pi = expit(u[idx] + F @ b)
        w = pi * (1.0 - pi) * (1.0 if pesos is None else pesos)
        banda = self._banda(np.bincount(idx, w, T), prec)
        B = np.zeros((T, p))
        np.add.at(B, idx, w[:, None] * F)
        S = (F * w[:, None]).T @ F + lam * np.eye(p) - B.T @ _resolver_banda(banda, B)
        e_T = np.zeros(T)
        e_T[-1] = 1.0
        v = _resolver_banda(banda, e_T)
        g = B.T @ v
        Sinv = np.linalg.inv(S)
        Sinv = (Sinv + Sinv.T) / 2.0
        cov = np.empty((p + 1, p + 1))
        cov[0, 0] = v[-1] + g @ Sinv @ g
        cov[0, 1:] = cov[1:, 0] = -(Sinv @ g)
        cov[1:, 1:] = Sinv
        return np.r_[u[-1], b], cov

    # ------------------------------------------------------------------------------------------- fase 2

    def _probabilidad(self, F, m, P):
        """Probabilidad predictiva σ(μ / sqrt(1 + π s² / 8)) con μ = m₀ + F m₁ y s² = zᵀ P z, z = (1, x)."""
        mu = m[0] + F @ m[1:]
        if not self.integrar_incertidumbre:
            return expit(mu)
        s2 = P[0, 0] + 2.0 * F @ P[0, 1:] + np.einsum("ij,jk,ik->i", F, P[1:, 1:], F)
        return expit(mu / np.sqrt(1.0 + np.pi * np.maximum(s2, 0.0) / 8.0))

    def _asimilar(self, m, P, F, y01, q, delta):
        """Propaga el estado ``delta`` días y lo actualiza con las etiquetas de un día (aproximación de Laplace).

        Maximiza Σ_i [y_i η_i − log(1 + e^{η_i})] − ½ (θ − m⁻)ᵀ (P⁻)⁻¹ (θ − m⁻), con m⁻ = m y P⁻ = P + diag(q)·Δ,
        sobre los componentes activos de θ (nivel y, si corresponde, pendientes); los inactivos quedan fijos.
        """
        P_prev = P + np.diag(q * delta)
        activos = []
        if self.actualizar_nivel or self.actualizar_pendientes:
            activos.append(0)
        if self.actualizar_pendientes:
            activos.extend(range(1, len(m)))
        if not activos:
            return m, P_prev
        a = np.asarray(activos)
        Z = np.column_stack([np.ones(len(F)), F])
        offset = Z[:, np.setdiff1d(np.arange(len(m)), a)] @ m[np.setdiff1d(np.arange(len(m)), a)]
        Za = Z[:, a]
        m_prev = m[a]
        Pa = P_prev[np.ix_(a, a)]
        prec_prev = cho_solve(cho_factor(Pa), np.eye(len(a)))
        th = m_prev.copy()
        for _ in range(50):
            pi = expit(offset + Za @ th)
            g = Za.T @ (y01 - pi) - prec_prev @ (th - m_prev)
            H = (Za * (pi * (1.0 - pi))[:, None]).T @ Za + prec_prev
            paso = cho_solve(cho_factor(H), g)
            th = th + paso
            if np.max(np.abs(paso)) < 1e-10:
                break
        pi = expit(offset + Za @ th)
        H = (Za * (pi * (1.0 - pi))[:, None]).T @ Za + prec_prev
        P_post = P_prev.copy()
        cov_a = cho_solve(cho_factor(H), np.eye(len(a)))
        P_post[np.ix_(a, a)] = (cov_a + cov_a.T) / 2.0
        m_post = m.copy()
        m_post[a] = th
        return m_post, P_post

    def predict_proba_secuencial(self, X, y, retardo=None, devolver_traza=False):
        """Probabilidades día a día, asimilando las etiquetas de los días cerrados (evaluación precuencial).

        Para cada día d en orden cronológico: (1) se asimilan, en orden, las etiquetas de los días ≤ d − retardo que aún
        no se hayan usado; (2) se predicen los registros del día d con el estado propagado hasta d. Las etiquetas de un
        día nunca influyen en sus propias predicciones ni en las de los ``retardo − 1`` días siguientes. El estado
        ajustado del estimador no se modifica.

        Parameters
        ----------
        X : array-like de forma (n, p + 1)
        y : array-like de forma (n,)
            Etiquetas de los mismos registros, que solo se usan con la demora indicada.
        retardo : int, optional
            Sustituye a ``self.retardo``.
        devolver_traza : bool
            Si es verdadero, también devuelve la trayectoria del estado (un registro por día predicho).

        Returns
        -------
        ndarray de forma (n, 2), y opcionalmente un dict con la traza (día, registros, media y varianza del nivel,
        pendientes vigentes).
        """
        check_is_fitted(self, "estado_media_")
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("X e y deben tener el mismo número de registros")
        if not np.isin(y, self.classes_).all():
            raise ValueError("y contiene etiquetas que no se vieron en el ajuste")
        retardo = self.retardo if retardo is None else retardo
        if int(retardo) != retardo or retardo < 1:
            raise ValueError("retardo debe ser un entero ≥ 1")
        y01 = (y == self.classes_[1]).astype(np.float64)
        t, F = self._separar(X)
        q = self._ruido()
        m, P, t_ult = self.estado_media_.copy(), self.estado_cov_.copy(), self.dia_final_
        prob = np.empty(len(X))
        pendientes = deque()
        traza = {"dia": [], "registros": [], "nivel": [], "var_nivel": [], "pendientes": []}
        for d in np.unique(t):
            while pendientes and pendientes[0] <= d - retardo:
                dd = pendientes.popleft()
                fila = t == dd
                m, P = self._asimilar(m, P, F[fila], y01[fila], q, max(dd - t_ult, 0))
                t_ult = max(t_ult, dd)
            fila = t == d
            P_d = P + np.diag(q * max(d - t_ult, 0))
            prob[fila] = self._probabilidad(F[fila], m, P_d)
            pendientes.append(d)
            if devolver_traza:
                traza["dia"].append(int(d))
                traza["registros"].append(int(fila.sum()))
                traza["nivel"].append(float(m[0]))
                traza["var_nivel"].append(float(P_d[0, 0]))
                traza["pendientes"].append(m[1:].copy())
        salida = np.column_stack([1.0 - prob, prob])
        if devolver_traza:
            traza["pendientes"] = np.asarray(traza["pendientes"])
            return salida, {k: np.asarray(v) for k, v in traza.items()}
        return salida

    def actualizar(self, X, y):
        """Asimila en el estado del estimador las etiquetas de uno o más días ya cerrados (uso en producción).

        Los días se procesan en orden cronológico; el estado y ``dia_final_`` quedan modificados.
        """
        check_is_fitted(self, "estado_media_")
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        y01 = (np.asarray(y) == self.classes_[1]).astype(np.float64)
        t, F = self._separar(X)
        q = self._ruido()
        for d in np.unique(t):
            fila = t == d
            self.estado_media_, self.estado_cov_ = self._asimilar(self.estado_media_, self.estado_cov_, F[fila],
                                                                  y01[fila], q, max(d - self.dia_final_, 0))
            self.dia_final_ = max(self.dia_final_, int(d))
        return self

    # ---------------------------------------------------------------------------------- interfaz estándar

    def decision_function(self, X):
        """Media del logit con el estado vigente (sin asimilar etiquetas nuevas)."""
        check_is_fitted(self, "estado_media_")
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        _, F = self._separar(X)
        return self.estado_media_[0] + F @ self.estado_media_[1:]

    def predict_proba(self, X):
        """Probabilidades con el estado vigente propagado hasta el día de cada registro, sin usar etiquetas nuevas."""
        check_is_fitted(self, "estado_media_")
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tiene {X.shape[1]} columnas; el modelo se ajustó con {self.n_features_in_}")
        t, F = self._separar(X)
        q = self._ruido()
        prob = np.empty(len(X))
        for d in np.unique(t):
            fila = t == d
            prob[fila] = self._probabilidad(F[fila], self.estado_media_,
                                            self.estado_cov_ + np.diag(q * max(d - self.dia_final_, 0)))
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X):
        """Clase con probabilidad estimada ≥ 0,5."""
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def filtrar_nivel(puntaje, dias, y, q, retardo=1, es_probabilidad=True, var_inicial=1.0):
    """Componente C2 aislado: filtra un nivel latente sobre el puntaje de cualquier clasificador.

    El logit del clasificador actúa como término fijo (o la función de decisión, si no es una probabilidad) y solo se
    estima el nivel, con el mismo paseo aleatorio y la misma actualización de Laplace del modelo L2E. Se usa como
    control con igual información: cuánto ganan los modelos de referencia si reciben las mismas etiquetas diferidas.
    """
    s = np.asarray(puntaje, dtype=np.float64)
    o = np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1)) if es_probabilidad else s
    dias = np.asarray(dias)
    y = np.asarray(y, dtype=np.float64)
    m, P, t_ult = 0.0, float(var_inicial), int(dias.min()) - 1
    salida = np.empty(len(s))
    pendientes = deque()
    for d in np.unique(dias):
        while pendientes and pendientes[0] <= d - retardo:
            dd = pendientes.popleft()
            fila = dias == dd
            P_prev = P + q * (dd - t_ult)
            a = m
            for _ in range(50):
                pi = expit(a + o[fila])
                paso = (np.sum(y[fila] - pi) - (a - m) / P_prev) / (np.sum(pi * (1 - pi)) + 1 / P_prev)
                a += paso
                if abs(paso) < 1e-10:
                    break
            pi = expit(a + o[fila])
            P = 1.0 / (np.sum(pi * (1 - pi)) + 1.0 / P_prev)
            m, t_ult = a, dd
        fila = dias == d
        salida[fila] = expit(m + o[fila])
        pendientes.append(d)
    return salida
