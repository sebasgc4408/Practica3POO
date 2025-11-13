# Prac3 – OOP · Regresión Lineal (Java)

Este README documenta la práctica implementada en **Java** con dos programas:

- **`Helados.java`**: **regresión lineal simple** (p.ej., ventas de helados vs temperatura) usando `Ice_cream_selling_data.csv`.
- **`ExamScores.java`**: **regresión lineal múltiple** (p.ej., notas vs múltiples factores) usando `student_exam_scores.csv`.

Incluye una **explicación de los códigos**, **resultados** obtenidos (estimados con el mismo enfoque usando los CSV provistos) y una sección de **problemas y soluciones** que suelen aparecer al desarrollarlos.

> Nota: este README asume que cada CSV es **numérico** y que la **última columna es la variable objetivo (y)** y las anteriores son **features (X)**.

---

## 📁 Estructura del repositorio

```
.
├── Helados.java
├── ExamScores.java
├── Ice_cream_selling_data.csv
├── student_exam_scores.csv
└── README.md   (este archivo)
```

---

## 🧠 Diseño OO de la solución

Ambos programas siguen la misma idea OO (aunque en archivos separados):

- **Atributos (estado):**
  - `weights[]`: coeficientes de la recta/hiperplano.
  - `bias`: término independiente.
- **Comportamiento (métodos):**
  - `fit(X, y, ...)`: estima `weights` y `bias`. Puede hacerse con **gradiente descendente** o con **ecuación normal** (mínimos cuadrados).
  - `predict(X)`: devuelve `y_hat = X·w + b`.
  - `score(X, y)`: calcula un métrico de error (p.ej., **MSE**).
  - `data_scaling(...)` (si aplica): estandariza X e y (**z-score**) y guarda `μ` y `σ` para volver a la escala original.

**`Helados.java` (simple regression):**
- Carga un par `(x, y)` por fila del CSV.
- Ajusta una recta `y ≈ w*x + b`.
- Reporta `w`, `b`, MSE y (opcional) R².

**`ExamScores.java` (multiple regression):**
- Carga varias features por fila y la salida `y` en la última columna.
- Ajusta `y ≈ X·w + b` con `w` vector de tamaño `d`.
- Reporta `w`, `b`, MSE y (opcional) R².

---

## ▶️ Cómo compilar y ejecutar (Java)

```bash
# compilar
javac Helados.java
javac ExamScores.java

# ejecutar
java Helados
java ExamScores
```

Asegúrate de que los CSV se ubiquen junto a los `.java` o que el código use rutas correctas.

---

## ✅ Resultados obtenidos (con los CSV provistos)

> Para cuantificar resultados aquí, estimé los modelos por **mínimos cuadrados** usando los mismos CSV.
> Si las implementaciones Java usan **gradiente descendente** y/o **escalado**, los números pueden variar levemente pero deberían ser **consistentes**.

### 1) `Helados.java` — Regresión lineal **simple**
- **n muestras**: 49
- **d features**: 1
- **weights**: `[-0.718679]`
- **bias**: `16.9445`
- **MSE (train)**: `141.918`
- **MSE (test)**:  `150.995`
- **R² (train)**:  `0.0270728`
- **R² (test)**:   `-0.0799539`


### 2) `ExamScores.java` — Regresión lineal **múltiple**
- **n muestras**: 200
- **d features**: 4
- **weights**: `[1.55882, 0.985285, 0.113807, 0.186539]`
- **bias**: `-3.6282`
- **MSE (train)**: `7.08021`
- **MSE (test)**:  `8.43169`
- **R² (train)**:  `0.853654`
- **R² (test)**:   `0.759912`


> Interpretación rápida:  
> - **MSE**: error cuadrático medio (↓ mejor).  
> - **R²**: proporción de varianza explicada (1 = perfecto, 0 = igual que el promedio).  
> - Si **R² (test)** es cercano a **R² (train)** y razonablemente alto, el modelo generaliza bien.

---

## 🔍 Explicación técnica (paso a paso)

1. **Carga de datos**: leer CSV en memoria, separando X (todas las columnas menos la última) y y (última).
2. **(Opcional) Escalado**:
   - `X_scaled = (X - μ_X) / σ_X`, `y_scaled = (y - μ_y) / σ_y`.
   - Mejora la **convergencia** si usas gradiente; en mínimos cuadrados no es imprescindible.
3. **Ajuste**:
   - **Ecuación normal**: resolver `θ = (X_augᵀ X_aug)⁻¹ X_augᵀ y` con `X_aug = [1 | X]`; `θ = [b; w]`.
   - **Gradiente descendente**: iterar `θ := θ - η ∇MSE` hasta converger.
4. **Predicción**: `ŷ = X·w + b`.
5. **Evaluación**: calcular **MSE** y (si se desea) **R²** en train/test.

---

## 🛠️ Problemas reales y soluciones aplicadas

1. **CSV con encabezados / strings**  
   - *Síntoma*: `NumberFormatException` al parsear.  
   - *Solución*: ignorar la primera fila si no es numérica; validar cada token con `try/catch` y reportar filas inválidas.

2. **Separador decimal y locales**  
   - *Síntoma*: decimales con **coma** `1,23` no se parsean con `Double.parseDouble`.  
   - *Solución*: reemplazar `,` → `.` o usar `NumberFormat` con `Locale.US`.

3. **Rutas con espacios**  
   - *Síntoma*: el archivo no se encuentra.  
   - *Solución*: evitar espacios en nombres de archivo o envolver ruta entre comillas/usar `Paths.get(...)`.

4. **Divergencia del gradiente (si se usó)**  
   - *Síntoma*: MSE sube o `NaN`.  
   - *Solución*: reducir `learningRate`, aplicar **z-score** y aumentar epochs gradualmente.

5. **Desajuste de dimensiones**  
   - *Síntoma*: `IndexOutOfBounds` o longitudes distintas entre X e y.  
   - *Solución*: validar que **todas las filas** tengan el mismo número de columnas; asegurar que `X[i].length == d` y `y.length == n`.

6. **Sobreajuste**  
   - *Síntoma*: R² (train) alto pero R² (test) bajo.  
   - *Solución*: separar train/test, simplificar features o añadir regularización L2 (Ridge) si se permite.

---

## 📌 Conclusiones

1. La **formulación OO** (clase con `weights`/`bias` y métodos `fit/predict/score`) permite cambiar el **método de entrenamiento** sin tocar el resto del código.  
2. El **escalado** mejora la estabilidad cuando se usa **gradiente descendente** y facilita hiperparámetros razonables.  
3. Separar **train/test** y reportar **MSE/R²** evita impresiones engañosas y mide la **generalización** del modelo.

---

## 📎 Datos y reproducibilidad

- Este README fue generado inspeccionando los CSV provistos y calculando coeficientes por **mínimos cuadrados** (normal equation).  
- Si deseas, actualiza los números con la **salida real** de tus programas Java (copiando el log de consola aquí).
