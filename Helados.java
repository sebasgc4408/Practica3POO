import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;


// Operaciones de matrices
class Mat {
    // Transpone una matriz
    static double[][] T(double[][] A) {
        int n = A.length, m = A[0].length;
        double[][] R = new double[m][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) R[j][i] = A[i][j];
        return R;
    }

    // Multiplica matrices
    static double[][] mm(double[][] A, double[][] B) {
        int n = A.length, m = A[0].length, p = B[0].length;
        double[][] R = new double[n][p];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < m; k++)
                for (int j = 0; j < p; j++)
                    R[i][j] += A[i][k] * B[k][j];
        return R;
    }

    // Multiplica matriz por vector
    static double[] mv(double[][] A, double[] v) {
        int n = A.length, m = A[0].length;
        double[] r = new double[n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                r[i] += A[i][j] * v[j];
        return r;
    }

    // Añade columna de unos
    static double[][] addOnes(double[][] X) {
        int n = X.length, m = X[0].length;
        double[][] R = new double[n][m + 1];
        for (int i = 0; i < n; i++) {
            R[i][0] = 1.0;
            System.arraycopy(X[i], 0, R[i], 1, m);
        }
        return R;
    }

    // Copia matriz identidad
    static double[][] eye(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++) I[i][i] = 1.0;
        return I;
    }

    // Inversa por Gauss-Jordan
    static double[][] inv(double[][] A) {
        int n = A.length;
        double[][] M = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, M[i], 0, n);
            System.arraycopy(eye(n)[i], 0, M[i], n, n);
        }
        for (int col = 0; col < n; col++) {
            int piv = col;
            for (int r = col + 1; r < n; r++)
                if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
            double[] tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
            double div = M[col][col];
            for (int j = 0; j < 2 * n; j++) M[col][j] /= div;
            for (int r = 0; r < n; r++)
                if (r != col) {
                    double f = M[r][col];
                    for (int j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
                }
        }
        double[][] R = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(M[i], n, R[i], 0, n);
        return R;
    }
}

// Escalador simple
class Scaler {
    private double[] mean;
    private double[] std;

    public void fit(double[][] X) { // Calcula medias y desviaciones
        int n = X.length, m = X[0].length;
        mean = new double[m]; std = new double[m];
        for (int j = 0; j < m; j++) {
            double s = 0;
            for (int i = 0; i < n; i++) s += X[i][j];
            mean[j] = s / n;
            double v = 0;
            for (int i = 0; i < n; i++) {
                double d = X[i][j] - mean[j];
                v += d * d;
            }
            std[j] = Math.sqrt(v / (n - 1));
            if (std[j] == 0) std[j] = 1.0;
        }
    }

    public double[][] transform(double[][] X) { // Aplica la escala
        int n = X.length, m = X[0].length;
        double[][] R = new double[n][m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                R[i][j] = (X[i][j] - mean[j]) / std[j];
        return R;
    }
}

// Modelo de regresión lineal
class LinearRegression {
    private double[] weights; // Pesos
    private double bias;      // Intercepto
    private boolean fitIntercept;

    public LinearRegression(boolean fitIntercept) { // Constructor
        this.fitIntercept = fitIntercept;
    }

    public void dataScaling(double[][] X, Scaler scaler) { // Escala datos
        scaler.fit(X);
    }

    public void fit(double[][] X, double[] y, boolean scale) { // Entrena
        double[][] Xproc = X;
        if (scale) {
            Scaler s = new Scaler();
            dataScaling(X, s);
            Xproc = s.transform(X);
        }
        double[][] Xb = fitIntercept ? Mat.addOnes(Xproc) : Xproc;
        double[][] Xt = Mat.T(Xb);
        double[][] XtX = Mat.mm(Xt, Xb);
        double[][] XtXInv = Mat.inv(XtX);
        double[] Xty = Mat.mv(Xt, y);
        double[] wFull = Mat.mv(XtXInv, Xty);
        if (fitIntercept) {
            bias = wFull[0];
            weights = Arrays.copyOfRange(wFull, 1, wFull.length);
        } else {
            bias = 0.0;
            weights = wFull;
        }
    }

    public double[] predict(double[][] X) { // Predice
        int n = X.length;
        double[] yhat = new double[n];
        for (int i = 0; i < n; i++) {
            double s = bias;
            for (int j = 0; j < weights.length; j++) s += weights[j] * X[i][j];
            yhat[i] = s;
        }
        return yhat;
    }

    public double score(double[] y, double[] yhat) { // Calcula R²
        double mean = 0;
        for (double v : y) mean += v;
        mean /= y.length;
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < y.length; i++) {
            ssTot += Math.pow(y[i] - mean, 2);
            ssRes += Math.pow(y[i] - yhat[i], 2);
        }
        return 1 - ssRes / ssTot;
    }

    public double[] getWeights() { return weights; } // Devuelve pesos
    public double getBias() { return bias; }         // Devuelve bias
}

// Clase principal
public class Helados {
    public static void main(String[] args) {
        String csvPath = "Ice_cream_selling_data.csv"; // nombre del archivo CSV

        // Listas temporales para guardar datos leídos del CSV
        java.util.List<double[]> filas = new java.util.ArrayList<>();

        // 1) Leer el CSV: Temperature (°C), Ice Cream Sales (units)
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String linea = br.readLine(); // Leer y saltar la cabecera

            while ((linea = br.readLine()) != null) {
                if (linea.trim().isEmpty()) continue; // Saltar líneas vacías

                String[] partes = linea.split(",");
                if (partes.length < 2) continue; // Por seguridad

                double temp = Double.parseDouble(partes[0]);  // Columna Temperature (°C)
                double ventas = Double.parseDouble(partes[1]); // Columna Ice Cream Sales (units)

                filas.add(new double[]{ temp, ventas });
            }
        } catch (IOException e) {
            System.err.println("Error leyendo el CSV: " + e.getMessage());
            return; // Salimos si no se puede leer el archivo
        }

        if (filas.isEmpty()) {
            System.err.println("No se cargaron datos desde el CSV.");
            return;
        }

        // 2) Pasar de listas a arrays X (temperatura) e y (ventas)
        int n = filas.size();
        double[][] X = new double[n][1]; // una sola característica: temperatura
        double[] y = new double[n];      // ventas

        for (int i = 0; i < n; i++) {
            X[i][0] = filas.get(i)[0]; // temperatura
            y[i]    = filas.get(i)[1]; // ventas
        }

        // 3) Crear y entrenar el modelo de regresión lineal
        LinearRegression model = new LinearRegression(true); // con bias/intercepto
        model.fit(X, y, false); // false = sin imprimir pasos intermedios (según tu implementación)

        // 4) Predicción sobre los mismos datos (para calcular R²)
        double[] yPred = model.predict(X);
        double r2 = model.score(y, yPred);

        System.out.println("Pesos: " + Arrays.toString(model.getWeights()));
        System.out.println("Bias: " + model.getBias());
        System.out.println("R²: " + String.format(Locale.US, "%.4f", r2));

        // 5) Predicciones para nuevas temperaturas
        double[][] Xtest = { {22}, {28}, {33} }; // Nuevos datos (°C)
        double[] yhat = model.predict(Xtest);

        System.out.println("Predicciones para nuevas temperaturas:");
        System.out.println("Temperaturas: " + Arrays.toString(new double[]{22, 28, 33}));
        System.out.println("Ventas estimadas: " + Arrays.toString(yhat));
    }
}