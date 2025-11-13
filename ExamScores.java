import java.io.*;
import java.util.*;

// Operaciones con matrices
class Mat {
    static double[][] T(double[][] A) {
        int n=A.length, m=A[0].length;
        double[][] R=new double[m][n];
        for(int i=0;i<n;i++) for(int j=0;j<m;j++) R[j][i]=A[i][j];
        return R;
    }
    static double[][] mm(double[][] A,double[][] B){
        int n=A.length,m=A[0].length,p=B[0].length;
        double[][] R=new double[n][p];
        for(int i=0;i<n;i++) for(int k=0;k<m;k++) for(int j=0;j<p;j++) R[i][j]+=A[i][k]*B[k][j];
        return R;
    }
    static double[] mv(double[][] A,double[] v){
        int n=A.length,m=A[0].length; double[] r=new double[n];
        for(int i=0;i<n;i++) for(int j=0;j<m;j++) r[i]+=A[i][j]*v[j];
        return r;
    }
    static double[][] addOnes(double[][] X){
        int n=X.length,m=X[0].length; double[][] R=new double[n][m+1];
        for(int i=0;i<n;i++){ R[i][0]=1.0; System.arraycopy(X[i],0,R[i],1,m); }
        return R;
    }
    static double[][] eye(int n){
        double[][] I=new double[n][n]; for(int i=0;i<n;i++) I[i][i]=1.0; return I;
    }
    static double[][] inv(double[][] A){
        int n=A.length; double[][] M=new double[n][2*n];
        for(int i=0;i<n;i++){ System.arraycopy(A[i],0,M[i],0,n); M[i][n+i]=1.0; }
        for(int c=0;c<n;c++){
            int piv=c; for(int r=c+1;r<n;r++) if(Math.abs(M[r][c])>Math.abs(M[piv][c])) piv=r;
            double[] tmp=M[c]; M[c]=M[piv]; M[piv]=tmp;
            double div=M[c][c]; if(Math.abs(div)<1e-12) throw new RuntimeException("Matriz no invertible");
            for(int j=0;j<2*n;j++) M[c][j]/=div;
            for(int r=0;r<n;r++) if(r!=c){
                double f=M[r][c]; for(int j=0;j<2*n;j++) M[r][j]-=f*M[c][j];
            }
        }
        double[][] R=new double[n][n]; for(int i=0;i<n;i++) System.arraycopy(M[i],n,R[i],0,n); return R;
    }
}

// Escalador de variables
class Scaler {
    private double[] mean, std; private boolean fitted=false;
    public void fit(double[][] X){
        int n=X.length,m=X[0].length; mean=new double[m]; std=new double[m];
        for(int j=0;j<m;j++){ double s=0; for(int i=0;i<n;i++) s+=X[i][j]; mean[j]=s/n; }
        for(int j=0;j<m;j++){
            double v=0; for(int i=0;i<n;i++){ double d=X[i][j]-mean[j]; v+=d*d; }
            std[j]=Math.sqrt(v/Math.max(1,n-1)); if(std[j]==0) std[j]=1.0;
        }
        fitted=true;
    }
    public double[][] transform(double[][] X){
        if(!fitted) throw new IllegalStateException("Scaler no entrenado");
        int n=X.length,m=X[0].length; double[][] R=new double[n][m];
        for(int i=0;i<n;i++) for(int j=0;j<m;j++) R[i][j]=(X[i][j]-mean[j])/std[j];
        return R;
    }
}

// Regresión Lineal Múltiple
class LinearRegression {
    private double[] weights; private double bias;
    private final boolean fitIntercept, scale; private final double lambda;
    private Scaler scaler;

    public LinearRegression(boolean fitIntercept, boolean scale, double lambda){
        this.fitIntercept=fitIntercept; this.scale=scale; this.lambda=Math.max(0.0,lambda);
    }

    public void fit(double[][] X, double[] y){
        double[][] Xp = X;
        if(scale){ scaler=new Scaler(); scaler.fit(X); Xp=scaler.transform(X); }
        double[][] Xb = fitIntercept? Mat.addOnes(Xp) : Xp;
        double[][] Xt = Mat.T(Xb);
        double[][] XtX = Mat.mm(Xt, Xb);
        if(lambda>0){
            int p=XtX.length; double[][] I=Mat.eye(p);
            if(fitIntercept) I[0][0]=0.0; // no regularizar bias
            for(int i=0;i<p;i++) for(int j=0;j<p;j++) XtX[i][j]+=lambda*I[i][j];
        }
        double[] Xty = Mat.mv(Xt, y);
        double[][] XtXInv = Mat.inv(XtX);
        double[] wFull = Mat.mv(XtXInv, Xty);
        if(fitIntercept){ bias=wFull[0]; weights=Arrays.copyOfRange(wFull,1,wFull.length); }
        else { bias=0.0; weights=wFull; }
    }

    public double[] predict(double[][] X){
        double[][] Xp = (scale && scaler!=null)? scaler.transform(X) : X;
        int n=Xp.length,d=Xp[0].length; double[] yhat=new double[n];
        for(int i=0;i<n;i++){ double s=bias; for(int j=0;j<d;j++) s+=weights[j]*Xp[i][j]; yhat[i]=s; }
        return yhat;
    }

    public static double mse(double[] y,double[] yhat){ double e=0; for(int i=0;i<y.length;i++){ double d=y[i]-yhat[i]; e+=d*d; } return e/y.length; }
    public static double rmse(double[] y,double[] yhat){ return Math.sqrt(mse(y,yhat)); }
    public static double r2(double[] y,double[] yhat){
        double m=0; for(double v:y) m+=v; m/=y.length;
        double ssTot=0, ssRes=0; for(int i=0;i<y.length;i++){ ssTot+=Math.pow(y[i]-m,2); ssRes+=Math.pow(y[i]-yhat[i],2); }
        return 1 - ssRes/Math.max(1e-12, ssTot);
    }

    public double[] getWeights(){ return weights; }
    public double   getBias(){ return bias; }
}

// Lector de datos CSV
class CSV {
    static class Data { double[][] X; double[] y; }
    static CSV.Data read(String path) throws Exception {
        List<double[]> rows=new ArrayList<>();
        try(BufferedReader br=new BufferedReader(new FileReader(path))){
            String line; boolean first=true;
            while((line=br.readLine())!=null){
                line=line.trim(); if(line.isEmpty()) continue;
                if(first){ first=false; continue; } // salta encabezado
                String[] parts=line.split(",");
                double[] nums=new double[parts.length];
                for(int i=0;i<parts.length;i++) nums[i]=Double.parseDouble(parts[i].trim());
                rows.add(nums);
            }
        }
        int n=rows.size(), m=rows.get(0).length-1;
        double[][] X=new double[n][m]; double[] y=new double[n];
        for(int i=0;i<n;i++){ double[] r=rows.get(i); for(int j=0;j<m;j++) X[i][j]=r[j]; y[i]=r[m]; }
        CSV.Data d=new CSV.Data(); d.X=X; d.y=y; return d;
    }
}

// Asignacion de datos de entrenamiento y prueba
class DataSplit {
    double[][] Xtr, Xte; double[] ytr, yte;
    static DataSplit split(double[][] X,double[] y,double testRatio,long seed){
        int n=X.length,m=X[0].length; Integer[] idx=new Integer[n];
        for(int i=0;i<n;i++) idx[i]=i; Collections.shuffle(Arrays.asList(idx), new Random(seed));
        int nTest=Math.max(1,(int)Math.round(n*testRatio)), nTrain=n-nTest;
        double[][] Xtr=new double[nTrain][m], Xte=new double[nTest][m];
        double[] ytr=new double[nTrain], yte=new double[nTest];
        for(int i=0;i<nTrain;i++){ int r=idx[i]; Xtr[i]=Arrays.copyOf(X[r],m); ytr[i]=y[r]; }
        for(int i=0;i<nTest;i++){ int r=idx[nTrain+i]; Xte[i]=Arrays.copyOf(X[r],m); yte[i]=y[r]; }
        DataSplit s=new DataSplit(); s.Xtr=Xtr; s.Xte=Xte; s.ytr=ytr; s.yte=yte; return s;
    }
}

// Main
public class ExamScores {
    public static void main(String[] args) throws Exception {
        // Parámetros modelo
        boolean fitIntercept=true;
        boolean scale=true;      
        double lambda=0.0;    
        double testRatio=0.20;
        long seed=123;

        double[][] X;
        double[] y;

        if(args.length==1){
            CSV.Data d=CSV.read(args[0]); X=d.X; y=d.y;
        }else{
            // Si no hay archivo CSV, usa este ejemplo
            X=new double[][]{
                {5, 7, 0.9}, {3, 6, 0.8}, {8, 6, 0.95}, {2, 5, 0.7}, {7, 6, 0.92},
                {4, 8, 0.85}, {6, 7, 0.88}, {9, 6, 0.97}, {1, 4, 0.6}, {5, 5, 0.8}
            };
            y=new double[]{65,58,88,45,82,60,74,92,38,68};
        }

        // Split
        DataSplit ds=DataSplit.split(X,y,testRatio,seed);

        // Entrena
        LinearRegression lr=new LinearRegression(fitIntercept, scale, lambda);
        lr.fit(ds.Xtr, ds.ytr);

        // Evalúa
        double[] yhatTr=lr.predict(ds.Xtr), yhatTe=lr.predict(ds.Xte);
        double r2Tr=LinearRegression.r2(ds.ytr,yhatTr), r2Te=LinearRegression.r2(ds.yte,yhatTe);
        double mseTe=LinearRegression.mse(ds.yte,yhatTe), rmseTe=LinearRegression.rmse(ds.yte,yhatTe);

        // Resultados
        System.out.println("Pesos: " + Arrays.toString(lr.getWeights()));
        System.out.println("Bias : " + lr.getBias());
        System.out.printf(Locale.US,"R² (train): %.4f%n", r2Tr);
        System.out.printf(Locale.US,"R² (test) : %.4f%n", r2Te);
        System.out.printf(Locale.US,"MSE(test) : %.6f%n", mseTe);
        System.out.printf(Locale.US,"RMSE(test): %.6f%n", rmseTe);

        // Predicción de ejemplo
        double[][] Xnew={{6,7,0.9},{2,5,0.7}};
        System.out.println("Predicciones: " + Arrays.toString(lr.predict(Xnew)));
    }
}

