// Regresion lineal simple, optimización por minimos cuadrados.
//Hugo Valencia Vargas
//import java.util.Arrays; //para imprimir arreglos con toString()

class RegresionLineal{
    private static float[][] data; // (x,y) observed
    private static float[] beta;  // array of beta parameters

    public RegresionLineal() //constructor
    {
        //inicializar el dataset (Hardcodeado)
        data = new float[][]{
        {23, 651},
        {26, 762},
        {30, 856},
        {34, 1063},
        {43, 1190},
        {48, 1298},
        {52, 1421},
        {57, 1440},
        {58, 1518},
    };
        beta = new float[2]; //alocate memory for beta parameters
    }

    public static void main(String[] args) { //recibe X, imprime ŷ
        int x = castToInt(args[0]);
        calcularParametros();
        float yPredicted = 0;
        yPredicted= predecir(x);
        System.out.println("Y predicted:\t" + yPredicted);
        System.out.println("Ecuación:\t" + beta[0] + " + ( " +beta[1]+ " * " + x+" ) = "+yPredicted);
        //costo:
        //double loss = lossFunc();
        //System.out.println("Mean squared error:\t" + loss);
    }

    public static  float predecir (int x) // ŷ = beta[0] + beta[1]*x
    {
        float y=0;
        y = beta[0] + (x* beta[1]);
        //NOTE revisar error cuando con/sin redondeos
        // int a = (int) beta[0];
        // int b = (int) beta[1];
        // float yRounded = a+(x*b);
        return y;
    }

    public static void calcularParametros()
    {
        beta = new float[2];
        beta[0] = 0;
        beta[1] = 0;
        int n =0;
        float sumXSqrd = 0;
        float sumX =0;
        float sumY=0;
        float sumXY=0;
        float frstTerm =0;
        float scndTerm =0;
        float[][] data2 = {
        {5, 5},
        {10, 10},
        {15, 15},
        {20, 20},
        {25, 25}
        };

        data = data2;
        n = data.length;

        for (int i=0; i<n ;i++ ) {
            sumY = sumY + data[i][1]; //sum y component
            sumX = sumX + data[i][0]; //sum x component
            sumXY = sumXY + (data[i][0] * data[i][1]); //sum the product of xy
            sumXSqrd = sumXSqrd + (data[i][0]*data[i][0]); //sum xi^2
        }
        //calcular beta1
        frstTerm =((n*sumXY)-(sumX*sumY));
        scndTerm = ((n*sumXSqrd)-(sumX*sumX));
        beta[1]= frstTerm/scndTerm;
        //calcular beta0
        beta[0]= (sumY-(beta[1]*sumX))/n;

        //DEBUG
        // System.out.println("sumY : " + sumY);
        // System.out.println("sumX : " + sumX);
        // System.out.println("sumXY : " + sumXY);
        // System.out.println("sumX^2 : " + sumXSqrd);
        // System.out.println("n : " + n);
        // System.out.println("frstTerm: "+frstTerm);
        // System.out.println("scndTerm: "+scndTerm);
        System.out.println("beta0 : " + beta[0]);
        System.out.println("beta1 : " + beta[1]);
    }

    //funcion de costo: 1/2 mean squared error! no sum squared error
    private static double lossFunc()
    {
        int n = 0;
        double loss =0;
        n = data.length;
        for (int i=0;i<n;i++) {
            loss = loss + Math.pow((((beta[0] +(beta[1]*data[i][0]))-data[i][1])),2);
        }
        return loss/(2*n);
    }

    //utility:
    private static int castToInt(String cadena)
    {
        int x = 0;
        try {
            x = Integer.parseInt(cadena);
        } catch(Exception e) {
            System.out.println("El argumento no es un entero");
        }
        return x;
    }
}
