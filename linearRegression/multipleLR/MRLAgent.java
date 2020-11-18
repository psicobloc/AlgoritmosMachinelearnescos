import jade.core.Agent;
import jade.core.behaviours.Behaviour;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
//import org.apache.commons.math3.linear.*;

public class MRLAgent extends Agent {
    private float[][] data;
    private float[] beta;
    private int X;
    private double X0;
    private double X1;
    private double X2;

    double[][] matrixData = { {11,68.028164,0},
            {12,69.086446,0},
            {13,84.806730,1},
            {14,19.011313,1},
            {15,63.046323,1},
            {16,82.626964,1},
            {17,59.263664,1},
            {18,88.756598,0},
            {19,77.884304,1},
            {20,9.346073,0}};
    double[][] matrixDataY = {
            {21.46126},
            {23.28792},
            {18.71906},
            {18.50209},
            {22.37717},
            {24.19955},
            {20.64198},
            {29.50144},
            {24.49684},
            {27.15191}};
    RealMatrix dataset;
    RealMatrix dataTranspose;
    RealMatrix Xmultiply;
    RealMatrix XmultiplyInverse;
    RealMatrix Ymatrix;
    RealMatrix Betas;
    RealMatrix XtransY;

    protected void setup()
    {
        //NOTA normalizar dataset
        //cargar dataset X
        dataset = MatrixUtils.createRealMatrix(matrixData); //matriz X
        Ymatrix = MatrixUtils.createRealMatrix(matrixDataY); // matriz Y
        //print mensaje bienvenida
        System.out.println("\n\n\n\n\n\nHola agente MRL " + getAID().getName() + " esta listo..." );
        Object[] args = getArguments();
        String cadena = "";
        String cadena1 = "";
        String cadena2 = "";
        System.out.println("\n\n argumentos: " + args.length + "\n arg1= " + args[0] + "  arg2= " + args[1]+ "  arg3= " + args[2]);

        if (args != null && args.length ==1) {
            cadena = (String)args[0];
            int x = Integer.parseInt(cadena);
            X = x;
            addBehaviour(new SimpleLinearRegression());
        }
        else if (args.length == 3)
        {
            cadena = (String) args[0];
            cadena1 = (String) args[1];
            cadena2 = (String) args[2];
            double x0 = Double.parseDouble(cadena);
            double x1 = Double.parseDouble(cadena1);
            double x2 = Double.parseDouble(cadena2);
            X0 = x0;
            X1 = x1;
            X2 = x2;
            addBehaviour(new MultipleLinearRegression());
        }
        else {
            System.out.println("no se especificó un valor para x");
            doDelete();
        }
    }
    protected void takeDown()
    {
        System.out.println("Agente" + getAID().getName() + " Terminando..." );
    }

    private class SimpleLinearRegression extends Behaviour  {
        private int n = 0;
        private float sumXSqrd = 0;
        private float sumX =0;
        private float sumY=0;
        private float yPredicted=0;
        private float sumXY=0;
        private float frstTerm =0;
        private float scndTerm =0;
        private boolean flagTerminado = false;

        public void action()
        {
            data = new float[][] {
            {36, 31},
            {28, 29},
            {35, 34},
            {39, 35},
            {30, 29},
            {30, 30},
            {31, 30},
            {38, 38},
            {36, 34},
            {38, 33},
            {29, 29},
            {26, 26},
            };
            beta = new float[2];
            beta[0] =0;
            beta[1] =0;
            n = data.length;
            System.out.println("\n\naction\n\n");
            for (int i=0; i<n ;i++ ) {
                sumY = sumY + data[i][1]; //sum y component
                sumX = sumX + data[i][0]; //sum x component
                sumXY = sumXY + (data[i][0] * data[i][1]); //sum the product of xy
                sumXSqrd = sumXSqrd + (data[i][0]*data[i][0]); //sum xi^2
            }
            // //calcular beta1
            frstTerm =((n*sumXY)-(sumX*sumY));
            scndTerm = ((n*sumXSqrd)-(sumX*sumX));
            beta[1]= frstTerm/scndTerm;
            // //calcular beta0
            beta[0]= (sumY-(beta[1]*sumX))/n;
            //realizar prediccion
            yPredicted = beta[0] + (X* beta[1]);
            //imprimir resultados
            System.out.println("\n\n****\nY predicted:\t" + yPredicted);
            System.out.println("****\n\nEcuación:\t" + beta[0] + " + ( " +beta[1]+ " * " + X+" ) = "+yPredicted);
            flagTerminado = true; //one shot
        }

        public boolean done() {
			System.out.println("\n\nTerminado\n\n");
            return flagTerminado;
		}
    }

    private class MultipleLinearRegression extends Behaviour  {
        private boolean flagTerminado = false;

        public void action()
        {
            //transpuesta  X'
            dataTranspose = dataset.transpose();
            // X' * X
            Xmultiply = dataTranspose.multiply(dataset);
            //inversa (X'*X)^⁻1
            XmultiplyInverse = new LUDecomposition(Xmultiply).getSolver().getInverse();
            // X'*Y
            XtransY = dataTranspose.multiply(Ymatrix);
            //betas
            Betas = XmultiplyInverse.multiply(XtransY);
            System.out.println("\n\n\nBetas = \n" + Betas);

            //Prediccion
            double[] Xvector = {X0,X1,X2};
            RealVector Xin;
            Xin = MatrixUtils.createRealVector(Xvector);
            RealVector resultado;
            RealMatrix betaTransp;
            betaTransp = MatrixUtils.createRealMatrix(matrixData);
            betaTransp = Betas.transpose();
            //resultado
            resultado = MatrixUtils.createRealVector(Xvector);
            resultado = betaTransp.operate(Xin);
            System.out.println("\n\nResultado: " + resultado);

            //DEBUG
            //System.out.println("\ntranspuesta\n\n" + dataTranspose);
            //System.out.println("Multiplicacion de X y X\'\n" + Xmultiply); //producto punto o element-wise??
            //System.out.println("\n\nX\' * Y\n" + XtransY);
            //System.out.println("inversa de X*X\'\n" + XmultiplyInverse);
            flagTerminado = true;
        }

        public boolean done() {
            System.out.println("\n\nTerminado MLR\n\n");
            return flagTerminado;
        }
    }
}
