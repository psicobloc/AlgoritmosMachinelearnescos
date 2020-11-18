// package linearalgebra;

import org.apache.commons.math3.linear.*;
import jade.core.Agent;
import jade.core.behaviours.*;
// import org.apache.commons.math3.linear.*;



public class SRLAgent extends Agent {
    private float[][] data;
    private float[] beta;
    private int X;

    protected void setup()
    {
        //print mensaje bienvenida
        System.out.println("\n\n\n\n\n\nHola agente SRL" + getAID().getName() + " esta listo..." );
        //The name in JADE has the form <nickname>@<platform-name>
        Object[] args = getArguments();
        String cadena = "";

        if (args != null && args.length >0) {
            cadena = (String)args[0];
            int x = Integer.parseInt(cadena);
            X = x;
            addBehaviour(new SimpleLinearRegression());
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
}
