import jade.core.Agent;
import jade.core.behaviours.Behaviour;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import javax.swing.JOptionPane;

public class KNNAgent extends Agent {
    private double[] distanciasMinimas;
    private int[] vecinosCercanos;
    private int K;
    private int opcionDistancia =1;
    private double[] instanciaPrediccion = {0.0,0.0};
    private String resultadoPrediccion;
    private Dataset datasetKNN;
    private double[][] datosDouble = {
            {158.0,58.0},
            {158.0,59.0},
            {158,63},
            {160,59},
            {160,60},
            {163,60},
            {163,61},
            {160,64},
            {163,64},
            {165,61},
            {165,62},
            {165,65},
            {168,62},
            {168,63},
            {168,66},
            {170,63},
            {170,64},
            {170,68}
    };

    private String[] labels = {"M", "M", "M", "M", "M", "M", "M", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L"};

    protected void setup()
    {
        datasetKNN = new Dataset(datosDouble,labels);
        double valorInput1 = 0.0;
        double valorInput2 = 0.0;
        String cadena = "";
        cadena = JOptionPane.showInputDialog("Ingrese el valor de x1(altura): ");
        valorInput1 = Double.parseDouble(cadena);
        cadena = JOptionPane.showInputDialog("Ingrese el valor de x2(peso): ");
        valorInput2 = Double.parseDouble(cadena);
        cadena = JOptionPane.showInputDialog("Ingrese el valor de K: ");
        K = Integer.parseInt(cadena);
        cadena = JOptionPane.showInputDialog("seleccione la opcion de metrica de distancia\n1 -> Eucllidiana, 2-> Manhattan");
        opcionDistancia = Integer.parseInt(cadena);
        instanciaPrediccion[0] = valorInput1;
        instanciaPrediccion[1] = valorInput2;
        distanciasMinimas = new double[K];
        for (int iterd=0;iterd < K; iterd++)
        {
            distanciasMinimas[iterd] = 999999.9;
        }
        vecinosCercanos = new int[K];
        for (int iterv=0;iterv < K; iterv++)
        {
            vecinosCercanos[iterv] = 0;
        }
        addBehaviour(new KNN());
    }
    protected void takeDown()
    {
        System.out.println("Agente" + getAID().getName() + " Terminando..." );
    }

    private class KNN extends Behaviour  {
        private boolean flagTerminado = false;

        public void action()
        {
            RunKNN(opcionDistancia); //podemos agregar opcion 3, ambas metricas
            flagTerminado = true;
        }

        public boolean done() {
            System.out.println("\n\nTerminado KNN\n\n");
            return flagTerminado;
        }

        public void RunKNN(int opc)
        {
            double nuevaDistancia =0.0;
            int dimRows = datasetKNN.getDatos().getRowDimension();
            int contadorClases =0;

            for (int i =0; i < dimRows; i++)
            {
                if (opc == 1)
                {
                    nuevaDistancia = distanciaEuclidiana(datasetKNN.getDatos().getRow(i), instanciaPrediccion);
                } else if (opc == 2)
                {
                    nuevaDistancia = distanciaManhattan(datasetKNN.getDatos().getRow(i), instanciaPrediccion);
                }

                //System.out.println("Nueva dist:" + nuevaDistancia);
                for (int j=0;j <K; j++)
                {
                    if (nuevaDistancia < distanciasMinimas[j])
                    {
                        insertInPlace(1,j,nuevaDistancia);
                        insertInPlace(2,j,i); //se puede mandar directamente el int como un doble? niceee
                        break;
                    }
                }
            }

            System.out.println("Vecinos Más cercanos:\n");
            for (int iterVec=0; iterVec < vecinosCercanos.length; iterVec++)
            {
                System.out.println(vecinosCercanos[iterVec] + "\tdistancia: " + distanciasMinimas[iterVec] + "\t Clase: " + labels[vecinosCercanos[iterVec]]);
                if (labels[vecinosCercanos[iterVec]] == "M")
                {
                    contadorClases = contadorClases+1;
                } else if (labels[vecinosCercanos[iterVec]] == "L")
                {
                    contadorClases = contadorClases -1;
                }
            }
            if (contadorClases >=1)
            {
                resultadoPrediccion = "M";
            } else
            {
                resultadoPrediccion = "L"; //si hay empate elegimos L
            }
            System.out.println("\nPrediccion:\t" + resultadoPrediccion);
        }

        public void insertInPlace(int opc, int place, double valorIN) //opc1 distancias minimas, opc2 vecinos cercanos, place(0->...
        {
            //distancias minimas
            if (opc ==1)
            {
                for (int i= (K-1); i>place; i--)
                {
                    distanciasMinimas[i] = distanciasMinimas[i-1];
                }
                distanciasMinimas[place] = valorIN;

            } else if (opc ==2)//vecinos cercanos
            {
                for (int i= (K-1); i>place; i--)
                {
                    vecinosCercanos[i] = vecinosCercanos[i-1];
                }
                Double doubleObj = new Double(valorIN);
                int valorInt = doubleObj.intValue();
                vecinosCercanos[place] = valorInt;
            }
        }

        public double distanciaEuclidiana(double[] vector1, double[] vector2)
        {
            double resultado =0;
            for (int i=0; i< vector1.length; i++)
            {
                resultado = resultado + Math.pow((vector1[i] - vector2[i]),2);
            }
            return resultado;
        }

        public double distanciaManhattan(double[] vector1, double[] vector2)
        {
            double resultado =0;
            for (int i=0; i< vector1.length; i++)
            {
                resultado = resultado + Math.abs((vector1[i] - vector2[i]));
            }
            return resultado;
        }
    }

    private class Dataset { //se guardan los datos en una RealMatrix
        RealMatrix datos;
        String[] labels;

        public Dataset(double[][] dataIN, String[] labelsIN)
        {
            datos = MatrixUtils.createRealMatrix(dataIN);
            labels = labelsIN;
        }

//            public Dataset()
//            {
//                double[][] datosDouble = {
//                        {158.0,58.0},
//                        {158.0,59.0},
//                        {158,63},
//                        {160,59},
//                        {160,60},
//                        {163,60},
//                        {163,61},
//                        {160,64},
//                        {163,64},
//                        {165,61},
//                        {165,62},
//                        {165,65},
//                        {168,62},
//                        {168,63},
//                        {168,66},
//                        {170,63},
//                        {170,64},
//                        {170,68}
//                };
//                datos = MatrixUtils.createRealMatrix(datosDouble);
//                labels = new String[]{"M", "M", "M", "M", "M", "M", "M", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L"};
//            }

        public RealMatrix getDatos ()
        {
            return datos;
        }

        public  String[] getLabels()
        {
            return labels;
        }

        public void makeMatrix(double[][] dataIN, String[] labelsIN)
        {
            datos = MatrixUtils.createRealMatrix(dataIN);
            labels = labelsIN;
        }

        public void setDatos(RealMatrix dataIN)
        {
            datos = dataIN;
        }
    }
}
