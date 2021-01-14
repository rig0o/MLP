using System;
using System.Collections.Generic;
using System.Text;

namespace MLP
{
    class Neurona
    {
        public double[] pesos;
        public double bias;

        public double ultimaActivacion;

        public Neurona(int inputCount, Random r)// constructor -- inputCount = numero de entradas a la neurona
        {
            bias = r.NextDouble();
            pesos = new double[inputCount];
            for(int i = 0; i < inputCount; i++)
            {
                pesos[i] = r.NextDouble();              ///por cada entrada i, se le asigna un peso aleatorio
            }
        
        }

        public double activacion(double[] inputs)               ///sumatoria regresión lineal
        {
            double ultimaActivacion = bias;
            for(int i = 0; i < inputs.Length; i++)
            {
                ultimaActivacion += inputs[i] * pesos[i];       ///entrada por su peso
            }
            return Sigmoide(ultimaActivacion);
        }


        //Funcion de activacion sigmoidal
        public static double Sigmoide(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }
        //Derivada de la f.sigmoidal
        public static double SigmaideDerivada(double input)
        {
            double y = Sigmoide(input);
            return y * (1 - y);
        }

    }
}
