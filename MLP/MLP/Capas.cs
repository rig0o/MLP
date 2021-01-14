using System;
using System.Collections.Generic;
using System.Text;

namespace MLP
{
    class Capas
    {
        public List<Neurona> neuronas;  ///lista de neuronas
        public double[] outputs; ///salidas de cada neurona, se pasarana la sigiente capa

        public Capas(int inputCount, int neuronasCount, Random r)
        {
            neuronas = new List<Neurona>();
            for(int i = 0; i<neuronasCount; i++)
            {
                neuronas.Add(new Neurona(inputCount, r));   ///a la lista se le agrega un numero x de neurona         
            }
        }
        public double[] activacion(double[] inputs)
        {
            outputs = new double[neuronas.Count];
            for(int i =0; i< neuronas.Count; i++)
            {
                outputs[i] = neuronas[i].activacion(inputs);
            }
           return outputs;
        }

    }
}
