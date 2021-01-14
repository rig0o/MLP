using System;
using System.Collections.Generic;

namespace MLP
{
    class Program
    {
        static void Main(string[] args)
        {

            List<double[]> entradas = new List<double[]>();
            List<double[]> salidas = new List<double[]>();

            for(int i = 0; i < 4; i++)
            {
                entradas.Add(new double[2]);
                salidas.Add(new double[1]);
            }

            entradas[0][0] = 0; entradas[0][1] = 0; salidas[0][0]= 1;
            entradas[1][0] = 0; entradas[1][1] = 1; salidas[1][0] = 0;
            entradas[2][0] = 1; entradas[2][1] = 0; salidas[2][0] = 0;
            entradas[3][0] = 1; entradas[3][1] = 1; salidas[3][0]= 1;



            Mlp p = new Mlp(new int[] { entradas[0].Length, 3, salidas[0].Length });
            p.Aprender(entradas, salidas, 1.6, 0.01);
            while (true)
            {
                Console.WriteLine("inserte valores");
                double d = double.Parse(Console.ReadLine());
                double d2 = double.Parse(Console.ReadLine());

                Console.WriteLine("Respuesta: {0}", p.Activacion(new double[] { d,d2})[0]);
            }


        }
    }
}
