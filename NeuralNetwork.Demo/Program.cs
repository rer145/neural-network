using NeuralNetwork.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Demo
{
	class Program
	{
		static void Main(string[] args)
		{
			Console.WriteLine("START");
			var network = new SimpleNeuralNetwork(3);

			var layerFactory = new NeuralLayerFactory();
			network.AddLayer(layerFactory.CreateNeuralLayer(3, new RectifiedActivationFunction(), new WeightedSumFunction()));
			network.AddLayer(layerFactory.CreateNeuralLayer(1, new SigmoidActivationFunction(0.7), new WeightedSumFunction()));

			network.PushExpectedValues(
				new double[][]
				{
					new double[] { 0 },
					new double[] { 1 },
					new double[] { 1 },
					new double[] { 0 },
					new double[] { 1 },
					new double[] { 0 },
					new double[] { 0 },
				});

			Console.WriteLine("TRAINING NETWORK");
			network.Train(
				new double[][]
				{
					new double[] { 150, 2, 0 },
					new double[] { 1002, 56, 1 },
					new double[] { 1002, 59, 1 },
					new double[] { 200, 3, 0 },
					new double[] { 300, 2, 1 },
					new double[] { 120, 1, 0 },
					new double[] { 80, 1, 0 },
				}, 10000);

			network.PushInputValues(new double[] { 1054, 54, 1 });
			var outputs = network.GetOutput();

			Console.WriteLine("OUTPUT");
			outputs.ForEach(x => Console.WriteLine(x));

			Console.WriteLine("DONE");
			Console.ReadLine();
		}
	}
}
