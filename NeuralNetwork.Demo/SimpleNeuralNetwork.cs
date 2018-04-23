using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Framework;

namespace NeuralNetwork.Demo
{
	public class SimpleNeuralNetwork
	{
		private NeuralLayerFactory _layerFactory;

		internal List<NeuralLayer> _layers;
		internal double _learningRate;
		internal double[][] _expectedResult;

		public SimpleNeuralNetwork(int inputNeurons)
		{
			_layers = new List<NeuralLayer>();
			_layerFactory = new NeuralLayerFactory();

			CreateInputLayer(inputNeurons);
			_learningRate = 2.95;
		}

		public void AddLayer(NeuralLayer newLayer)
		{
			if (_layers.Any())
			{
				var lastLayer = _layers.Last();
				newLayer.ConnectLayers(lastLayer);
			}
			_layers.Add(newLayer);
		}

		public void PushInputValues(double[] inputs)
		{
			_layers.First().Neurons.ForEach(x => x.PushValueOnInput(inputs[_layers.First().Neurons.IndexOf(x)]));
		}

		public void PushExpectedValues(double[][] expectedOutputs)
		{
			_expectedResult = expectedOutputs;
		}

		public List<double> GetOutput()
		{
			var output = new List<double>();
			_layers.Last().Neurons.ForEach(neuron =>
			{
				output.Add(neuron.CalculateOutput());
			});
			return output;
		}

		public void Train(double[][] inputs, int epochs)
		{
			double totalError = 0;

			for (int i = 0; i < epochs; i++)
			{
				for (int j = 0; j < inputs.GetLength(0); j++)
				{
					PushInputValues(inputs[j]);

					var outputs = new List<double>();
					_layers.Last().Neurons.ForEach(x =>
					{
						outputs.Add(x.CalculateOutput());
					});

					totalError = CalculateTotalError(outputs, j);
					HandleOutputLayer(j);
					HandleHiddenLayers();
				}
			}
		}

		private void CreateInputLayer(int inputNeurons)
		{
			var inputLayer = _layerFactory.CreateNeuralLayer(inputNeurons, new RectifiedActivationFunction(), new WeightedSumFunction());
			inputLayer.Neurons.ForEach(x => x.AddInputSynapse(0));
			this.AddLayer(inputLayer);
		}

		private double CalculateTotalError(List<double> outputs, int row)
		{
			double totalError = 0;
			outputs.ForEach(output =>
			{
				var error = Math.Pow(output - _expectedResult[row][outputs.IndexOf(output)], 2);
				totalError += error;
			});
			return totalError;
		}

		private void HandleOutputLayer(int row)
		{
			_layers.Last().Neurons.ForEach(neuron =>
			{
				neuron.Inputs.ForEach(connection =>
				{
					var output = neuron.CalculateOutput();
					var netInput = connection.GetOutput();
					var expectedOutput = _expectedResult[row][_layers.Last().Neurons.IndexOf(neuron)];
					var nodeDelta = (expectedOutput - output) * output * (1 - output);
					var delta = -1 * netInput * nodeDelta;
					connection.UpdateWeight(_learningRate, delta);
					neuron.PreviousPartialDerivate = nodeDelta;
				});
			});
		}

		private void HandleHiddenLayers()
		{
			for (int i = _layers.Count - 2; i > 0; i--)
			{
				_layers[i].Neurons.ForEach(neuron =>
				{
					neuron.Inputs.ForEach(connection =>
					{
						var output = neuron.CalculateOutput();
						var netInput = connection.GetOutput();
						double sumPartial = 0;

						_layers[i + 1].Neurons.ForEach(outputNeuron =>
						  {
							  outputNeuron.Inputs.Where(j => j.IsFromNeuron(neuron.Id)).ToList().ForEach(outConnection =>
							  {
								  sumPartial += outConnection.PreviousWeight * outputNeuron.PreviousPartialDerivate;
							  });
						  });

						var delta = -1 * netInput * sumPartial * output * (1 - output);
						connection.UpdateWeight(_learningRate, delta);
					});
				});
			}
		}
	}
}
