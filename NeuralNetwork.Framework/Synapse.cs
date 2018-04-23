using System;

namespace NeuralNetwork.Framework
{
	public class Synapse : ISynapse
	{
		internal INeuron _fromNeuron;
		internal INeuron _toNeuron;

		public double Weight { get; set; }
		public double PreviousWeight { get; set; }

		public Synapse(INeuron fromNeuron, INeuron toNeuron, double weight)
		{
			_fromNeuron = fromNeuron;
			_toNeuron = toNeuron;
			Weight = weight;
			PreviousWeight = 0;
		}

		public Synapse(INeuron fromNeuron, INeuron toNeuron)
		{
			_fromNeuron = fromNeuron;
			_toNeuron = toNeuron;
			Weight = new Random().NextDouble();
			PreviousWeight = 0;
		}

		public double GetOutput()
		{
			return _fromNeuron.CalculateOutput();
		}

		/// <summary>
		/// Checks if the Neuron has a certain Neuron as input.
		/// </summary>
		/// <param name="fromNeuronId"></param>
		/// <returns>
		/// True if the Neuron is the input of the connection.
		/// False if the Neuron is not hte input of the connection.
		/// </returns>
		public bool IsFromNeuron(Guid fromNeuronId)
		{
			return _fromNeuron.Id.Equals(fromNeuronId);
		}

		/// <summary>
		/// Updates the weight.
		/// </summary>
		/// <param name="learningRate">The chosen learning rate.</param>
		/// <param name="delta">The calculated differene for the weight of the connection that needs to be modified.</param>
		public void UpdateWeight(double learningRate, double delta)
		{
			PreviousWeight = Weight;
			Weight += learningRate * delta;
		}
	}
}
