using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Framework
{
	public class Neuron : INeuron
	{
		public IActivationFunction _activationFunction;
		public IInputFunction _inputFunction;

		public Guid Id { get; private set; }
		public List<ISynapse> Inputs { get; set; }
		public List<ISynapse> Outputs { get; set; }
		public double PreviousPartialDerivate { get; set; }

		public Neuron(IActivationFunction activationFunction, IInputFunction inputFunction)
		{
			Id = Guid.NewGuid();
			Inputs = new List<ISynapse>();
			Outputs = new List<ISynapse>();
			_activationFunction = activationFunction;
			_inputFunction = inputFunction;
		}

		/// <summary>
		/// Connects two neurons to each other. This neuron is the OUTPUT neuron of the connection.
		/// </summary>
		/// <param name="inputNeuron">Neuron that will be the input neuron of the newly created connection.</param>
		public void AddInputNeuron(INeuron inputNeuron)
		{
			var synapse = new Synapse(inputNeuron, this);
			Inputs.Add(synapse);
			inputNeuron.Outputs.Add(synapse);
		}
		
		/// <summary>
		/// Input layer neurons only receive input values. This function adds this kind of connection to the neuron.
		/// </summary>
		/// <param name="inputValue">Initial value that will be pushed as an input to a connection.</param>
		public void AddInputSynapse(double inputValue)
		{
			var inputSynapse = new InputSynapse(this, inputValue);
			Inputs.Add(inputSynapse);
		}

		/// <summary>
		/// Connects two neurons. This neuron is the INPUT neuron of the connection.
		/// </summary>
		/// <param name="inputNeuron"></param>
		public void AddOutputNeuron(INeuron outputNeuron)
		{
			var synapse = new Synapse(this, outputNeuron);
			Outputs.Add(synapse);
			outputNeuron.Inputs.Add(synapse);
		}

		/// <summary>
		/// Calculate the output value of the neuron.
		/// </summary>
		/// <returns>The output of the neuron.</returns>
		public double CalculateOutput()
		{
			return _activationFunction.CalculateOutput(_inputFunction.CalculateInput(this.Inputs));
		}

		/// <summary>
		/// Sets a new value on the input connection.
		/// </summary>
		/// <param name="inputValue">The value that will be pushed as an input to a connection.</param>
		public void PushValueOnInput(double inputValue)
		{
			((InputSynapse)Inputs.First()).Output = inputValue;
		}
	}
}
