namespace NeuralNetwork.Framework
{
	public class NeuralLayerFactory
	{
		public NeuralLayer CreateNeuralLayer(int neurons, IActivationFunction activationFunction, IInputFunction inputFunction)
		{
			var layer = new NeuralLayer();

			for (int i = 0; i < neurons; i++)
			{
				var neuron = new Neuron(activationFunction, inputFunction);
				layer.Neurons.Add(neuron);
			}

			return layer;
		}
	}
}
