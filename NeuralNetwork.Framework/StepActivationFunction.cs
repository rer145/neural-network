using System;

namespace NeuralNetwork.Framework
{
	public class StepActivationFunction : IActivationFunction
	{
		private double _threshold;

		public StepActivationFunction(double threshold)
		{
			_threshold = threshold;
		}

		public double CalculateOutput(double input)
		{
			return Convert.ToDouble(input > _threshold);
		}
	}
}
