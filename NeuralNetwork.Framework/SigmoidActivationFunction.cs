using System;

namespace NeuralNetwork.Framework
{
	public class SigmoidActivationFunction : IActivationFunction
	{
		private double _coefficient;

		public SigmoidActivationFunction(double coefficient)
		{
			_coefficient = coefficient;
		}

		public double CalculateOutput(double input)
		{
			return (1 / (1 + Math.Exp(-input * _coefficient)));
		}
	}
}
