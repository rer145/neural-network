﻿using System;

namespace NeuralNetwork.Framework
{
	public class RectifiedActivationFunction : IActivationFunction
	{
		public double CalculateOutput(double input)
		{
			return Math.Max(0, input);
		}
	}
}
