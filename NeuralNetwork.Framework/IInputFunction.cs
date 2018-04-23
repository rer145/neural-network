using System.Collections.Generic;

namespace NeuralNetwork.Framework
{
    public interface IInputFunction
    {
		double CalculateInput(List<ISynapse> inputs);
    }
}
