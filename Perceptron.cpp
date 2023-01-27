#include "Perceptron.h"


Perceptron::Perceptron(unsigned numOutputs, unsigned myIndex)
{
	//Constructor of the Perceptron which sets the number of
	//outputs and the index of the perceptron declared
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}


Perceptron::~Perceptron()
{
	//Destructor
}

double Perceptron::eta = 0.15;
double Perceptron::alpha = 0.5;
void Perceptron::setOutputVal(double val)
{
	//Sets the output of the current perceptron (defines it)
	m_outputVal = val;
}
double Perceptron::getOutputVal(void) const
{
	//Gets the output val of the current perceptron
	//returns the defined value of the perceptron
	return m_outputVal;
}
void Perceptron::feedForward(const Layer& prevLayer)
{
	//This part sums the previous layer's outputs 
	//which are our inputs. Also includes the bias
	//node from the previous layer.
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	m_outputVal = Perceptron::transferFunction(sum);
}
void Perceptron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Perceptron::transferFunctionDerivative(m_outputVal);
}
void Perceptron::calcHiddenGradients(const Layer& nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Perceptron::transferFunctionDerivative(m_outputVal);
}
void Perceptron::updateInputWeights(Layer& prevLayer)
{
	//The weights to be updated are in the connection structure 
	//(which is defined in the header) in the neurons in the layer before this one

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Perceptron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			//Individual input, magnified by the gradient and train rate
			eta
			* neuron.getOutputVal()
			* m_gradient
			//Also add momentum = a fraction of the previous delta weight
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}
double Perceptron::transferFunction(double x)
{
	return tanh(x);
}
double Perceptron::transferFunctionDerivative(double x)
{
	//return (1.0 - x * x);
	return (1.0 - tanh(x) * tanh(x));
}
double Perceptron::randomWeight(void)
{
	return rand() / double(RAND_MAX);
}
double Perceptron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;
	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}