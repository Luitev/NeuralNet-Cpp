#pragma once
#include <vector>


class Perceptron

{
public:
	Perceptron(unsigned numOutputs, unsigned myIndex);
	~Perceptron();
	typedef std::vector<Perceptron> Layer;
	struct Connection
	{
		double weight;
		double deltaWeight;
	};
	void setOutputVal(double val);
	double getOutputVal(void) const;
	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);



private:
	static double eta;	// [0.0 .. 1.0] training rate
	static double alpha;// [0.0 .. n] multiplier of last weight change
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void);
	double sumDOW(const Layer& nextLayer) const;
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};