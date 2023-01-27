#pragma once

#include <iostream>
#include "Perceptron.h"
#include <assert.h>

typedef std::vector<Perceptron> Layer;
class Network
{
public:
	Network(const std::vector<unsigned>& topology);
	~Network();
	void feedForward(const std::vector<double>& inputVals);
	void backProp(const std::vector<double>& targetVals);
	void getResults(std::vector<double>& resultsVals) const;
	double getRecentAverageError(void) const;

private:
	std::vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};