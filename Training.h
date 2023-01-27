#pragma once

#include <sstream>
#include <fstream>
#include <vector>

class Training
{
public:
	Training(const std::string filename);
	~Training();
	bool isEof(void);
	void getTopology(std::vector<unsigned>& topology);

	// Returns the number of input values read from the file:
	unsigned getNextInputs(std::vector<double>& inputVals);
	unsigned getTargetOutputs(std::vector<double>& targetOutputVals);

private:
	std::ifstream m_trainingDataFile;
};