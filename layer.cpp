#pragma once
#include <vector>
#include "Neuron.cpp"

class Layer{
private:
	std::vector<Neuron> neuronas;
public:
	Layer(size_t nin, size_t nout, std::mt19937& gen){
	    neuronas.reserve(nout);
	    for(size_t i = 0; i<nout; ++i){
		neuronas.emplace_back(nin, gen);
}//endfor 
}//endConstrcutor
	std::vector<ValuePtr> operator()(const std::vector<ValuePtr> & x){
	   std::vector<ValuePtr> out;

	   for(auto& aux : neuronas){
		out.emplace_back(aux(x));
}//endfor
	   return out;
}//endoperator
std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> params;
        for (auto& n : neuronas) {
            auto neuron_params = n.parameters();
            // Truco de C++ para concatenar vectores rápidamente
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }

};//endPublic

