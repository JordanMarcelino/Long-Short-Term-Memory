package unsri.ac.id.lstm.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import unsri.ac.id.lstm.activation.ActivationFunction;
import unsri.ac.id.lstm.initialization.InitializationFunction;
import unsri.ac.id.lstm.initialization.RandomInitializer;
import unsri.ac.id.lstm.utils.Utils;

@EqualsAndHashCode(callSuper = true)
@Data
public class Dense<T> extends Layer<T> {

    public Dense(int nInput, int nNeuron, ActivationFunction activationFunction, InitializationFunction initializationFunction) {
        this.biases = new double[nNeuron];
        this.activationFunction = activationFunction;
        this.initializationFunction = initializationFunction;
        initializeWeights(nNeuron, nInput);
    }

    public Dense(int nInput, int nNeuron, ActivationFunction activationFunction) {
        this.biases = new double[nNeuron];
        this.activationFunction = activationFunction;
        this.initializationFunction = new RandomInitializer();
        initializeWeights(nNeuron, nInput);
    }

    public Dense(int nNeuron, ActivationFunction activationFunction, InitializationFunction initializationFunction) {
        this.biases = new double[nNeuron];
        this.activationFunction = activationFunction;
        this.initializationFunction = initializationFunction;
    }

    public Dense(int nNeuron, ActivationFunction activationFunction) {
        this.biases = new double[nNeuron];
        this.activationFunction = activationFunction;
        this.initializationFunction = new RandomInitializer();
    }

    public void initializeWeights(int rows, int cols) {
        this.weights = this.initializationFunction.initialize(rows, cols);
    }

    @Override
    public void forward(T inputs) {
        if (inputs instanceof double[]) {
            double[] dotProduct = Utils.dotProduct((double[]) inputs, this.weights);
            double[] output = Utils.add(dotProduct, this.biases);
            this.output = (T) this.activationFunction.activate(output);
        } else if (inputs instanceof double[][]) {
            double[][] dotProduct = Utils.dotProduct((double[][]) inputs, this.weights);
            double[][] output = Utils.add(dotProduct, this.biases);

            for (int i = 0; i < output.length; i++) {
                output[i] = this.activationFunction.activate(output[i]);
            }

            this.output = (T) output;
        }
    }

}
