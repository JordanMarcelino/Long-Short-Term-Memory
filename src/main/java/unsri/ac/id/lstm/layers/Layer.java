package unsri.ac.id.lstm.layers;

import lombok.Data;
import unsri.ac.id.lstm.activation.ActivationFunction;
import unsri.ac.id.lstm.initialization.InitializationFunction;

@Data
public abstract class Layer<T> {
    protected double[][] weights;
    protected double[] biases;
    protected ActivationFunction<T> activationFunction;
    protected InitializationFunction initializationFunction;
    protected T output;

    public abstract void forward(T inputs);
}
