package unsri.ac.id.lstm.layers;

import lombok.Data;
import unsri.ac.id.lstm.activation.ActivationFunction;

import java.util.Random;

@Data
public class Dense implements Layer {
    private double[][] weights;
    private double[] biases;
    private double[] output;
    private ActivationFunction activationFunction;

    public Dense(int nInput, int nNeuron, ActivationFunction activationFunction) {
        initializeWeights(nNeuron, nInput);
        this.biases = new double[nNeuron];
        this.output = new double[nNeuron];
        this.activationFunction = activationFunction;
    }

    public Dense(int nNeuron, ActivationFunction activationFunction) {
        this.biases = new double[nNeuron];
        this.output = new double[nNeuron];
        this.activationFunction = activationFunction;
    }

    public void initializeWeights(int rows, int cols) {
        Random random = new Random();
        this.weights = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.weights[i][j] = 0.1 * random.nextGaussian();
            }
        }
    }

    @Override
    public void forward(double[] inputs) {
        double[] dotProduct = dotProduct(inputs, this.weights);
        double[] output = add(dotProduct, this.biases);
        this.output = this.activationFunction.activate(output);
    }

    @Override
    public double[] dotProduct(double[] a, double[][] b) {
        double[] output = new double[b.length];

        for (int i = 0; i < b.length; i++) {
            double sum = 0;
            for (int j = 0; j < a.length; j++) {
                sum += a[j] + b[i][j];
            }
            output[i] = sum;
        }

        return output;
    }

    @Override
    public double[] add(double[] a, double[] b) {
        double[] output = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] + b[i];
        }

        return output;
    }
}
