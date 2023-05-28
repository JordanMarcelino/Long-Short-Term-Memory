package unsri.ac.id.lstm.layers;

import lombok.Data;
import unsri.ac.id.lstm.activation.ActivationFunction;

import java.util.Random;

@Data
public class Dense implements Layer {
    private double[][] weights;
    private double[] biases;
    private double[][] output;
    private ActivationFunction activationFunction;

    public Dense(int nInput, int nNeuron, ActivationFunction activationFunction) {
        initializeWeights(nInput, nNeuron);
        this.biases = new double[nNeuron];
        this.activationFunction = activationFunction;
    }

    public Dense(int nNeuron, ActivationFunction activationFunction) {
        this.biases = new double[nNeuron];
        this.activationFunction = activationFunction;
    }

    private void initializeWeights(int rows, int cols){
        Random random = new Random();
        this.weights = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.weights[i][j] = 0.1 * random.nextGaussian();
            }
        }
    }

    @Override
    public void forward(double[][] inputs) {
        double[][] dotProduct = dotProduct(inputs, this.weights);
        this.output = add(dotProduct, biases);

        for (int i = 0; i < this.output.length; i++) {
            this.output[i] = this.activationFunction.activate(this.output[i]);
        }
    }

    @Override
    public double[][] dotProduct(double[][] a, double[][] b) {
        double[][] output = new double[a.length][b[0].length];

        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                double value = 0;

                for (int k = 0; k < a[0].length; k++) {
                    value += a[i][k] * b[k][j];
                }

                output[i][j] = value;
            }
        }


        return output;
    }

    @Override
    public double[][] add(double[][] a, double[] b) {
        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                output[i][j] = a[i][j] + b[j];
            }
        }

        return output;
    }
}
