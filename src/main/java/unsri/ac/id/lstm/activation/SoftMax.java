package unsri.ac.id.lstm.activation;

import java.util.Arrays;

public class SoftMax implements ActivationFunction{
    @Override
    public double[] activate(double[] input) {
        double[] result = new double[input.length];
        double eSum = Arrays.stream(input).map(Math::exp).sum();

        for (int i = 0; i < input.length; i++) {
            result[i] = Math.exp(input[i]) / eSum;
        }

        return result;
    }

    @Override
    public double[] derivative(double[] input) {
        double[] result = activate(input);


        for (int i = 0; i < result.length; i++) {
            result[i] = result[i] * (1 - result[i]);
        }

        return result;
    }
}
