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
}
