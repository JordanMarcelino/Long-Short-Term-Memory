package unsri.ac.id.lstm.activation;

import java.util.Arrays;

public class SoftMax<T> implements ActivationFunction<T> {
    @Override
    public T activate(T input) {
        if (input instanceof double[]) {
            double[] result = new double[((double[]) input).length];
            double eSum = Arrays.stream((double[]) input).map(Math::exp).sum();

            for (int i = 0; i < ((double[]) input).length; i++) {
                result[i] = Math.exp(((double[]) input)[i]) / eSum;
            }

            return (T) result;
        }

        return null;
    }

    @Override
    public T derivative(T input) {
        if (input instanceof double[]) {
            double[] result = (double[]) activate(input);


            for (int i = 0; i < result.length; i++) {
                result[i] = result[i] * (1 - result[i]);
            }

            return (T) result;
        }

        return null;
    }
}
