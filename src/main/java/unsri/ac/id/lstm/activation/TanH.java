package unsri.ac.id.lstm.activation;

public class TanH<T> implements ActivationFunction<T> {
    @Override
    public T activate(T input) {
        if (input instanceof double[]) {
            double[] result = new double[((double[]) input).length];

            for (int i = 0; i < ((double[]) input).length; i++) {
                result[i] = Math.tanh(((double[]) input)[i]);
            }

            return (T) result;
        } else if (input instanceof double[][]) {
            double[][] result = new double[((double[][]) input).length][((double[][]) input)[0].length];

            for (int i = 0; i < ((double[][]) input).length; i++) {
                for (int j = 0; j < ((double[][]) input)[0].length; j++) {
                    result[i][j] = Math.tanh(((double[][]) input)[i][j]);
                }
            }

            return (T) result;
        }

        return null;
    }

    @Override
    public T derivative(T input) {
        if (input instanceof double[]) {
            double[] result = new double[((double[]) input).length];

            for (int i = 0; i < ((double[]) input).length; i++) {
                result[i] = 1.0 - (Math.pow(Math.tanh(((double[]) input)[i]), 2));
            }

            return (T) result;
        }else if (input instanceof double[][]){
            double[][] result = new double[((double[][]) input).length][((double[][]) input)[0].length];

            for (int i = 0; i < ((double[][]) input).length; i++) {
                for (int j = 0; j < ((double[][]) input)[0].length; j++) {
                    result[i][j] = 1.0 - (Math.pow(Math.tanh(((double[][]) input)[i][j]), 2));
                }
            }

            return (T) result;
        }

        return null;
    }
}
