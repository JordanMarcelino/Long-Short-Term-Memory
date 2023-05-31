package unsri.ac.id.lstm.utils;

public class Utils {

    public static double[] multiplyElementWise(double[] a, double[] b) {
        double[] output = new double[a.length];

        for(int i = 0; i < a.length; i++) {
            output[i] = a[i] * b[i];
        }

        return output;
    }

    public static double[] dotProduct(double[] a, double[][] b) {
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

    public static double[][] dotProduct(double[][] a, double[][] b) {
        double[][] output = new double[a.length][b[0].length];

        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                double sum = 0;

                for (int k = 0; k < a[0].length; k++) {
                    sum += a[i][k] + b[k][j];
                }

                output[i][j] = sum;
            }
        }

        return output;
    }

    public static double[] add(double[] a, double[] b) {
        double[] output = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] + b[i];
        }

        return output;
    }

    public static double[][] add(double[][] a, double[] b) {
        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                output[i][j] = a[i][j] + b[j];
            }
        }

        return output;
    }

    public static double[][] transpose(double[][] input){
        double[][] output = new double[input[0].length][input.length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[j][i] = input[i][j];
            }
        }

        return output;
    }
}
