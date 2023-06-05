package unsri.ac.id.lstm.utils;

import java.util.Arrays;

public class Utils {

    public static double[] multiplyElementWise(double[] a, double[] b) {
        double[] output = new double[a.length];

        for (int i = 0; i < a.length; i++) {
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

    public static double[][] add(double[][] a, double[][] b) {
        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                output[i][j] = a[i][j] + b[i][j];
            }
        }

        return output;
    }


    public static double[][] add(double[][] a, double b) {
        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                output[i][j] = a[i][j] + b;
            }
        }

        return output;
    }

    public static double[] subtract(double[] a, double[] b) {
        double[] output = new double[a.length];

        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] - b[i];
        }

        return output;
    }

    public static double[][] subtract(double[][] a, double[] b) {
        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                output[i][j] = a[i][j] - b[j];
            }
        }

        return output;
    }

    public static double[][] subtract(double[][] a, double[][] b) {
        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                output[i][j] = a[i][j] - b[i][j];
            }
        }

        return output;
    }

    public static double[][] multiply(double[][] a, double[][] b) {
        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                output[i][j] = a[i][j] * b[i][j];
            }
        }

        return output;
    }

    public static double[][] multiply(double[][] a, double b) {
        double[][] output = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                output[i][j] = a[i][j] * b;
            }
        }

        return output;
    }

    public static double[][] transpose(double[][] input) {
        double[][] output = new double[input[0].length][input.length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[j][i] = input[i][j];
            }
        }

        return output;
    }

    public static double[][] transpose(double[] input) {
        double[][] output = new double[1][input.length];

        System.arraycopy(input, 0, output[0], 0, input.length);

        return output;
    }

    public static double[][] transpose(double input) {
        return new double[][]{{input}};
    }

    public static void scale(double[][] input) {
        for (int i = 0; i < input[0].length; i++) {
            double[] data = new double[input.length];
            for (int j = 0; j < input.length; j++) {
                data[j] = input[j][i];
            }

            double sum = Arrays.stream(data).sum();
            double mean = sum / data.length;

            double squaredDiffSum = Arrays.stream(data)
                    .map(d -> (d - mean) * (d - mean))
                    .sum();
            double standardDeviation = Math.sqrt(squaredDiffSum / data.length);

            data = Arrays.stream(data).map(d -> (d - mean) / standardDeviation).toArray();

            for (int j = 0; j < input.length; j++) {
                input[j][i] = data[j];
            }
        }

    }


    public static double[] scale(double[] input) {
        double[] output = new double[input.length];

        double sum = Arrays.stream(output).sum();
        double mean = sum / output.length;

        double squaredDiffSum = Arrays.stream(output)
                .map(d -> (d - mean) * (d - mean))
                .sum();
        double standardDeviation = Math.sqrt(squaredDiffSum / output.length);

        output = Arrays.stream(output).map(d -> (d - mean) / standardDeviation).toArray();

        return output;
    }
}
