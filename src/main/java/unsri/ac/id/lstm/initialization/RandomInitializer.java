package unsri.ac.id.lstm.initialization;

import java.util.Random;

public class RandomInitializer implements InitializationFunction{
    @Override
    public double[][] initialize(int rows, int cols) {
        Random random = new Random();
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][j] = 0.1 * random.nextGaussian();
            }
        }

        return output;
    }
}
