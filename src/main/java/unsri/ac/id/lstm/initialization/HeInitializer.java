package unsri.ac.id.lstm.initialization;

import java.util.Random;

public class HeInitializer implements InitializationFunction{
    @Override
    public double[][] initialize(int rows, int cols) {
        Random random = new Random();
        double[][] output = new double[rows][cols];

        double stdDev = Math.sqrt(2.0 / rows);


        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][j] = random.nextGaussian() * stdDev;
            }
        }

        return output;
    }
}
