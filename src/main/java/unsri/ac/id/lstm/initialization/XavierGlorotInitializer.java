package unsri.ac.id.lstm.initialization;

import java.util.Random;

public class XavierGlorotInitializer implements InitializationFunction{
    @Override
    public double[][] initialize(int rows, int cols) {
        Random random = new Random();
        double[][] output = new double[rows][cols];

        double limit = Math.sqrt(2.0 / (rows + cols));
        double lowerLimit = limit * -1;


        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][j] = lowerLimit + random.nextDouble() * (limit - lowerLimit);
            }
        }

        return output;
    }

}
