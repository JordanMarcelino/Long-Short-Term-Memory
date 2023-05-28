package unsri.ac.id.lstm.dataset;

import lombok.Getter;

import java.util.Random;

@Getter
public class Dataset {
    private double[][] X;
    private int[] Y;
    private final Random random = new Random();

    public void createSpiralData(int points, int classes) {
        X = new double[points * classes][2];
        Y = new int[points * classes];
        int ix = 0;
        for (int classNumber = 0; classNumber < classes; classNumber++) {
            double r = 0;
            double t = classNumber * 4;
            while (r <= 1 && t <= (classNumber + 1) * 4) {
                double randomT = t + random.nextInt(points) * 0.2;
                X[ix][0] = r * Math.sin(randomT * 2.5);
                X[ix][1] = r * Math.cos(randomT * 2.5);
                Y[ix] = classNumber;
                r += 1.0 / (points - 1);
                t += 4.0 / (points - 1);
                ix++;
            }
        }
    }
}
