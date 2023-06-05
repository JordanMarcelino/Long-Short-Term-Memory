package unsri.ac.id.lstm;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.activation.Sigmoid;
import unsri.ac.id.lstm.activation.SoftMax;
import unsri.ac.id.lstm.dataset.Dataset;
import unsri.ac.id.lstm.initialization.UniformXavierGlorotInitializer;
import unsri.ac.id.lstm.layers.Dense;
import unsri.ac.id.lstm.layers.RNN;
import unsri.ac.id.lstm.loss.LossFunction;
import unsri.ac.id.lstm.loss.MSE;
import unsri.ac.id.lstm.utils.Utils;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

@Slf4j
public class LayerTest {

    private Dataset dataset;
    private double[][] trainX;
    private double[][] trainY;
    private double[][] validX;
    private double[][] validY;

    private static double parseDouble(String value) {
        if (value == null || value.isEmpty()) {
            return 0.0;
        }
        return Double.parseDouble(value);
    }

    @BeforeEach
    void setUp() {
        dataset = new Dataset();
        dataset.createSpiralData(100, 3);

        Path path = Path.of("src", "main", "resources", "clean_weather.csv");
        try (Reader reader = Files.newBufferedReader(path)) {
            try (CSVParser parser = new CSVParser(reader,
                    CSVFormat.DEFAULT.builder().setHeader().build())) {
                trainX = new double[5000][3];
                trainY = new double[5000][1];

                List<CSVRecord> records = parser.getRecords();
                for (int i = 0; i < 5000; i++) {
                    trainX[i][0] = parseDouble(records.get(i).get("tmax"));
                    trainX[i][1] = parseDouble(records.get(i).get("tmin"));
                    trainX[i][2] = parseDouble(records.get(i).get("rain"));
                    trainY[i][0] = parseDouble(records.get(i).get("tmax_tomorrow"));
                }

                validX = new double[2000][3];
                validY = new double[2000][1];

                for (int i = 5001; i < 7000; i++) {
                    validX[i - 5001][0] = parseDouble(records.get(i).get("tmax"));
                    validX[i - 5001][1] = parseDouble(records.get(i).get("tmin"));
                    validX[i - 5001][2] = parseDouble(records.get(i).get("rain"));
                    validY[i - 5001][0] = parseDouble(records.get(i).get("tmax_tomorrow"));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Utils.scale(trainX);
        Utils.scale(validX);
    }

    @Test
    void testDense() {
        double[] input = Arrays.stream(dataset.getX()).flatMapToDouble(Arrays::stream).toArray();
        System.out.println(Arrays.toString(input));

        Dense<double[]> dense = new Dense<>(input.length, 4, new SoftMax<>());

        dense.forward(input);

        System.out.println(Arrays.toString(dense.getOutputAfterActivation()));
        System.out.println(Arrays.stream(dense.getOutputAfterActivation()).sum());
    }

    @Test
    void testRNN2DArrayInput() {
        double[][] input = {
                {-0.72725587, -2.27150212, -0.25366126},
                {-1.68779357, -1.6825982, -0.25366126},
                {-1.68779357, -2.27150212, -0.25366126},
                {-1.56772636, -2.12427614, -0.25366126},
                {-1.68779357, -2.27150212, -0.25366126},
                {-1.92792799, -1.82982418, -0.25366126},
                {-1.68779357, -1.09369428, -0.2536612}
        };
        RNN<double[][]> rnn = new RNN<>(3, 4, 1);

        rnn.forward(input);
        System.out.println(Arrays.deepToString(rnn.getWeights()));
        System.out.println(Arrays.deepToString(rnn.getWeightsHidden()));
        System.out.println(Arrays.deepToString(rnn.getWeightsOutput()));
        System.out.println(Arrays.deepToString(rnn.getHidden()));
        System.out.println(Arrays.deepToString(rnn.getOutput()));
    }

    @Test
    void testRNN1DArrayInput() {
        double[] input = {-0.47770861, -0.07590197, -0.60530582, -0.89787035, 0.48763788, -0.17473821, -0.08713886, -0.70513484, -0.85287193, 0.72443911};
        RNN<double[]> rnn = new RNN<>(1, 4, 1);

        rnn.forward(input);
        System.out.println(Arrays.deepToString(rnn.getWeights()));
        System.out.println(Arrays.deepToString(rnn.getWeightsHidden()));
        System.out.println(Arrays.deepToString(rnn.getWeightsOutput()));
        System.out.println(Arrays.deepToString(rnn.getHidden()));
        System.out.println(Arrays.deepToString(rnn.getOutput()));
    }

    @Test
    void testRNNWithCSVInput() {
        RNN<double[][]> rnn = new RNN<>(3, 4, 1, new UniformXavierGlorotInitializer());
//        rnn.setActivationFunction(new Sigmoid<>());
        LossFunction mse = new MSE();
        int epoch = 251;

        System.out.println(Arrays.deepToString(rnn.getWeights()));
        System.out.println(Arrays.deepToString(rnn.getWeightsHidden()));
        System.out.println(Arrays.deepToString(rnn.getWeightsOutput()));

        for (int i = 0; i < epoch; i++) {
            int seqLen = 7;
            double epochLoss = 0;

            for (int j = 0; j < trainX.length - seqLen; j++) {
                double[][] seqX = new double[seqLen][3];
                double[][] seqY = new double[seqLen][1];
                System.arraycopy(trainX, j, seqX, 0, seqLen);
                System.arraycopy(trainY, j, seqY, 0, seqLen);

                rnn.forward(seqX);

                double[] lossGrad = mse.derivative(
                        Arrays.stream(seqY).flatMapToDouble(Arrays::stream).toArray(),
                        Arrays.stream(rnn.getOutput()).flatMapToDouble(Arrays::stream).toArray()
                );
                rnn.backward(seqX, lossGrad);
                epochLoss += mse.call(
                        Arrays.stream(seqY).flatMapToDouble(Arrays::stream).toArray(),
                        Arrays.stream(rnn.getOutput()).flatMapToDouble(Arrays::stream).toArray()
                );
            }

            if (i % 50 == 0) {
                double validLoss = 0;

                for (int j = 0; j < validX.length - seqLen; j++) {
                    double[][] seqX = new double[seqLen][3];
                    double[][] seqY = new double[seqLen][1];
                    System.arraycopy(validX, j, seqX, 0, seqLen);
                    System.arraycopy(validY, j, seqY, 0, seqLen);

                    rnn.forward(seqX);
                    validLoss += mse.call(
                            Arrays.stream(seqY).flatMapToDouble(Arrays::stream).toArray(),
                            Arrays.stream(rnn.getOutput()).flatMapToDouble(Arrays::stream).toArray()
                    );
                }

                System.out.println("Epoch: " + i + " train loss " + (epochLoss / trainX.length) + ", valid loss: " + (validLoss / validX.length));
            }
        }

        System.out.println(Arrays.deepToString(rnn.getWeights()));
        System.out.println(Arrays.deepToString(rnn.getWeightsHidden()));
        System.out.println(Arrays.deepToString(rnn.getWeightsOutput()));

        rnn.forward(trainX);
        System.out.println(Arrays.deepToString(trainY));
        System.out.println(Arrays.deepToString(rnn.getOutput()));
    }
}


