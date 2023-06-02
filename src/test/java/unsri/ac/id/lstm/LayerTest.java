package unsri.ac.id.lstm;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.activation.SoftMax;
import unsri.ac.id.lstm.dataset.Dataset;
import unsri.ac.id.lstm.layers.Dense;
import unsri.ac.id.lstm.layers.RNN;

import java.util.Arrays;

@Slf4j
public class LayerTest {

    private Dataset dataset;

    @BeforeEach
    void setUp() {
        dataset = new Dataset();
        dataset.createSpiralData(100, 3);
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
}


