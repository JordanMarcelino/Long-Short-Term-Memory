package unsri.ac.id.lstm;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.activation.SoftMax;
import unsri.ac.id.lstm.dataset.Dataset;
import unsri.ac.id.lstm.layers.Dense;

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

        Dense<double[]> dense = new Dense<>(input.length, 4, new SoftMax());

        dense.forward(input);

        System.out.println(Arrays.toString(dense.getOutput()));
        System.out.println(Arrays.stream(dense.getOutput()).sum());
    }
}
