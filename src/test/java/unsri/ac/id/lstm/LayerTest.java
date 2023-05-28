package unsri.ac.id.lstm;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.activation.ReLU;
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
        Dense dense = new Dense(dataset.getX()[0].length, 16, new ReLU());
        dense.forward(dataset.getX());
        System.out.println(Arrays.deepToString(dense.getOutput()));
        System.out.println(dense.getOutput().length);
        System.out.println(dense.getOutput()[0].length);
    }
}
