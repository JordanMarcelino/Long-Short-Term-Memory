package unsri.ac.id.lstm;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.activation.ReLU;
import unsri.ac.id.lstm.activation.SoftMax;
import unsri.ac.id.lstm.dataset.Dataset;
import unsri.ac.id.lstm.layers.Dense;
import unsri.ac.id.lstm.loss.LossFunction;
import unsri.ac.id.lstm.loss.MSE;
import unsri.ac.id.lstm.model.Sequential;

import java.util.Arrays;

public class ModelTest {
    private Dataset dataset;

    @BeforeEach
    void setUp() {
        dataset = new Dataset();
        dataset.createSpiralData(100, 3);
    }

    @Test
    void testSequential() {

    }
}
