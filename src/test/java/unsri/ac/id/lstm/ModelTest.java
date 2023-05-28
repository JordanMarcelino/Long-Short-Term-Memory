package unsri.ac.id.lstm;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.activation.ReLU;
import unsri.ac.id.lstm.activation.Sigmoid;
import unsri.ac.id.lstm.dataset.Dataset;
import unsri.ac.id.lstm.layers.Dense;
import unsri.ac.id.lstm.layers.Layer;
import unsri.ac.id.lstm.model.Sequential;

public class ModelTest {
    private Dataset dataset;

    @BeforeEach
    void setUp() {
        dataset = new Dataset();
        dataset.createSpiralData(100, 3);
    }

    @Test
    void testSequential() {
        Sequential sequential = new Sequential();
        sequential.add(new Dense(2, 16, new ReLU()));
        sequential.add(new Dense(2, 16, new ReLU()));
        sequential.add(new Dense(2, 16, new ReLU()), new Dense(3, 16, new Sigmoid()));
        for (Layer layer : sequential.getLayers()) {
            System.out.println(layer);
        }
    }

}
