package unsri.ac.id.lstm;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.activation.ReLU;
import unsri.ac.id.lstm.activation.SoftMax;
import unsri.ac.id.lstm.dataset.Dataset;
import unsri.ac.id.lstm.initialization.XavierGlorotInitializer;
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
        Sequential<double[]> sequential = new Sequential<>();
        sequential.add(new Dense<>(300, 16, new ReLU(), new XavierGlorotInitializer()));
        sequential.add(new Dense<>(16, new ReLU(), new XavierGlorotInitializer()));
        sequential.add(new Dense<>(16, new SoftMax(), new XavierGlorotInitializer()));

        for (Layer<double[]> l : sequential.getLayers()) {
            System.out.println(l);
        }
    }
}
