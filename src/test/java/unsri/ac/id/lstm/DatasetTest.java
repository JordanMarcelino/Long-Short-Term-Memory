package unsri.ac.id.lstm;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.dataset.Dataset;

import java.util.Arrays;

public class DatasetTest {
    private Dataset dataset;

    @BeforeEach
    void setUp() {
        this.dataset = new Dataset();
    }

    @Test
    void testSpiralDataset() {
        dataset.createSpiralData(100, 3);

        System.out.println(dataset.getX().length);
        System.out.println(dataset.getX()[0].length);
        System.out.println(dataset.getY().length);

        System.out.println(Arrays.deepToString(dataset.getX()));
        System.out.println(Arrays.toString(dataset.getY()));
    }
}
