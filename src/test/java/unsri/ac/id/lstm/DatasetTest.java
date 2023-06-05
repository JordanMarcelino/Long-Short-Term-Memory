package unsri.ac.id.lstm;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.dataset.Dataset;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
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

    @Test
    void readCsvFiles() throws IOException {
        Path path = Path.of("src", "main", "resources", "clean_weather.csv");
        Reader reader = Files.newBufferedReader(path);

        CSVParser parser = new CSVParser(reader, CSVFormat.DEFAULT.builder().setHeader().build());

        for (CSVRecord record : parser){
            System.out.println(record.get("tmax"));
            System.out.println(record.get("tmin"));
            System.out.println(record.get("rain"));
        }

        parser.close();
        reader.close();
    }
}
