package unsri.ac.id.lstm;

import org.junit.jupiter.api.Test;
import unsri.ac.id.lstm.utils.Utils;

import java.util.Arrays;
import java.util.Random;

public class UtilsTest {

    @Test
    void testTranspose1DArrayInput() {
        double[] input = {-0.47770861, -0.07590197, -0.60530582, -0.89787035, 0.48763788};

        System.out.println(Arrays.deepToString(Utils.transpose(input)));
    }

    @Test
    void testTransposeSingleInput() {
        double input = 2.9;

        System.out.println(Arrays.deepToString(Utils.transpose(input)));
    }

    @Test
    void testEditArray() {
        double[][] input = {
                {1,1,1},
                {1,1,1}
        };

        editArray(input);
        System.out.println(Arrays.deepToString(input));
    }

    void editArray(double[][] input){
        input = Utils.multiply(input, 2);
    }
}
