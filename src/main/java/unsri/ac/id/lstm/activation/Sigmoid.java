package unsri.ac.id.lstm.activation;

public class Sigmoid implements Activation{
    @Override
    public double activate(double input) {
        return Math.exp(input) / (Math.exp(input) + 1);
    }
}
