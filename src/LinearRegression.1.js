const LinearRegression = function() {

    //Theta*X + B = Y,
    const LEARNING_RATE = 0.0001;
    const MAX_ITERATIONS = 10000000;

    const computeError = (b, m, x, y) => {
        let totalError = 0;
        const len = x.length;
        for (let i = 0; i < len; i++) { //change this with reduce
            totalError += Math.pow(y[i] - (m * x[i] + b), 2);
        }
        return totalError / (2 * len);
    };

    function stepGradient(X, Y, a_current, b_current, learningRate = LEARNING_RATE) {
        let b_gradient = 0,
            a_gradient = 0,
            len = X.length,
            new_b, new_a;

        for (let i = 0; i < len; i++) {
            let temp = (a_current * X[i] + b_current) - Y[i];
            b_gradient += (1 / len) * temp;
            a_gradient += (1 / len) * temp * X[i];
        }
        b_current -= (learningRate * b_gradient);
        a_current -= (learningRate * a_gradient);

        return [b_current, a_current]
    }

    return {
        converge: (X, Y, maxIterations = MAX_ITERATIONS) => {
            let a;
            let b;
            let converged_a = Math.random();
            let converged_b = Math.random();
            let count = 0;
            do {
                count++;
                if (count > maxIterations)
                    return console.log("Maximized the number of iterations without conversion")

                a = converged_a;
                b = converged_b;
                let converged = stepGradient(X, Y, a, b);
                converged_a = converged[1];
                converged_b = converged[0];

            }
            while (Math.abs(a - converged_a) > 0.00000000000001 && Math.abs(b - converged_b) > 0.00000000000001)

            let roundedA = parseFloat(a.toFixed(6))
            let roundedB = parseFloat(b.toFixed(6))

            return [roundedA, roundedB];
        }
    };
};

export default LinearRegression();