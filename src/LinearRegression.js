const LinearRegression = function() {

    //Theta*X + B = Y,
    let LEARNING_RATE;
    const MAX_ITERATIONS = 10000000;

    const computeError = (thetas, x, y) => {
        let totalError = 0;
        const len = x.length;
        for (let i = 0; i < len; i++) {
            let t = (y[i] - activation(thetas, x[i]));
            totalError += t * t;
            if (totalError > 1)
                return totalError;
        }
        return totalError / (2 * len);
    };

    const activation = (thetas, X) => {
        let len = thetas.length,
            sum = 0;
        for (let i = 0; i < len; i++) { // This is much faster than mapreduce or Vector.dot
            sum += thetas[i] * X[i];
        }
        return sum;
    };

    let len,
        len_,
        x_len,
        gradients = [];


    const stepGradient = (X, Y, thetas, learningRate = LEARNING_RATE) => {

        // console.time('test'); 
        // console.timeEnd('test');
        // 	var g_counter = 0;
        //     var len,
        //         len_,
        //         x_len,
        //         gradients = [];
        //             len = X.length;
        //             len_ = 1 / len;
        // 			x_len = X[0].length;

        len = X.length,
            len_ = 1 / len,
            x_len = X[0].length;


        for (let i = 0; i < x_len; i++) {
            gradients[i] = 0; // this is faster than Array.prototype.fill
        }

        for (let i = 0; i < len; i++) {
            let temp = activation(thetas, X[i]) - Y[i],
                x_i = X[i]; // this made the algorithm 10x faster

            for (let j = 0; j < x_len; j++) {
                gradients[j] += temp * x_i[j];
            }
        }
        for (let j = 0; j < x_len; j++) {
            thetas[j] -= (learningRate * gradients[j]);
        }
        // console.log(thetas)
        // return thetas;
    };


    return {
        converge: (X, Y, learningRate = LEARNING_RATE, maxIterations = MAX_ITERATIONS) => {

            let thetas = [];
            for (let i = 0; i < X[0].length + 1; i++)
                thetas.push(Math.random())

            let count = 0,
                len = X.length;

            for (let i = 0; i < len; i++)
                X[i].push(1);


            if (learningRate == LEARNING_RATE)
                LEARNING_RATE = 1 / Math.pow(10, (len * X[0].length).toString().split("").length - 1);
            else
                LEARNING_RATE = learningRate;

            let count_check = Math.max(10, Math.pow(10, (len * X[0].length).toString().split("").length - 3));
            let error = 0;
            while (count < maxIterations) {
                count++;
                stepGradient(X, Y, thetas);
                if (count % count_check == 0) {
                    error = computeError(thetas, X, Y);
                    if (error < 0.01)
                        return thetas.map(x => x.toFixed(4));
                    if (error > 9999999)
                        return console.log("INFINITY ERROR ")
                }

            }

            return console.log("Maximized the number of iterations without conversion");
        }
    };
};

export default LinearRegression();