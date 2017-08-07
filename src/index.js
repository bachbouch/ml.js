import logisticRegression from './LogisticRegression';
import LinearRegression from './LinearRegression';


let
    num_features = 10,
    num_samples = 10000,
    thetas = [], //[2.289, 11.788, 23.2, 1.4, 45.4, 3.5, 65.2, 12.5, 6, 12.4, 7.3],
    x = [],
    y = [];

for (let i = 0; i < num_features + 1; i++) {
    thetas.push((Math.random() * 10).toFixed(4) - 0);
}
console.log(thetas);

for (let i = 0; i < num_samples; i += 1) {
    let _x = [];
    for (let k = 0; k < num_features; k++)
        _x.push(Math.random().toFixed(4) - 0);
    x.push(_x);

    let _y = 0;
    for (let j = 0; j < num_features; j++) {
        _y += thetas[j] * _x[j];
    }
    _y += thetas[num_features];

    y.push(_y);

}

console.time('tester');
console.log(LinearRegression.converge(x, y));
console.timeEnd('tester');