var v = require('vectorious'),
    Matrix = v.Matrix,
    Vector = v.Vector,
    BLAS = v.BLAS; // access BLAS routines

var A = new Matrix([
        [1],
        [2],
        [3]
    ]),
    B = new Matrix([
        [1, 3, 5]
    ]),
    C = A.multiply(B);

console.log('C:', C.toArray());

import logisticRegression from './LogisticRegression';
console.log(logisticRegression.fit);