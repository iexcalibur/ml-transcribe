import { Tensor, Linear, ReLU, Sigmoid, Conv1d, Conv2d, Parameter, softmax, gelu, dropout, crossEntropyLoss, layerNorm, avgpool2d, maxpool2d, tile, Adam } from '../dist/index.js';

function assert(cond, msg) {
    if (!cond) throw new Error(`FAIL: ${msg}`);
}

function assertClose(a, b, tol = 1e-4, msg = '') {
    if (Math.abs(a - b) > tol) throw new Error(`FAIL ${msg}: ${a} != ${b} (tol=${tol})`);
}

// ---- Tensor creation ----

console.log('Testing tensor creation...');
const z = Tensor.zeros([2, 3]);
assert(z.shape[0] === 2 && z.shape[1] === 3, 'zeros shape');
const zd = z.toFloat32();
assert(zd.every(v => v === 0), 'zeros values');

const o = Tensor.ones([3]);
assert(o.toFloat32().every(v => v === 1), 'ones values');

const r = Tensor.rand([100]);
const rd = r.toFloat32();
assert(rd.every(v => v >= 0 && v <= 1), 'rand range');

// ---- Basic ops ----

console.log('Testing basic ops...');
const a = Tensor.fromFloat32(new Float32Array([1, 2, 3]), [3]);
const b = Tensor.fromFloat32(new Float32Array([4, 5, 6]), [3]);

const sum_ab = a.add(b).toFloat32();
assert(sum_ab[0] === 5 && sum_ab[1] === 7 && sum_ab[2] === 9, 'add');

const diff = a.sub(b).toFloat32();
assert(diff[0] === -3, 'sub');

const prod = a.mul(b).toFloat32();
assert(prod[0] === 4 && prod[1] === 10 && prod[2] === 18, 'mul');

const neg = a.neg().toFloat32();
assert(neg[0] === -1 && neg[1] === -2 && neg[2] === -3, 'neg');

const scaled = a.mul(2).toFloat32();
assert(scaled[0] === 2 && scaled[1] === 4 && scaled[2] === 6, 'mul scalar');

// ---- New ops: div, pow, comparisons ----

console.log('Testing new ops...');
const x = Tensor.fromFloat32(new Float32Array([6, 10, 15]), [3]);
const y = Tensor.fromFloat32(new Float32Array([2, 5, 3]), [3]);

const quotient = x.div(y).toFloat32();
assert(quotient[0] === 3 && quotient[1] === 2 && quotient[2] === 5, 'div');

const divScalar = x.div(2).toFloat32();
assertClose(divScalar[0], 3, 1e-4, 'div scalar');

const powered = a.pow(2).toFloat32();
assertClose(powered[0], 1, 1e-4, 'pow 1^2');
assertClose(powered[1], 4, 1e-4, 'pow 2^2');
assertClose(powered[2], 9, 1e-4, 'pow 3^2');

const ltResult = a.lt(b).toFloat32();
assert(ltResult.every(v => v === 1), 'lt all true');

const gtResult = b.gt(a).toFloat32();
assert(gtResult.every(v => v === 1), 'gt all true');

const eqResult = a.eq(a).toFloat32();
assert(eqResult.every(v => v === 1), 'eq self');

const closeResult = a.isClose(a).toFloat32();
assert(closeResult.every(v => v === 1), 'isClose self');

// ---- Sigmoid ----

console.log('Testing sigmoid...');
const sigInput = Tensor.fromFloat32(new Float32Array([0, 100, -100]), [3]);
const sigOut = sigInput.sigmoid().toFloat32();
assertClose(sigOut[0], 0.5, 1e-4, 'sigmoid(0)');
assertClose(sigOut[1], 1.0, 1e-2, 'sigmoid(100)');
assertClose(sigOut[2], 0.0, 1e-2, 'sigmoid(-100)');

// ---- Exp, Log ----

console.log('Testing exp/log...');
const expData = Tensor.fromFloat32(new Float32Array([0, 1, 2]), [3]);
const expResult = expData.exp().toFloat32();
assertClose(expResult[0], 1.0, 1e-4, 'exp(0)');
assertClose(expResult[1], Math.E, 1e-4, 'exp(1)');

const logData = Tensor.fromFloat32(new Float32Array([1, Math.E, Math.E * Math.E]), [3]);
const logResult = logData.log().toFloat32();
assertClose(logResult[0], 0.0, 1e-4, 'log(1)');
assertClose(logResult[1], 1.0, 1e-4, 'log(e)');

// ---- Reduction ----

console.log('Testing reductions...');
const mat = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const sumDim1 = mat.sum(1).toFloat32();
assertClose(sumDim1[0], 6, 1e-4, 'sum dim=1 row0');
assertClose(sumDim1[1], 15, 1e-4, 'sum dim=1 row1');

const meanDim1 = mat.mean(1).toFloat32();
assertClose(meanDim1[0], 2, 1e-4, 'mean dim=1 row0');
assertClose(meanDim1[1], 5, 1e-4, 'mean dim=1 row1');

const sumAll = mat.sum().toFloat32();
assertClose(sumAll[0], 21, 1e-4, 'sum all');

const meanAll = mat.mean().toFloat32();
assertClose(meanAll[0], 3.5, 1e-4, 'mean all');

// ---- View, Permute ----

console.log('Testing layout ops...');
const reshaped = mat.view(3, 2);
assert(reshaped.shape[0] === 3 && reshaped.shape[1] === 2, 'view shape');

const perm = mat.permute(1, 0);
assert(perm.shape[0] === 3 && perm.shape[1] === 2, 'permute shape');

// ---- MatMul ----

console.log('Testing matmul...');
const m1 = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [2, 2]);
const m2 = Tensor.fromFloat32(new Float32Array([5, 6, 7, 8]), [2, 2]);
const mm = m1.matmul(m2).toFloat32();
assertClose(mm[0], 19, 1e-3, 'matmul [0,0]');
assertClose(mm[1], 22, 1e-3, 'matmul [0,1]');

// ---- Activations ----

console.log('Testing activations...');
const actInput = Tensor.fromFloat32(new Float32Array([-1, 0, 1, 2]), [4]);
const reluOut = actInput.relu().toFloat32();
assert(reluOut[0] === 0 && reluOut[1] === 0 && reluOut[2] === 1 && reluOut[3] === 2, 'relu');

// ---- Linear module ----

console.log('Testing Linear module...');
const linear = new Linear(3, 2);
const linInput = Tensor.rand([2, 3]);
const linOutput = linear.forward(linInput);
assert(linOutput.shape[0] === 2 && linOutput.shape[1] === 2, 'linear output shape');

// ---- Softmax ----

console.log('Testing softmax...');
const smInput = Tensor.fromFloat32(new Float32Array([1, 2, 3, 1, 2, 3]), [2, 3]);
const smOut = softmax(smInput, 1);
const smData = smOut.toFloat32();
assertClose(smData[0] + smData[1] + smData[2], 1.0, 1e-4, 'softmax sums to 1');

// ---- Cross-entropy loss ----

console.log('Testing cross-entropy loss...');
const ceLogits = Tensor.fromFloat32(new Float32Array([2, 1, 0.1, 0.1, 1, 2]), [2, 3]);
const ceTargets = [[0], [2]];
const ceLoss = crossEntropyLoss(ceLogits, ceTargets);
const lossVal = ceLoss.toFloat32()[0];
assert(lossVal > 0 && lossVal < 5, 'cross-entropy loss positive and reasonable');

// ---- Layer norm ----

console.log('Testing layer norm...');
const lnInput = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const gamma = Tensor.ones([3]).setRequiresGrad(true);
const beta = Tensor.zeros([3]).setRequiresGrad(true);
const lnOut = layerNorm(lnInput, gamma, beta);
assert(lnOut.shape[0] === 2 && lnOut.shape[1] === 3, 'layernorm shape');

// ---- Conv1d module ----

console.log('Testing Conv1d...');
const conv1d = new Conv1d(3, 8, 3, 1, 1);
const conv1dInput = Tensor.rand([2, 3, 10]);
const conv1dOut = conv1d.forward(conv1dInput);
assert(conv1dOut.shape[0] === 2 && conv1dOut.shape[1] === 8, 'conv1d output shape');

// ---- Conv2d module ----

console.log('Testing Conv2d...');
const conv2d = new Conv2d(3, 16, 3, 1, 1);
const conv2dInput = Tensor.rand([1, 3, 8, 8]);
const conv2dOut = conv2d.forward(conv2dInput);
assert(conv2dOut.shape[0] === 1 && conv2dOut.shape[1] === 16, 'conv2d output shape');

// ---- Pooling ----

console.log('Testing pooling...');
const poolInput = Tensor.rand([1, 1, 4, 4]);
const avgOut = avgpool2d(poolInput, 2, 2);
assert(avgOut.shape[2] === 2 && avgOut.shape[3] === 2, 'avgpool2d output shape');

const maxOut = maxpool2d(poolInput, 2, 2);
assert(maxOut.shape[2] === 2 && maxOut.shape[3] === 2, 'maxpool2d output shape');

// ---- Tile ----

console.log('Testing tile...');
const tileInput = Tensor.fromFloat32(new Float32Array([1, 2, 3]), [1, 3]);
const tiled = tile(tileInput, [2, 1]);
assert(tiled.shape[0] === 2 && tiled.shape[1] === 3, 'tile shape');
const tiledData = tiled.toFloat32();
assert(tiledData[0] === 1 && tiledData[3] === 1, 'tile values');

// ---- Clone / Detach / toString ----

console.log('Testing utilities...');
const orig = Tensor.fromFloat32(new Float32Array([1, 2, 3]), [3]);
const cloned = orig.clone();
assert(cloned.shape[0] === 3, 'clone shape');
assert(cloned.toFloat32()[0] === 1, 'clone values');

const detached = orig.detach();
assert(detached.shape[0] === 3, 'detach shape');

const str = orig.toString();
assert(str.includes('Tensor'), 'toString contains Tensor');
assert(str.includes('3'), 'toString contains shape');

// ---- Backward pass ----

console.log('Testing backward...');
const paramA = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [2, 2]).setRequiresGrad(true);
const paramB = Tensor.fromFloat32(new Float32Array([5, 6, 7, 8]), [2, 2]).setRequiresGrad(true);
const c = paramA.mul(paramB).sum(0).sum(0);
c.backward();
const gradA = paramA.grad;
assert(gradA !== null, 'gradient exists');

// ---- Optimizer ----

console.log('Testing Adam optimizer...');
const paramTensor = Tensor.fromFloat32(new Float32Array([1, 2, 3, 4]), [2, 2]).setRequiresGrad(true);
const paramObj = new Parameter(paramTensor);
const target = Tensor.zeros([2, 2]);
const loss = paramObj.value.sub(target).pow(2).mean(0).mean(0);
loss.backward();
const adam = new Adam([paramObj], 0.01);
adam.step();
adam.zeroGrad();

console.log('\n✅ All tests passed!');
