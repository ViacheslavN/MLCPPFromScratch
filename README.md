# MLCPPFromScratch

Machine learning algorithms implemented from scratch in C++, for education.
The project follows three courses: supervised learning, unsupervised learning
and deep learning. Nothing is taken from an ML or linear algebra library: the
matrix code, the data pipeline, the optimizers and the models are all written here.

## Layout

| Folder | Contents |
|---|---|
| `CommonLib` | Shared utilities: streams, files, strings, logging, exceptions (copied from LanguageTrainer) |
| `NumericsCore` | Vectors, matrices, tensors, linear algebra, random numbers, statistics |
| `DataPipeline` | CSV reader, preprocessing, train/test split, k-fold, batching |
| `Optimization` | Gradient descent, SGD, momentum, Adam, learning-rate schedules |
| `Evaluation` | Loss functions, metrics, cross-validation |
| `SupervisedLib` | Linear, ridge, lasso and logistic regression, k-NN, Naive Bayes, decision tree, random forest, gradient boosting, SVM |

The other model libraries (unsupervised, deep learning) are added next to these as they are written.

## Plan

1. Numerics core: vectors, matrices, linear algebra, random numbers
2. Data pipeline: CSV reader, encoding, scaling, train/test split, batching
3. Optimization: gradient descent, SGD, momentum, Adam
4. Evaluation: loss functions, metrics, cross-validation
5. Algorithms:
   - Supervised: linear and logistic regression, k-NN, Naive Bayes, decision trees, random forest, SVM
   - Unsupervised: K-means, hierarchical clustering, DBSCAN, GMM, PCA
   - Deep learning: autograd, dense and convolution layers, optimizers, regularization
6. Infrastructure: gradient checking, tests against NumPy / scikit-learn, model serialization

## Build

Requirements:

- CMake 3.17 or newer
- A C++17 compiler (MSVC, GCC or Clang)
- OpenSSL (required by `CommonLib`)

### Windows (MSVC + vcpkg)

```
vcpkg install openssl:x64-windows-static
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

The project links the static CRT (`/MT`, `/MTd`) and uses the
`x64-windows-static` vcpkg triplet by default.

### Linux

```
sudo apt install libssl-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `MLCPP_BUILD_TESTS` | `ON` | Build the tests of the ML libraries |
| `GRAPHENGINE_BUILD_TESTS` | `OFF` | Build the `CommonLib` tests (need Catch2 v3) |
